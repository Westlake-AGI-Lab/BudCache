import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import lpips
import torch
from PIL import Image
from accelerate import Accelerator
from diffusers import AutoencoderKL, FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import yaml

from src.flux.config import TrainingConfig
from src.flux.sampling import get_schedule, prepare, unpack, vae_decode
from src.flux.scheduler_trainer import (
    FluxRunner,
    LearnableSigmasV1,
    LearnableSigmasV2,
    LearnableSigmasV2_offset,
    transformer_cache_forward,
)
from src.train_util import (
    append_image_panel,
    build_teacher_student_panel,
    make_png2gif,
    plot_grad_norm_curve,
    plot_loss_curve,
    plot_sigma_diff_bar,
    plot_sigma_evolution,
)
from src.util import create_logger, make_train_exp_name_flux, model_path, read_prompts


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, prompts_path: str):
        self.path = Path(prompts_path)
        self.prompts = read_prompts(str(self.path))

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "prompt": self.prompts[idx],
            "index": idx,
        }


def prompt_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"prompt": [item["prompt"] for item in batch]}


def _to_fp32_cpu_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().to(torch.float32).cpu() for key, value in state_dict.items()}


def load_config(path: str) -> TrainingConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TrainingConfig(**raw)


def build_sigma_schedule(num_steps: int, image_seq_len: int, device: torch.device, scheduler_mu: float) -> torch.Tensor:
    sigmas = get_schedule(
        int(num_steps),
        image_seq_len=image_seq_len,
        base_shift=float(scheduler_mu),
        max_shift=float(scheduler_mu),
    )
    return torch.tensor(sigmas, device=device, dtype=torch.float32)


def decode_latents_to_tensor(latents: torch.Tensor, vae, height: int, width: int) -> torch.Tensor:
    unpacked = unpack(latents, height, width)
    unpacked = (unpacked / vae.config.scaling_factor) + vae.config.shift_factor
    with torch.amp.autocast("cuda", enabled=False):
        vae.to(dtype=torch.float32)
        unpacked = unpacked.to(dtype=torch.float32).contiguous()
        image_tensor = vae.decode(unpacked, return_dict=False)[0]
    return image_tensor


def decode_latents_to_pil(latents: torch.Tensor, vae, image_processor, height: int, width: int) -> List[Image.Image]:
    return vae_decode(unpack(latents, height, width), vae=vae, image_processor=image_processor)


def configure_cache_transformer(transformer, cfg, logger) -> None:
    FluxTransformer2DModel.forward = transformer_cache_forward
    transformer._supports_cache_flags = True
    transformer.cache_step = torch.load(cfg.stage1_cacheStep, map_location="cpu")["cache_step"]
    expected = int(cfg.student_steps - cfg.student_nfe)
    if len(transformer.cache_step) != expected:
        raise ValueError(f"cache_step length mismatch: got {len(transformer.cache_step)}, expected {expected}")
    if transformer.cache_step:
        logger.info(f"Cache steps: {transformer.cache_step}")


def build_learnable_sigmas(cfg, transformer, student_init_sigmas: torch.Tensor, device: torch.device, logger):
    init_sigmas = None if cfg.sigmas_init_mode == "uniform" and cfg.sigmas_learn_mode == "ld3" else student_init_sigmas
    if cfg.sigmas_learn_mode == "offset":
        learn_sigmas = LearnableSigmasV2_offset(
            steps=cfg.student_steps,
            init_sigmas=student_init_sigmas,
            cache_step=transformer.cache_step,
            bound=0.005,
            optimize_all=cfg.sigmas_optimize_range,
        ).to(device)
    elif cfg.sigmas_learn_mode == "seg_ld3_offset":
        learn_sigmas = LearnableSigmasV2(
            steps=cfg.student_steps,
            init_sigmas=student_init_sigmas,
            cache_step=transformer.cache_step,
        ).to(device)
    else:
        learn_sigmas = LearnableSigmasV1(
            steps=cfg.student_steps,
            init_sigmas=init_sigmas,
        ).to(device)

    if cfg.sigmas_resume_path:
        checkpoint = torch.load(cfg.sigmas_resume_path, map_location="cpu")
        if checkpoint["student_steps"] != cfg.student_steps:
            raise ValueError(
                f"student_steps mismatch: ckpt={checkpoint['student_steps']} vs cfg={cfg.student_steps}"
            )
        learn_sigmas.load_state_dict(checkpoint["ld3_params"])
        logger.info(f"Loaded schedule params from {cfg.sigmas_resume_path}")
    return learn_sigmas


def init_loss_fn(cfg, device: torch.device):
    if str(cfg.loss) == "lpips":
        return lpips.LPIPS(net="vgg").to(device)
    if str(cfg.loss) == "mse":
        return torch.nn.MSELoss()
    raise ValueError(f"Unsupported loss: {cfg.loss}")


def build_optimizer(cfg, params):
    optimizer_name = str(cfg.optimizer).lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(params, lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def select_visual_images(images: List[Image.Image]) -> List[Image.Image]:
    if not images:
        return []
    if len(images) == 1:
        return [images[0]]
    return [images[0], images[-1]]


def build_checkpoint(unwrapped, cfg) -> Dict[str, Any]:
    """Serialize the learnable schedule to a CPU-side dict (used for both periodic
    intermediate snapshots and the final `ckpt/final.pt`)."""
    cache_step = list(getattr(unwrapped.transformer, "cache_step", []))
    return {
        "learned_sigmas": unwrapped.learn_sigmas().detach().to(torch.float32).cpu(),
        "ld3_params": _to_fp32_cpu_state(unwrapped.learn_sigmas.state_dict()),
        "student_steps": cfg.student_steps,
        "cache_step": cache_step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.use_traj_loss:
        raise ValueError("Trajectory loss is not implemented in train_flux_schedule.py.")

    # === path assembly ===
    # results       — ckpt/step*.pt + ckpt/final.pt
    # analysis (0_) — 0_sigmas/ + 0_compare/ + 0_curves/
    model_dir = model_path(cfg.model_name)
    run_dir = os.path.join(cfg.output_root, make_train_exp_name_flux(cfg), os.environ["RUN_ID"])
    ckpt_dir = os.path.join(run_dir, "ckpt")
    sigmas_dir = os.path.join(run_dir, "0_sigmas")
    compare_dir = os.path.join(run_dir, "0_compare")
    curves_dir = os.path.join(run_dir, "0_curves")
    compare_path = os.path.join(curves_dir, "teacher_student_compare.png")

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(sigmas_dir, exist_ok=True)
        os.makedirs(compare_dir, exist_ok=True)
        os.makedirs(curves_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    logger = create_logger(run_dir, rank=accelerator.process_index, name="flux-train-schedule")
    logger.info(f"Output: {run_dir}")
    logger.info(f"Config: {args.config}")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    logger.info("Loading CLIP text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(model_dir, subfolder="text_encoder", torch_dtype=weight_dtype)
    logger.info("Loading T5 tokenizer...")
    tokenizer_2 = T5TokenizerFast.from_pretrained(model_dir, subfolder="tokenizer_2")
    logger.info("Loading T5 text encoder...")
    text_encoder_2 = T5EncoderModel.from_pretrained(model_dir, subfolder="text_encoder_2", torch_dtype=weight_dtype)
    logger.info("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float32)
    logger.info("Loading FLUX transformer...")
    transformer = FluxTransformer2DModel.from_pretrained(
        model_dir,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    logger.info("Finished loading model components.")
    image_processor = VaeImageProcessor(vae_scale_factor=16)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    vae.disable_slicing()
    vae.disable_tiling()
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    configure_cache_transformer(transformer, cfg, logger)
    loss_fn = init_loss_fn(cfg, accelerator.device)

    train_dataset = PromptDataset(cfg.prompts_file)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=prompt_collate,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
        drop_last=True,
    )

    latent_height = 2 * math.ceil(cfg.height / 16)
    latent_width = 2 * math.ceil(cfg.width / 16)
    image_seq_len = (latent_height * latent_width) // 4
    teacher_sigmas = build_sigma_schedule(cfg.teacher_steps, image_seq_len, accelerator.device, cfg.scheduler_mu)
    student_base_sigmas = build_sigma_schedule(cfg.student_steps, image_seq_len, accelerator.device, cfg.scheduler_mu)
    learn_sigmas = build_learnable_sigmas(cfg, transformer, student_base_sigmas, accelerator.device, logger)
    flux_runner = FluxRunner(transformer=transformer, learn_sigmas=learn_sigmas)

    init_sigmas = flux_runner.learn_sigmas().detach().cpu().numpy()
    base_sigmas = student_base_sigmas.detach().cpu().numpy()
    optimizer = build_optimizer(cfg, flux_runner.learn_sigmas.parameters())

    if cfg.max_train_steps is None:
        steps_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        updates_per_epoch = math.ceil(steps_after_sharding / cfg.gradient_accumulation_steps)
        scheduler_total_steps = cfg.num_train_epochs * updates_per_epoch * accelerator.num_processes
    else:
        scheduler_total_steps = cfg.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(scheduler_total_steps * cfg.lr_warmup_ratio),
        num_training_steps=scheduler_total_steps,
        num_cycles=int(cfg.lr_num_cycles),
        power=float(cfg.lr_power),
    )
    flux_runner, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_runner, optimizer, train_dataloader, lr_scheduler
    )

    updates_per_epoch_real = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * updates_per_epoch_real
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / updates_per_epoch_real)
    if accelerator.is_main_process:
        with open(os.path.join(run_dir, "00_run_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(vars(cfg), f, sort_keys=False, allow_unicode=True)

    logger.info("Running schedule training")
    logger.info(f"Num examples: {len(train_dataset)}")
    logger.info(f"Prompts file: {cfg.prompts_file}")
    logger.info(f"Num epochs: {cfg.num_train_epochs}")
    logger.info(
        f"Total train batch size: {cfg.train_batch_size}x{cfg.gradient_accumulation_steps}x{accelerator.num_processes}"
    )
    logger.info(f"Total optimization steps: {cfg.max_train_steps}")
    logger.info(f"Teacher steps={cfg.teacher_steps}, Student steps={cfg.student_steps}")
    logger.info(f"Loss={cfg.loss}, Sigma mode={cfg.sigmas_learn_mode}")
    logger.info(f"Optimizer={cfg.optimizer}, LR scheduler={cfg.lr_scheduler_type}")

    progress_bar = tqdm(
        range(int(cfg.max_train_steps)),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_main_process,
    )
    global_step = 0
    acc_loss_sum = torch.tensor(0.0, device=accelerator.device)
    acc_n = torch.tensor(0, device=accelerator.device, dtype=torch.long)
    log_loss_list: List[float] = []
    log_grad_norm_list: List[float] = []
    begin_training = time.perf_counter()

    flux_runner.train()
    for _ in range(cfg.num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(flux_runner):
                prompts = batch["prompt"]
                if accelerator.is_main_process:
                    for prompt in prompts:
                        logger.info(prompt)
                batch_size = len(prompts)
                noises = torch.randn(
                    (batch_size, 16, latent_height, latent_width),
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                inp = prepare(
                    img=noises,
                    prompt=prompts,
                    guidance=float(cfg.guidance_scale),
                    clip_encoder=text_encoder,
                    t5_encoder=text_encoder_2,
                    clip_tokenizer=tokenizer,
                    t5_tokenizer=tokenizer_2,
                )

                with torch.no_grad():
                    teacher_samples = flux_runner(
                        inp["img"],
                        teacher_sigmas,
                        inp["guidance"],
                        inp["pooled_prompt_embeds"],
                        inp["prompt_embeds"],
                        inp["text_ids"],
                        inp["img_ids"],
                        mode="teacher",
                        return_traj=False,
                    )
                student_samples = flux_runner(
                    inp["img"],
                    None,
                    inp["guidance"],
                    inp["pooled_prompt_embeds"],
                    inp["prompt_embeds"],
                    inp["text_ids"],
                    inp["img_ids"],
                    mode="student",
                    return_traj=False,
                )

                if str(cfg.loss) == "mse":
                    loss = loss_fn(teacher_samples.float(), student_samples.float())
                else:
                    with torch.no_grad():
                        teacher_img_tensor = decode_latents_to_tensor(teacher_samples.detach(), vae, cfg.height, cfg.width).detach()
                    student_img_tensor = decode_latents_to_tensor(student_samples, vae, cfg.height, cfg.width)
                    loss = loss_fn(teacher_img_tensor, student_img_tensor).mean()

                acc_loss_sum += loss.detach() * batch_size
                acc_n += batch_size
                accelerator.backward(loss)

                current_grad_norm = 0.0
                if accelerator.sync_gradients:
                    grad_norm_tensor = accelerator.clip_grad_norm_(flux_runner.parameters(), float(cfg.max_grad_norm))
                    if grad_norm_tensor is not None:
                        current_grad_norm = float(grad_norm_tensor.item())

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                total_loss_sum = accelerator.gather_for_metrics(acc_loss_sum).sum()
                total_n = accelerator.gather_for_metrics(acc_n).sum()
                step_loss = (total_loss_sum / total_n).item()
                acc_loss_sum.zero_()
                acc_n.zero_()

                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(flux_runner)
                    learned_sigmas = unwrapped.learn_sigmas().detach().to(torch.float32).cpu().numpy()
                    cache_step = list(getattr(unwrapped.transformer, "cache_step", []))
                    plot_sigma_evolution(
                        learned_sigmas=learned_sigmas,
                        init_sigmas=init_sigmas,
                        base_sigmas=base_sigmas,
                        out_png=os.path.join(sigmas_dir, f"step{global_step:06d}.png"),
                        cache_steps=cache_step,
                        title=f"FLUX schedule step {global_step}",
                    )
                    plot_sigma_diff_bar(
                        learned_sigmas=learned_sigmas,
                        init_sigmas=init_sigmas,
                        out_png=os.path.join(curves_dir, "sigma_diff.png"),
                        cache_steps=cache_step,
                    )
                    log_loss_list.append(step_loss)
                    log_grad_norm_list.append(current_grad_norm)
                    progress_bar.set_postfix(step_loss=step_loss, lr=lr_scheduler.get_last_lr()[0])

                    if global_step % 2 == 0:
                        plot_loss_curve(curves_dir, log_loss_list)
                        plot_grad_norm_curve(curves_dir, log_grad_norm_list)
                    if global_step % int(cfg.visual_steps) == 0:
                        with torch.no_grad():
                            teacher_pils = decode_latents_to_pil(
                                teacher_samples.detach(), vae, image_processor, cfg.height, cfg.width
                            )
                            student_pils = decode_latents_to_pil(
                                student_samples.detach(), vae, image_processor, cfg.height, cfg.width
                            )
                        panel = build_teacher_student_panel(
                            select_visual_images(teacher_pils),
                            select_visual_images(student_pils),
                            step_label=f"step {global_step}",
                        )
                        if panel is not None:
                            step_compare_path = os.path.join(compare_dir, f"step{global_step:06d}.png")
                            panel.save(step_compare_path)
                            append_image_panel(compare_path, panel)

                if global_step % int(cfg.ckpt_save_steps) == 0:
                    accelerator.wait_for_everyone()
                    unwrapped = accelerator.unwrap_model(flux_runner)
                    accelerator.save(
                        build_checkpoint(unwrapped, cfg),
                        os.path.join(ckpt_dir, f"step{global_step:06d}.pt"),
                    )

                progress_bar.update(1)
                if global_step >= cfg.max_train_steps:
                    break
        if global_step >= cfg.max_train_steps:
            break

    progress_bar.close()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(flux_runner)
        checkpoint = build_checkpoint(unwrapped, cfg)
        accelerator.save(checkpoint, os.path.join(ckpt_dir, "final.pt"))
        make_png2gif(sigmas_dir, os.path.join(curves_dir, "sigma_evolution.gif"))
        plot_loss_curve(curves_dir, log_loss_list)
        plot_grad_norm_curve(curves_dir, log_grad_norm_list)
        plot_sigma_diff_bar(
            learned_sigmas=checkpoint["learned_sigmas"].numpy(),
            init_sigmas=init_sigmas,
            out_png=os.path.join(curves_dir, "sigma_diff_final.png"),
            cache_steps=checkpoint["cache_step"],
        )

    logger.info(f"Finished training in {time.perf_counter() - begin_training:.3f}s")
    accelerator.end_training()


if __name__ == "__main__":
    main()
