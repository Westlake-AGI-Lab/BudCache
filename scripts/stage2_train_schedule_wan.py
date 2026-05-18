import argparse
import copy
import math
import os
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import yaml

import src.wan as wan
from src.train_util import (
    append_image_panel,
    build_teacher_student_panel,
    make_png2gif,
    plot_grad_norm_curve,
    plot_loss_curve,
    plot_sigma_diff_bar,
    plot_sigma_evolution,
)
from src.util import create_logger, extract_frames, get_preview_indices, make_train_exp_name_wan, model_path, read_prompts
from src.wan.config import WanTrainingConfig
from src.wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from src.wan.dit_forward import budcache_forward
from src.wan.scheduler_trainer import LearnableSigmasV1, WanRunner, build_sampling_sigmas, pack_contexts

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, prompts_path: str):
        self.path = Path(prompts_path)
        self.prompts = read_prompts(str(self.path))

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int):
        return {"prompt": self.prompts[idx], "index": idx}


def prompt_collate(batch):
    return {"prompt": [item["prompt"] for item in batch]}


def load_config(path: str | None) -> WanTrainingConfig:
    if not path:
        return WanTrainingConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a mapping config at {path}, got {type(raw).__name__}")
    return WanTrainingConfig(**raw)


def _to_fp32_cpu_state(state_dict):
    return {key: value.detach().to(torch.float32).cpu() for key, value in state_dict.items()}


def resolve_weight_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def build_optimizer(cfg, params):
    optimizer_name = str(cfg.optimizer).lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(params, lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def resolve_cache_step(cfg, logger=None) -> list[int]:
    if not cfg.cache_step_path:
        return []
    checkpoint = torch.load(cfg.cache_step_path, map_location="cpu", weights_only=True)
    cache_step = checkpoint["cache_step"]
    cache_step = sorted({int(step) for step in cache_step if 0 <= int(step) < int(cfg.student_steps)})
    if 0 in cache_step:
        raise ValueError("cache_step must not include step 0, because the first step has no cached residual.")
    if logger is not None:
        logger.info(f"Loaded cache_step from {cfg.cache_step_path}")
    return cache_step


def reset_budcache_state(model, enabled: bool) -> None:
    model.enable_budcache = bool(enabled)
    model.cnt = 0
    model.is_even = False
    model.previous_residual_cond = None
    model.previous_residual_uncond = None
    model.previous_residual_even = None
    model.previous_residual_odd = None


def setup_budcache(model, cache_step: list[int], steps: int) -> None:
    model.__class__.forward = budcache_forward
    model._supports_cache_flags = True
    model.cache_step = list(cache_step)
    model.cache_schedule = [0 if step in cache_step else 1 for step in range(int(steps))]
    model.num_steps = int(steps) * 2
    reset_budcache_state(model, enabled=False)


def build_target_shape(pipe, size, frame_num: int):
    return (
        pipe.vae.model.z_dim,
        (int(frame_num) - 1) // pipe.vae_stride[0] + 1,
        size[1] // pipe.vae_stride[1],
        size[0] // pipe.vae_stride[2],
    )


def build_seq_len(pipe, target_shape) -> int:
    return (
        math.ceil(
            (target_shape[2] * target_shape[3])
            / (pipe.patch_size[1] * pipe.patch_size[2])
            * target_shape[1]
            / pipe.sp_size
        )
        * pipe.sp_size
    )


def _set_vae_device(pipe, device: torch.device) -> None:
    pipe.vae.device = device
    pipe.vae.mean = pipe.vae.mean.to(device)
    pipe.vae.std = pipe.vae.std.to(device)
    pipe.vae.scale = [pipe.vae.mean, 1.0 / pipe.vae.std]
    pipe.vae.model.to(device)


@torch.no_grad()
def decode_teacher_student_rows(pipe, teacher_latent, student_latent, indices: list[int], device):
    _set_vae_device(pipe, device)
    videos = pipe.vae.decode([teacher_latent.to(device), student_latent.to(device)])
    rows = [extract_frames(video, indices) for video in videos]
    _set_vae_device(pipe, torch.device("cpu"))
    if len(rows) != 2:
        return [], []
    return rows[0], rows[1]


def encode_prompt_contexts(pipe, prompts, device: torch.device):
    negative_prompts = [pipe.sample_neg_prompt] * len(prompts)
    with torch.no_grad():
        if pipe.t5_cpu:
            context = pipe.text_encoder(prompts, torch.device("cpu"))
            context_null = pipe.text_encoder(negative_prompts, torch.device("cpu"))
            context = [ctx.to(device) for ctx in context]
            context_null = [ctx.to(device) for ctx in context_null]
        else:
            context = pipe.text_encoder(prompts, device)
            context_null = pipe.text_encoder(negative_prompts, device)
    return pack_contexts(context), pack_contexts(context_null)


def build_learnable_sigmas(cfg, student_init_sigmas: torch.Tensor, device: torch.device, logger):
    init_sigmas = student_init_sigmas if cfg.sigmas_init_mode == "wan" else None
    learn_sigmas = LearnableSigmasV1(steps=cfg.student_steps, init_sigmas=init_sigmas).to(device)
    return learn_sigmas


def build_checkpoint(unwrapped, cfg, cache_step) -> dict:
    """Serialize the learnable schedule to a CPU-side dict (used for both periodic
    intermediate snapshots and the final `ckpt/final.pt`)."""
    return {
        "learned_sigmas": unwrapped.learn_sigmas().detach().to(torch.float32).cpu(),
        "schedule_params": _to_fp32_cpu_state(unwrapped.learn_sigmas.state_dict()),
        "teacher_steps": cfg.teacher_steps,
        "student_steps": cfg.student_steps,
        "shift": cfg.shift,
        "cache_step": list(cache_step),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if str(cfg.loss) != "mse":
        raise ValueError("train_wan_schedule.py currently only supports mse loss.")
    cache_step = resolve_cache_step(cfg)
    use_budcache = bool(cfg.cache_step_path)
    if use_budcache and int(cfg.teacher_steps) != int(cfg.student_steps):
        raise ValueError("Cache training requires teacher_steps == student_steps.")
    if use_budcache and not cache_step:
        raise ValueError("cache_step_path did not provide a valid non-empty cache_step.")

    # === path assembly ===
    # results       — ckpt/step*.pt + ckpt/final.pt
    # analysis (0_) — 0_sigmas/ + 0_compare/ + 0_curves/
    model_dir = model_path(cfg.model_name)
    run_dir = os.path.join(cfg.output_root, make_train_exp_name_wan(cfg), os.environ["RUN_ID"])
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

    logger = create_logger(run_dir, rank=accelerator.process_index, name="wan-train-schedule")
    logger.info(f"Output: {run_dir}")
    logger.info(f"Config: {args.config}")
    if use_budcache:
        logger.info(f"Loaded cache_step from {cfg.cache_step_path}")

    weight_dtype = resolve_weight_dtype(cfg.mixed_precision)
    model_cfg = copy.deepcopy(WAN_CONFIGS[cfg.task])
    model_cfg.param_dtype = weight_dtype
    model_cfg.t5_dtype = weight_dtype

    logger.info("Loading WAN pipeline...")
    pipe = wan.WanT2V(
        config=model_cfg,
        checkpoint_dir=model_dir,
        device_id=accelerator.local_process_index,
        rank=accelerator.process_index,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=bool(cfg.t5_cpu),
    )
    pipe.model.eval().requires_grad_(False)
    pipe.vae.model.cpu()
    if use_budcache:
        setup_budcache(pipe.model, cache_step, cfg.student_steps)
    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(accelerator.device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    size = SIZE_CONFIGS[cfg.size]
    target_shape = build_target_shape(pipe, size, cfg.frame_num)
    seq_len = build_seq_len(pipe, target_shape)
    preview_indices = get_preview_indices(int(cfg.frame_num))

    train_dataset = PromptDataset(cfg.prompts_file)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=prompt_collate,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
        drop_last=True,
    )

    teacher_sigmas = build_sampling_sigmas(cfg.teacher_steps, cfg.shift, accelerator.device)
    student_base_sigmas = build_sampling_sigmas(cfg.student_steps, cfg.shift, accelerator.device)
    learn_sigmas = build_learnable_sigmas(cfg, student_base_sigmas, accelerator.device, logger)
    init_sigmas = learn_sigmas().detach().to(torch.float32).cpu().numpy()
    base_sigmas = student_base_sigmas.detach().to(torch.float32).cpu().numpy()
    runner = WanRunner(
        model=pipe.model,
        learn_sigmas=learn_sigmas,
        num_train_timesteps=pipe.num_train_timesteps,
        seq_len=seq_len,
        guidance_scale=float(cfg.guidance_scale),
        param_dtype=weight_dtype,
        use_checkpoint=True,
    )

    optimizer = build_optimizer(cfg, runner.learn_sigmas.parameters())
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
    runner, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        runner, optimizer, train_dataloader, lr_scheduler
    )
    runner_model = accelerator.unwrap_model(runner).model

    updates_per_epoch_real = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * updates_per_epoch_real
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / updates_per_epoch_real)
    if accelerator.is_main_process:
        with open(os.path.join(run_dir, "00_run_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(vars(cfg), f, sort_keys=False, allow_unicode=True)

    logger.info("Running WAN schedule training")
    logger.info(f"Num examples: {len(train_dataset)}")
    logger.info(f"Prompts file: {cfg.prompts_file}")
    logger.info(f"Size={cfg.size}, frame_num={cfg.frame_num}")
    logger.info(f"Teacher steps={cfg.teacher_steps}, Student steps={cfg.student_steps}")
    logger.info(f"BudCache enabled: {use_budcache}")
    if use_budcache:
        logger.info(f"Cache steps: {cache_step}")
    logger.info(f"Total optimization steps: {cfg.max_train_steps}")
    logger.info(f"Optimizer={cfg.optimizer}, LR scheduler={cfg.lr_scheduler_type}")
    logger.info(f"T5 on CPU: {cfg.t5_cpu}")
    logger.info(f"Visual steps: {cfg.visual_steps}")

    progress_bar = tqdm(
        range(int(cfg.max_train_steps)),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_main_process,
    )
    global_step = 0
    acc_loss_sum = torch.tensor(0.0, device=accelerator.device)
    acc_n = torch.tensor(0, device=accelerator.device, dtype=torch.long)
    begin_training = time.perf_counter()
    loss_fn = torch.nn.MSELoss()
    log_loss_list = []
    log_grad_norm_list = []

    runner.train()
    for _ in range(cfg.num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(runner):
                prompts = batch["prompt"]
                if accelerator.is_main_process:
                    for prompt in prompts:
                        logger.info(prompt)
                batch_size = len(prompts)
                noises = torch.randn(
                    (batch_size, *target_shape),
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                (context, context_lens), (context_null, context_null_lens) = encode_prompt_contexts(
                    pipe,
                    prompts,
                    accelerator.device,
                )

                if use_budcache:
                    reset_budcache_state(runner_model, enabled=False)
                teacher_samples = runner(
                    noises,
                    teacher_sigmas,
                    context,
                    context_lens,
                    context_null,
                    context_null_lens,
                    mode="teacher",
                )
                if use_budcache:
                    reset_budcache_state(runner_model, enabled=True)
                student_samples = runner(
                    noises,
                    None,
                    context,
                    context_lens,
                    context_null,
                    context_null_lens,
                    mode="student",
                )
                loss = loss_fn(teacher_samples.float(), student_samples.float())

                acc_loss_sum += loss.detach() * batch_size
                acc_n += batch_size
                accelerator.backward(loss)

                current_grad_norm = 0.0
                if accelerator.sync_gradients:
                    grad_norm_tensor = accelerator.clip_grad_norm_(runner.parameters(), float(cfg.max_grad_norm))
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
                    unwrapped = accelerator.unwrap_model(runner)
                    learned_sigmas = unwrapped.learn_sigmas().detach().to(torch.float32).cpu().numpy()
                    plot_sigma_evolution(
                        learned_sigmas=learned_sigmas,
                        init_sigmas=init_sigmas,
                        base_sigmas=base_sigmas,
                        out_png=os.path.join(sigmas_dir, f"step{global_step:06d}.png"),
                        cache_steps=cache_step,
                        title=f"WAN schedule step {global_step}",
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
                    if int(cfg.visual_steps) > 0 and global_step % int(cfg.visual_steps) == 0:
                        teacher_frames, student_frames = decode_teacher_student_rows(
                            pipe,
                            teacher_samples[0].detach().to(torch.float32),
                            student_samples[0].detach().to(torch.float32),
                            preview_indices,
                            accelerator.device,
                        )
                        panel = build_teacher_student_panel(
                            teacher_frames,
                            student_frames,
                            step_label=f"step {global_step}",
                        )
                        if panel is not None:
                            step_compare_path = os.path.join(compare_dir, f"step{global_step:06d}.png")
                            panel.save(step_compare_path)
                            append_image_panel(compare_path, panel)

                if global_step % int(cfg.ckpt_save_steps) == 0:
                    accelerator.wait_for_everyone()
                    unwrapped = accelerator.unwrap_model(runner)
                    accelerator.save(
                        build_checkpoint(unwrapped, cfg, cache_step),
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
        unwrapped = accelerator.unwrap_model(runner)
        checkpoint = build_checkpoint(unwrapped, cfg, cache_step)
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
