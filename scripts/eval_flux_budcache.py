import argparse
import os
import time
from dataclasses import asdict

import torch
import torch.distributed as dist
from diffusers import FluxPipeline
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import yaml

from src.flux.cache_manager import set_budcache
from src.flux.config import FLUX_LORA_REGISTRY, FluxInferenceConfig
from src.flux.sampling import (
    denoising_euler,
    get_noise,
    get_schedule,
    prepare,
    unpack,
    vae_decode,
)
from src.util import create_logger, make_eval_exp_name_flux, model_path, read_prompts


def load_config(path: str) -> FluxInferenceConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a mapping config at {path}, got {type(raw).__name__}")
    return FluxInferenceConfig(**raw)


def load_learned_sigmas(ckpt_path: str, steps: int) -> tuple[list[float], list[int] | None]:
    """Load the stage2 learned schedule produced by stage2_train_schedule_flux.py.

    The checkpoint stores ``learned_sigmas`` as the full schedule of length
    ``student_steps + 1`` (it includes the trailing zero). ``denoising_euler``
    runs ``len(sigmas) - 1`` integration steps and consumes each sigma as a scalar
    fill, so we return a ``list[float]`` of length ``steps + 1`` unchanged.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "learned_sigmas" not in checkpoint:
        raise KeyError(f"Missing `learned_sigmas` in checkpoint: {ckpt_path}")
    sigmas = checkpoint["learned_sigmas"]
    if isinstance(sigmas, torch.Tensor):
        sigmas = sigmas.detach().to(torch.float32).cpu().flatten()
    else:
        sigmas = torch.tensor(sigmas, dtype=torch.float32).flatten()
    if len(sigmas) != int(steps) + 1:
        raise ValueError(f"Expected {int(steps) + 1} learned sigmas, got {len(sigmas)}")
    cache_step = checkpoint.get("cache_step")
    if cache_step is not None:
        cache_step = sorted({int(step) for step in cache_step if 0 <= int(step) < int(steps)})
    return sigmas.tolist(), cache_step


class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt_idx": idx, "prompt": self.prompts[idx]}


def collate_fn(examples):
    return (
        [e["prompt_idx"] for e in examples],
        [e["prompt"] for e in examples],
    )


def setup_distributed():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group("nccl")


def is_main_process():
    return dist.get_rank() == 0


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML file path")
    parser.add_argument("--stage1-ckpt", type=str, required=True, help="Stage1 BudCache checkpoint (cache_step)")
    parser.add_argument("--stage2-ckpt", type=str, default="", help="Stage2 BudCache checkpoint (learned_sigmas, optional)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # === path assembly ===
    model_dir = model_path(cfg.model_name)
    output_root = os.path.join(cfg.output_root, make_eval_exp_name_flux(cfg), os.environ["RUN_ID"])
    samples_dir = os.path.join(output_root, "samples")

    # === checkpoints ===
    budcache_step = torch.load(args.stage1_ckpt, map_location="cpu")["cache_step"]
    nfe = int(cfg.steps) - len(budcache_step)
    learned_sigmas = None
    if args.stage2_ckpt:
        learned_sigmas, stage2_cache_step = load_learned_sigmas(args.stage2_ckpt, int(cfg.steps))
        if stage2_cache_step is not None and sorted(stage2_cache_step) != sorted(budcache_step):
            raise ValueError("Stage2 checkpoint cache_step does not match stage1 checkpoint.")

    # === stage 1: output dir + logger ===
    if is_main_process():
        os.makedirs(samples_dir, exist_ok=True)
        save_cfg = asdict(cfg)
        save_cfg["stage1_ckpt"] = args.stage1_ckpt
        save_cfg["stage2_ckpt"] = args.stage2_ckpt
        with open(os.path.join(output_root, "00_run_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(save_cfg, f, sort_keys=False, allow_unicode=True)
    dist.barrier(device_ids=[local_rank])

    logger = create_logger(output_root, name="eval-flux-budcache", rank=rank)
    logger.info(f"Output: {output_root}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Stage1 ckpt: {args.stage1_ckpt}")
    if args.stage2_ckpt:
        logger.info(f"Stage2 ckpt: {args.stage2_ckpt}")
    logger.info(f"NFE: {nfe} (cache={len(budcache_step)}/{cfg.steps})")
    logger.info(f"Schedule: {'stage2 learned_sigmas' if learned_sigmas is not None else 'get_schedule'}")
    logger.info("Loading pipeline...")

    # === stage 2: prep environment — pipeline ===
    weight_dtype = torch.bfloat16
    pipe = FluxPipeline.from_pretrained(model_dir, torch_dtype=weight_dtype).to(local_rank)
    if cfg.use_hypersd:
        hypersd = FLUX_LORA_REGISTRY["hypersd"]
        load_kwargs = {"adapter_name": hypersd["adapter_name"]}
        if hypersd.get("weight_name"):
            load_kwargs["weight_name"] = hypersd["weight_name"]
        pipe.load_lora_weights(hypersd["path"], **load_kwargs)
        pipe.set_adapters([hypersd["adapter_name"]], adapter_weights=[hypersd["adapter_weight"]])

    # === experiment config — prompts ===
    input_prompts = read_prompts(path=cfg.dataset_path)
    num_prompts = len(input_prompts)
    samples_per_prompt = int(cfg.samples_per_prompt)
    logger.info(f"Prompts: {num_prompts} x {samples_per_prompt} samples")

    total_start = time.time()
    denoise_batch_seconds = []
    total_images = 0

    dataset = PromptDataset(input_prompts)
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )

    image_seq_len = (cfg.height // 16 * 2) * (cfg.width // 16 * 2) // 4
    if learned_sigmas is not None:
        sigmas = learned_sigmas
    else:
        sigmas = get_schedule(int(cfg.steps), image_seq_len=image_seq_len)

    for indices, prompts in loader:
        current_batch_size = len(prompts)
        effective_batch_size = current_batch_size * samples_per_prompt
        total_images += effective_batch_size
        repeated_prompts = [p for p in prompts for _ in range(samples_per_prompt)]
        print(f"[rank{rank}] indices={indices}", flush=True)
        noises = get_noise(effective_batch_size, cfg.height, cfg.width, local_rank, weight_dtype, int(cfg.seed))
        model_input = prepare(
            img=noises,
            prompt=repeated_prompts,
            guidance=float(cfg.guidance_scale),
            clip_encoder=pipe.text_encoder,
            t5_encoder=pipe.text_encoder_2,
            clip_tokenizer=pipe.tokenizer,
            t5_tokenizer=pipe.tokenizer_2,
        )

        set_budcache(pipe.transformer, budcache_step, cfg.steps)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        latents = denoising_euler(
            transformer=pipe.transformer,
            latents=model_input["img"].clone(),
            sigmas=sigmas,
            guidance=model_input["guidance"],
            pooled_prompt_embeds=model_input["pooled_prompt_embeds"],
            prompt_embeds=model_input["prompt_embeds"],
            text_ids=model_input["text_ids"],
            image_ids=model_input["img_ids"],
        )
        torch.cuda.synchronize()
        denoise_batch_seconds.append(time.perf_counter() - t0)

        images = vae_decode(
            latents=unpack(latents, cfg.height, cfg.width),
            vae=pipe.vae,
            image_processor=pipe.image_processor,
        )
        for k, image in enumerate(images):
            prompt_idx = indices[k // samples_per_prompt]
            sample_idx = k % samples_per_prompt
            image.save(os.path.join(samples_dir, f"p{prompt_idx:05d}_s{sample_idx:02d}.png"))

    dist.barrier(device_ids=[local_rank])
    total_time = time.time() - total_start
    denoise_total = sum(denoise_batch_seconds)
    logger.info("=" * 60)
    logger.info(f"budcache: {denoise_total / total_images:.3f} s/img  (total {denoise_total:.1f}s on {total_images} images)")
    logger.info(f"Total: {total_time:.2f}s | Saved: {output_root}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
