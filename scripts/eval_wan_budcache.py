import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import src.wan as wan
from src.util import (
    create_comparison_grid,
    create_logger,
    extract_frames,
    get_preview_indices,
    make_eval_exp_name_wan,
    model_path,
    read_prompts,
)
from src.wan.config import WanInferenceConfig
from src.wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from src.wan.dit_forward import budcache_forward
from src.wan.utils.utils import cache_video
from src.wan.wan_generate import custom_t2v_generate


def load_config(path: str) -> WanInferenceConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a mapping config at {path}, got {type(raw).__name__}")
    return WanInferenceConfig(**raw)


def load_learned_sigmas(ckpt_path: str, steps: int) -> tuple[list[float], list[int] | None]:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "learned_sigmas" not in checkpoint:
        raise KeyError(f"Missing `learned_sigmas` in checkpoint: {ckpt_path}")
    sigmas = checkpoint["learned_sigmas"]
    if isinstance(sigmas, torch.Tensor):
        sigmas = sigmas.detach().to(torch.float32).cpu().flatten()
    else:
        sigmas = torch.tensor(sigmas, dtype=torch.float32).flatten()
    if len(sigmas) == int(steps) + 1:
        sigmas = sigmas[:-1]
    elif len(sigmas) != int(steps):
        raise ValueError(f"Expected {steps} or {steps + 1} learned sigmas, got {len(sigmas)}")
    cache_step = checkpoint.get("cache_step")
    if cache_step is not None:
        cache_step = sorted({int(step) for step in cache_step if 0 <= int(step) < int(steps)})
    return sigmas.tolist(), cache_step


def setup_budcache(model, cache_step: list[int], steps: int) -> None:
    model.__class__.forward = budcache_forward
    model.enable_budcache = True
    model.cache_schedule = [0 if step in cache_step else 1 for step in range(int(steps))]
    model.cnt = 0
    model.num_steps = int(steps) * 2
    model.previous_residual_even = None
    model.previous_residual_odd = None


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
    parser.add_argument("config", type=str, help="YAML File Path")
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
    output_root = os.path.join(cfg.output_root, make_eval_exp_name_wan(cfg), os.environ["RUN_ID"])
    samples_dir = os.path.join(output_root, "samples")
    preview_dir = os.path.join(output_root, "0_extract_images")

    # === checkpoints ===
    cache_step = torch.load(args.stage1_ckpt, map_location="cpu")["cache_step"]
    sampling_sigmas = None
    if args.stage2_ckpt:
        sampling_sigmas, stage2_cache_step = load_learned_sigmas(args.stage2_ckpt, int(cfg.steps))
        if stage2_cache_step is not None and stage2_cache_step != cache_step:
            raise ValueError("Stage2 checkpoint cache_step does not match stage1 checkpoint.")

    # === stage 1: output dir + logger ===
    if is_main_process():
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(preview_dir, exist_ok=True)
        save_cfg = asdict(cfg)
        save_cfg["stage1_ckpt"] = args.stage1_ckpt
        save_cfg["stage2_ckpt"] = args.stage2_ckpt
        with open(os.path.join(output_root, "00_run_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(save_cfg, f, sort_keys=False, allow_unicode=True)
    dist.barrier(device_ids=[local_rank])

    logger = create_logger(output_root, name="eval-wan-budcache", rank=rank)
    logger.info(f"Output: {output_root}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Stage1 ckpt: {args.stage1_ckpt}")
    if args.stage2_ckpt:
        logger.info(f"Stage2 ckpt: {args.stage2_ckpt}")
    logger.info(f"NFE: {int(cfg.steps) - len(cache_step)} (cache={len(cache_step)}/{cfg.steps})")

    # === stage 2: prep environment — pipeline ===
    logger.info("Loading WAN pipeline...")
    model_cfg = WAN_CONFIGS[cfg.task]
    pipe = wan.WanT2V(
        config=model_cfg,
        checkpoint_dir=model_dir,
        device_id=local_rank,
        # This eval script uses ranks as independent data-parallel workers.
        # WanT2V returns None on nonzero self.rank, so force decode on every worker.
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    pipe.__class__.generate = custom_t2v_generate
    setup_budcache(pipe.model, cache_step, int(cfg.steps))

    # === experiment config — prompts + manifest ===
    prompts = read_prompts(cfg.dataset_path)
    num_prompts = len(prompts)
    samples_per_prompt = int(cfg.samples_per_prompt)
    logger.info(f"Prompts: {num_prompts} x {samples_per_prompt} samples")

    if is_main_process():
        with open(os.path.join(output_root, "00_prompt_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_name": cfg.dataset_name,
                    "dataset_path": cfg.dataset_path,
                    "num_prompts": num_prompts,
                    "prompts": prompts,
                    "cache_step": cache_step,
                    "nfe": int(cfg.steps) - len(cache_step),
                    "use_learned_sigmas": sampling_sigmas is not None,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    dataset = PromptDataset(prompts)
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate_fn, num_workers=0)

    preview_indices = get_preview_indices(int(cfg.frame_num))
    timings = []
    start_time = time.time()
    for sample_idx in range(samples_per_prompt):
        seed_k = int(cfg.seed) + sample_idx
        for indices, batch_prompts in loader:
            prompt_idx = int(indices[0])
            prompt = batch_prompts[0]
            print(f"[rank{rank}] prompt_idx={prompt_idx}, sample_idx={sample_idx}", flush=True)

            # Reset BudCache state for each generation
            pipe.model.cnt = 0
            pipe.model.previous_residual_even = None
            pipe.model.previous_residual_odd = None

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            video = pipe.generate(
                prompt,
                size=SIZE_CONFIGS[cfg.size],
                frame_num=int(cfg.frame_num),
                shift=float(cfg.shift),
                sample_solver="euler",
                sampling_steps=int(cfg.steps),
                sampling_sigmas=sampling_sigmas,
                guide_scale=float(cfg.guidance_scale),
                seed=seed_k,
                offload_model=bool(cfg.offload_model),
            )
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)

            base_name = f"{prompt_idx:05d}_{sample_idx:03d}"
            cache_video(
                tensor=video.unsqueeze(0),
                save_file=os.path.join(samples_dir, f"{base_name}.mp4"),
                fps=16,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            preview = create_comparison_grid([extract_frames(video, preview_indices)])
            if preview:
                preview.save(os.path.join(preview_dir, f"{base_name}.png"))

    dist.barrier(device_ids=[local_rank])
    total_time = time.time() - start_time
    mean_time = float(np.mean(timings)) if timings else 0.0
    std_time = float(np.std(timings)) if timings else 0.0
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.2f}s | world_size={world_size}")
    logger.info(
        f"Config: steps={cfg.steps}, size={cfg.size}, solver=euler, cfg={cfg.guidance_scale}, seed={cfg.seed}"
    )
    logger.info(f"Per-video: {mean_time:.3f}+-{std_time:.3f}s (rank {rank} measured {len(timings)} videos)")
    logger.info("=" * 60)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
