import argparse
import os
import time
from dataclasses import asdict
from functools import partial

import torch
import torch.distributed as dist
import yaml
from diffusers import ZImagePipeline
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.zimage.cache_manager import reset2original, set_budcache
from src.zimage.config import ZImageInferenceConfig
from src.zimage.sampling import get_noise
from src.util import create_logger, make_eval_exp_name_zimage, read_prompts, save_image_grid


# Diffusers-pipeline variant of eval_zimage_budcache.py.
# Instead of re-implementing the flow-match Euler loop (src.zimage.sampling.denoising_search),
# this drives the stock ZImagePipeline.__call__ for the denoising. BudCache is still applied the
# same way -- by monkey-patching ZImageTransformer2DModel.forward (cache_manager) -- because the
# pipeline calls `self.transformer(...)` internally, so the cache-aware forward is transparent to it.
# The key requirement for an apples-to-apples comparison (full vs budcache) is that every strategy
# sees the SAME initial noise; we get that by pre-sampling the latent once and passing it through
# the pipeline's `latents=` argument (so prepare_latents reuses it instead of drawing fresh noise).
# We pre-encode the caption embeddings once (shared across strategies, keeps text-encode out of the
# per-strategy timer) and let the pipeline return PIL images directly via its default decode path.
# The timed region therefore covers the denoise loop plus VAE decode; both are identical fixed costs
# across strategies, so the full-vs-budcache comparison still isolates the denoise speedup.


def load_config(path: str) -> ZImageInferenceConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a mapping config at {path}, got {type(raw).__name__}")
    return ZImageInferenceConfig(**raw)


class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt_idx": idx, "prompt": self.prompts[idx]}


def collate_fn(examples):
    return [e["prompt_idx"] for e in examples], [e["prompt"] for e in examples]


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
    parser.add_argument("--stage1-ckpt", required=True, type=str,
                        help="BudCache search ckpt (.pt with cache_step), used by the `budcache` strategy.")
    parser.add_argument("--run-full", action=argparse.BooleanOptionalAction, default=True,
                        help="Also run the full-compute baseline for comparison (--no-run-full for budcache only).")
    args = parser.parse_args()
    cfg: ZImageInferenceConfig = load_config(args.config)

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # === path assembly ===
    model_path = os.path.join(os.environ["MODEL_ROOT"], cfg.model_name)
    output_dir = os.path.join(cfg.output_root, make_eval_exp_name_zimage(cfg), os.environ["RUN_ID"])
    grid_dir = os.path.join(output_dir, "0_grid")

    # === cache checkpoint (the budcache strategy applies these skipped steps) ===
    cache_step = list(torch.load(args.stage1_ckpt, map_location="cpu")["cache_step"])

    # === stage 1: output dir + run config ready before any heavy work ===
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(grid_dir, exist_ok=True)
        save_cfg = asdict(cfg)
        save_cfg["stage1_ckpt"] = args.stage1_ckpt
        save_cfg["backend"] = "diffusers"
        with open(os.path.join(output_dir, "00_run_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(save_cfg, f, sort_keys=False, allow_unicode=True)
    dist.barrier(device_ids=[local_rank])

    logger = create_logger(output_dir, name="eval-zimage-cache-diffusers", rank=rank)
    logger.info(f"Output: {output_dir}")
    logger.info("Backend: diffusers ZImagePipeline.__call__ (denoising via stock pipeline)")
    logger.info(f"steps={cfg.steps}, guidance={cfg.guidance_scale}, res={cfg.width}x{cfg.height}")
    logger.info(f"BudCache: cache={len(cache_step)}/{cfg.steps}, NFE={cfg.steps - len(cache_step)}")
    logger.info("Loading pipeline...")

    # === pipeline (native forward; cache manager flips it per-strategy as needed) ===
    weight_dtype = torch.bfloat16
    pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=weight_dtype).to(local_rank)
    pipe.set_progress_bar_config(disable=True)

    # === strategies — each maps to (forward-patch setup, cache_step) ===
    # full     : native forward, every step computed (teacher); optional baseline.
    # budcache : cache-aware forward, skip the searched steps (student).
    # `full` first so the concat grid reads [full | budcache] when both are on.
    # The setup_fn (re)installs / removes the cache-aware transformer.forward AND resets the
    # per-generation cache counters, so it must be called before every pipeline run.
    strategy_setups = {}
    if args.run_full:
        strategy_setups["full"] = partial(reset2original, pipe.transformer)
    strategy_setups["budcache"] = partial(set_budcache, pipe.transformer, cache_step, int(cfg.steps))
    if is_main_process():
        for subdir in strategy_setups:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    dist.barrier(device_ids=[local_rank])

    # === prompts / loader ===
    input_prompts = read_prompts(path=cfg.dataset_path)
    samples_per_prompt = int(cfg.samples_per_prompt)
    logger.info(f"Strategies: {list(strategy_setups.keys())}")
    logger.info(f"Prompts: {len(input_prompts)} x {samples_per_prompt} samples")

    dataset = PromptDataset(input_prompts)
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=0)

    guidance = float(cfg.guidance_scale)
    do_cfg = guidance > 1.0  # base Z-Image: CFG on; Turbo: guidance 0 -> single forward per step
    denoise_batch_seconds = {subdir: [] for subdir in strategy_setups}
    total_images = 0
    # Reset peak-memory tracker AFTER load so the reported peak reflects inference, not load spikes.
    torch.cuda.reset_peak_memory_stats(local_rank)
    total_start = time.time()
    for indices, prompts in loader:
        effective_bsz = len(prompts) * samples_per_prompt
        total_images += effective_bsz
        repeated_prompts = [p for p in prompts for _ in range(samples_per_prompt)]
        print(f"[rank{rank}] indices={list(indices)}", flush=True)

        # Same fixed noise + prompt features feed every strategy so the comparison is apples-to-apples.
        # The noise is passed to the pipeline via `latents=` (prepare_latents reuses it verbatim).
        noise = get_noise(effective_bsz, cfg.height, cfg.width, local_rank, torch.float32, int(cfg.seed))
        # Pre-encode caption (and empty negative) features once -> keeps text-encode out of the timed
        # region and shared across strategies. pipe.encode_prompt returns a per-prompt list of tensors.
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=repeated_prompts,
            negative_prompt=None,
            do_classifier_free_guidance=do_cfg,
            device=local_rank,
            max_sequence_length=int(cfg.max_sequence_length),
        )
        neg_arg = negative_prompt_embeds if do_cfg else None

        all_strategy_images = []
        for subdir, setup_fn in strategy_setups.items():
            setup_fn()  # install/remove cache forward + reset cache counters for this generation
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            images = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_arg,
                height=int(cfg.height),
                width=int(cfg.width),
                num_inference_steps=int(cfg.steps),
                guidance_scale=guidance,
                num_images_per_prompt=1,
                latents=noise.clone(),     # identical initial noise for every strategy
                output_type="pil",         # pipeline returns PIL directly (VAE decode handled inside)
                return_dict=False,
            )[0]
            torch.cuda.synchronize()
            denoise_batch_seconds[subdir].append(time.perf_counter() - t0)

            all_strategy_images.append(images)
            for k, image in enumerate(images):
                prompt_idx = indices[k // samples_per_prompt]
                sample_idx = k % samples_per_prompt
                image.save(os.path.join(output_dir, subdir, f"p{prompt_idx:05d}_s{sample_idx:02d}.png"))

        # Concat grid: all strategies side-by-side per sample (full | budcache).
        for k in range(effective_bsz):
            prompt_idx = indices[k // samples_per_prompt]
            sample_idx = k % samples_per_prompt
            save_image_grid(
                [all_strategy_images[s][k] for s in range(len(strategy_setups))],
                os.path.join(grid_dir, f"grid_{prompt_idx:05d}_{sample_idx:02d}.png"),
            )

    dist.barrier(device_ids=[local_rank])
    total_time = time.time() - total_start
    logger.info("=" * 60)
    for subdir, secs in denoise_batch_seconds.items():
        strategy_total = sum(secs)
        logger.info(f"{subdir}: {strategy_total / total_images:.3f} s/img  "
                    f"(total {strategy_total:.1f}s on {total_images} images)")

    # Peak GPU memory: per-rank value, plus the max across ranks (the job's true peak).
    peak_alloc = torch.cuda.max_memory_allocated(local_rank) / 1024 ** 3
    peak_resv = torch.cuda.max_memory_reserved(local_rank) / 1024 ** 3
    mem = torch.tensor([peak_alloc, peak_resv], device=local_rank)
    dist.all_reduce(mem, op=dist.ReduceOp.MAX)
    logger.info(f"Peak GPU mem (max over ranks): allocated={mem[0].item():.2f} GiB, reserved={mem[1].item():.2f} GiB")

    logger.info(f"Total: {total_time:.2f}s | Saved: {output_dir}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
