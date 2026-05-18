import argparse
import copy
import hashlib
import math
import os
import random
from typing import List

import numpy as np
import torch
import yaml
from tqdm import tqdm

import src.wan
from src.util import create_logger, make_search_exp_name_wan, model_path
from src.wan.config import WanSearchConfig
from src.wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from src.wan.dit_forward import budcache_forward
from src.wan.wan_generate import t2v_generate


def load_config(path: str) -> WanSearchConfig:
    if not path:
        return WanSearchConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a mapping config at {path}, got {type(raw).__name__}")
    return WanSearchConfig(**raw)

def calc_mse(t1, t2):
    if isinstance(t1, list):
        t1 = t1[0] if len(t1) == 1 else torch.stack(t1)
    if isinstance(t2, list):
        t2 = t2[0] if len(t2) == 1 else torch.stack(t2)
    if t1.device != t2.device:
        t2 = t2.to(t1.device)
    return torch.mean((t1 - t2) ** 2).item()


@torch.no_grad()
def get_latents(pipe, cfg: WanSearchConfig):
    return pipe.generate(
        cfg.prompt,
        size=SIZE_CONFIGS[cfg.size],
        frame_num=cfg.frame_num,
        shift=cfg.shift,
        sample_solver=cfg.solver,
        sampling_steps=cfg.steps,
        guide_scale=cfg.guidance_scale,
        seed=cfg.seed,
        offload_model=cfg.offload_model,
    )

def mutate_schedule(schedule: List[int]) -> List[int]:
    new_schedule = copy.deepcopy(schedule)
    ones = [i for i, x in enumerate(new_schedule) if x == 1]
    zeros = [i for i, x in enumerate(new_schedule) if x == 0]

    prefix_len = max(1, int(len(ones) * 0.3))
    mutable_ones = [i for i in ones if i >= prefix_len]
    valid_zeros = [i for i in zeros if i >= prefix_len]

    if not mutable_ones or not valid_zeros:
        return new_schedule

    new_schedule[random.choice(mutable_ones)] = 0
    new_schedule[random.choice(valid_zeros)] = 1
    return new_schedule
def get_swap_neighbors(schedule: List[int], num_neighbors: int = 20, window_size: int = 3) -> List[List[int]]:
    ones = [i for i, x in enumerate(schedule) if x == 1]
    zeros = [i for i, x in enumerate(schedule) if x == 0]

    prefix_len = max(1, int(len(ones) * 0.3))
    mutable_ones = [i for i in ones if i >= prefix_len]
    valid_zeros = [i for i in zeros if i >= prefix_len]

    if not mutable_ones or not valid_zeros:
        return []

    candidates = []
    # Local window search
    for idx_1 in mutable_ones:
        local_zeros = [z for z in valid_zeros if abs(z - idx_1) <= window_size]
        for idx_0 in local_zeros:
            candidates.append((idx_1, idx_0))

    # Random global jumps
    for _ in range(num_neighbors * 2):
        o = random.choice(mutable_ones)
        z = random.choice(valid_zeros)
        if abs(o - z) > window_size:
            candidates.append((o, z))

    candidates = list(set(candidates))
    final_pairs = random.sample(candidates, min(num_neighbors, len(candidates)))

    neighbors = []
    for idx_1, idx_0 in final_pairs:
        new_sched = copy.deepcopy(schedule)
        new_sched[idx_1] = 0
        new_sched[idx_0] = 1
        neighbors.append(new_sched)
    return neighbors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="configs/search_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # [Path] Init
    prompt_hash = hashlib.md5(cfg.prompt.encode()).hexdigest()[:6]
    model_dir = model_path(cfg.model_name)
    output_dir = os.path.join(cfg.output_root, make_search_exp_name_wan(cfg, prompt_hash), os.environ["RUN_ID"])
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"wan_t2v_1_3B_{cfg.steps}nfe{cfg.nfe}_h{prompt_hash}.pt")

    logger = create_logger(output_dir, name="wan-search")
    logger.info("Running Hill-Climbing Cache Strategy Search")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Config: steps={cfg.steps}, nfe={cfg.nfe}, prompt_hash={prompt_hash}")
    logger.info(f"Prompt: {cfg.prompt[:80]}...")

    # Init model
    logger.info("Initializing Model...")
    model_cfg = WAN_CONFIGS[cfg.task]
    pipe = src.wan.WanT2V(
        config=model_cfg, checkpoint_dir=model_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False,
    )
    pipe.__class__.generate = t2v_generate

    # Generate teacher latents
    logger.info("Generating Teacher Latents...")
    pipe.model.enable_budcache = False
    gt_latents = get_latents(pipe, cfg)
    logger.info("Teacher Generated")

    # Init schedule with prefix protection
    current_schedule = [0] * cfg.steps
    strict_prefix_len = max(1, int(cfg.nfe * 0.3))

    for i in range(strict_prefix_len):
        current_schedule[i] = 1

    remaining_nfe = cfg.nfe - strict_prefix_len
    if remaining_nfe > 0:
        candidate_indices = list(range(strict_prefix_len, cfg.steps))
        random_indices = np.random.choice(candidate_indices, remaining_nfe, replace=False)
        for idx in random_indices:
            current_schedule[idx] = 1

    # Enable BudCache
    pipe.model.__class__.forward = budcache_forward
    pipe.model.num_steps = 2 * cfg.steps
    pipe.model.previous_residual_even = None
    pipe.model.previous_residual_odd = None
    pipe.model.enable_budcache = True
    pipe.model.cache_schedule = current_schedule
    pipe.model.cnt = 0

    current_latents = get_latents(pipe, cfg)
    current_loss = calc_mse(gt_latents, current_latents)
    best_schedule = copy.deepcopy(current_schedule)
    best_loss = current_loss

    logger.info(f"Initial Loss: {current_loss:.6f}")
    history = []

    # Simulated Annealing
    logger.info(f"Starting SA ({cfg.stage1_sa_iters} iters)...")
    t = cfg.stage1_sa_t_max
    alpha = (cfg.stage1_sa_t_min / cfg.stage1_sa_t_max) ** (1 / max(cfg.stage1_sa_iters, 1))

    pbar = tqdm(range(cfg.stage1_sa_iters), desc="SA Search")
    for i in pbar:
        candidate_schedule = mutate_schedule(current_schedule)
        pipe.model.cache_schedule = candidate_schedule
        pipe.model.cnt = 0

        cand_latents = get_latents(pipe, cfg)
        cand_loss = calc_mse(gt_latents, cand_latents)

        delta = cand_loss - current_loss
        if delta < 0 or random.random() < math.exp(-delta / t):
            current_schedule = candidate_schedule
            current_loss = cand_loss

            if current_loss < best_loss:
                best_loss = current_loss
                best_schedule = copy.deepcopy(current_schedule)

        t *= alpha
        history.append({"step": i, "loss": current_loss, "best_loss": best_loss, "stage": "SA"})
        pbar.set_postfix({"Best": f"{best_loss:.5f}", "Curr": f"{current_loss:.5f}", "T": f"{t:.4f}"})

    # Hill Climb Polish
    logger.info(f"Starting HC Polish ({cfg.stage2_hc_iters} iters)...")
    current_schedule = copy.deepcopy(best_schedule)
    current_loss = best_loss

    pbar = tqdm(range(cfg.stage2_hc_iters), desc="HC Polish")
    for i in pbar:
        neighbors = get_swap_neighbors(current_schedule, num_neighbors=20)
        step_best_loss = float('inf')
        step_best_schedule = None

        for candidate_schedule in neighbors:
            pipe.model.cache_schedule = candidate_schedule
            pipe.model.cnt = 0

            cand_latents = get_latents(pipe, cfg)
            cand_loss = calc_mse(gt_latents, cand_latents)

            if cand_loss < step_best_loss:
                step_best_loss = cand_loss
                step_best_schedule = candidate_schedule

        if step_best_loss < current_loss:
            current_loss = step_best_loss
            current_schedule = step_best_schedule

            if current_loss < best_loss:
                best_loss = current_loss
                best_schedule = copy.deepcopy(current_schedule)

            pbar.set_postfix({"Status": "Improve", "Best": f"{best_loss:.6f}"})
        else:
            pbar.set_postfix({"Status": "Stuck", "Best": f"{best_loss:.6f}"})

        history.append({"step": cfg.stage1_sa_iters + i, "loss": current_loss, "best_loss": best_loss, "stage": "HC"})

    # Save results
    final_indices = [idx for idx, val in enumerate(best_schedule) if val == 1]
    speedup = cfg.steps / len(final_indices)

    result_data = {
        "cache_step": [i for i in range(cfg.steps) if i not in final_indices],
        "best_loss": best_loss,
        "schedule_mask": best_schedule,
        "schedule_indices": final_indices,
        "speedup": speedup,
        "history": history
    }

    torch.save(result_data, save_path)
    logger.info(f"Results saved to {save_path}")
    logger.info(f"Final Loss: {best_loss:.6f} | Speedup: {speedup:.2f}x")

    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
