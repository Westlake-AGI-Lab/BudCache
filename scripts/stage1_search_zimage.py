import argparse
import hashlib
import math
import os
import random
import time
from typing import List, Optional

import torch
import yaml
from diffusers import ZImagePipeline, ZImageTransformer2DModel
from tqdm.auto import tqdm

from src.zimage.config import ZImageSearchConfig
from src.zimage.dit_forward import transformer_zimage_budcache_forward
from src.zimage.sampling import get_noise
from src.util import create_logger, make_search_exp_name_zimage

# Diffusers-pipeline variant of stage1_search_zimage.py.

def load_config(path: str) -> ZImageSearchConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a mapping config at {path}, got {type(raw).__name__}")
    return ZImageSearchConfig(**raw)


def _set_cache_state(transformer, cache_steps: Optional[List[int]]):
    """Configure the cache-aware transformer.forward for a single denoise.

    cache_steps=None -> teacher (full compute, cache disabled).
    cache_steps=list -> student: enable cache, skip those step indices, and re-zero the
    per-generation counters so the reused-residual bookkeeping starts clean every eval
    (mirrors src.zimage.sampling.denoising_search's student setup).
    """
    if cache_steps is None:
        transformer.enable_budcache = False
    else:
        transformer.enable_budcache = True
        transformer.cache_step = list(cache_steps)
        transformer.budcache_cnt = 0
        transformer.previous_residual = None


@torch.no_grad()
def _run_pipe_latent(pipe, prompt_embeds, neg_arg, noise, cfg, guidance_scale):
    """
    Run one full denoise through the stock pipeline and return the final latent.
    """
    return pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_arg,
        height=int(cfg.height),
        width=int(cfg.width),
        num_inference_steps=int(cfg.steps),
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        latents=noise.clone(),
        output_type="latent",
        return_dict=False,
    )[0]


@torch.no_grad()
def main():
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default=None, help="YAML File Path")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Generate prompt hash
    prompts_str = "||".join(cfg.prompts)
    prompt_hash = hashlib.md5(prompts_str.encode()).hexdigest()[:6]

    # [Config] Search params
    restarts = int(cfg.stage0_multiStart)
    sa_iters = int(cfg.stage1_sa_iters)
    sa_t_max = float(cfg.stage1_sa_t_max)
    sa_t_min = float(cfg.stage1_sa_t_min)
    polish_max_iters = int(cfg.stage2_hc_iters)

    # [Path] Init
    model_path = os.path.join(os.environ["MODEL_ROOT"], cfg.model_name)
    output_dir = os.path.join(cfg.output_root, make_search_exp_name_zimage(cfg, prompt_hash), os.environ["RUN_ID"])
    os.makedirs(output_dir, exist_ok=True)
    history_save_path = os.path.join(output_dir, f"searchHistory_{cfg.steps}nfe{cfg.nfe}.pt")
    final_save_path = os.path.join(output_dir, f"zimage_{cfg.steps}nfe{cfg.nfe}_h{prompt_hash}.pt")

    # Create logger
    logger = create_logger(output_dir, name=f"{cfg.model_name}-search")
    logger.info("Running Hill-Climbing Cache Strategy Search (diffusers backend)")
    logger.info("Backend: diffusers ZImagePipeline.__call__ (denoising via stock pipeline)")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Config: steps={cfg.steps}, nfe={cfg.nfe}, prompt_hash={prompt_hash}")
    logger.info(f"Prompts ({len(cfg.prompts)}):")
    for i, p in enumerate(cfg.prompts):
        logger.info(f"  [{i}] {p[:80]}...")

    # [Important] Modify Forward Function: install the cache-aware forward globally. The pipeline
    # calls self.transformer(...) internally, so this single patch makes BudCache transparent to it.
    # Teacher vs student is then selected per eval via the `enable_budcache` flag (_set_cache_state).
    run_device = "cuda"
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    ZImageTransformer2DModel.forward = transformer_zimage_budcache_forward
    # Load model
    logger.info("Loading Z-Image model...")
    pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=weight_dtype).to(run_device)
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.num_steps = int(cfg.steps)  # counter wraps here; = pipeline denoise length
    logger.info("Model loaded successfully")

    # Prepare inputs: fixed initial noise (passed to the pipeline via `latents=` every call) and
    # caption features encoded once. The pipeline's flow-match Euler schedule (static shift for
    # Z-Image base / Turbo, read from its own scheduler config) is built internally from steps.
    batch_prompts = cfg.prompts
    bsz = len(batch_prompts)
    guidance_scale = float(cfg.guidance_scale)
    do_cfg = guidance_scale > 1.0  # base Z-Image: CFG on; Turbo: 0 (single forward per step)
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=batch_prompts,
        negative_prompt=None,
        do_classifier_free_guidance=do_cfg,
        device=run_device,
        max_sequence_length=int(cfg.max_sequence_length),
    )
    neg_arg = negative_prompt_embeds if do_cfg else None
    latents = get_noise(num_samples=bsz, height=cfg.height, width=cfg.width,
                        device=run_device, dtype=torch.float32, seed=int(cfg.seed))
    logger.info(f"guidance_scale={guidance_scale}, cfg={'on' if do_cfg else 'off'}")

    # Generate teacher latent (full compute, cache disabled)
    logger.info("Generating teacher latent...")
    _set_cache_state(pipe.transformer, None)
    teacher_latent = _run_pipe_latent(pipe, prompt_embeds, neg_arg, latents, cfg, guidance_scale)
    logger.info("Teacher latent generated")

    # Initialize search
    mse_loss_fn = torch.nn.MSELoss()
    loss_cache = {}
    global_best_loss = float('inf')
    global_best_cache = []

    def eval_loss(cache_steps):
        _set_cache_state(pipe.transformer, cache_steps)
        student_latent = _run_pipe_latent(pipe, prompt_embeds, neg_arg, latents, cfg, guidance_scale)
        return float(mse_loss_fn(student_latent.float(), teacher_latent.float()).item())

    search_history = []
    global_step_counter = 0
    _t0 = time.time()
    for r in range(restarts):
        # ==============================================================
        # Stage 0: Initialization (Random Start)
        # ==============================================================
        cur_cache = random_cache_steps(
            cfg.steps, nfe=cfg.nfe,
            no_cache_prefix=max(1, cfg.nfe // 4),
        )
        k0 = _cache_key(cur_cache)
        if k0 not in loss_cache:  # init Loss
            loss_cache[k0] = eval_loss(cur_cache)
        cur_loss = loss_cache[k0]
        search_history.append({
            'global_step': global_step_counter,
            'restart': r,
            'stage': 'Init',
            'loss': cur_loss,
            'best_loss': global_best_loss
        })
        global_step_counter += 1
        logger.info(f"[Start] Restart {r+1}/{restarts} | Init Loss: {cur_loss:.6f}")
        # ==============================================================
        # Stage 1: Simulated Annealing (Global Exploration)
        # ==============================================================
        sa_bar = tqdm(range(sa_iters), desc=f"R{r+1} SA", leave=False,)
        for it in sa_bar:
            progress = it / sa_iters
            temp = sa_t_max * (sa_t_min / sa_t_max) ** progress
            cand = _gen_cand_sa(cur_cache, cfg.steps, nfe=cfg.nfe)
            k = _cache_key(cand)
            if k not in loss_cache:
                loss_cache[k] = eval_loss(cand)
            l = loss_cache[k]
            delta = l - cur_loss
            accept = False
            if delta < 0:
                accept = True
            else:
                prob = math.exp(-delta / temp)
                if random.random() < prob:
                    accept = True
            if accept:
                cur_cache = cand
                cur_loss = l
                if cur_loss < global_best_loss:
                    global_best_loss = cur_loss
                    global_best_cache = list(cur_cache)
                    sa_bar.set_postfix({"Best": f"{global_best_loss:.6f}"})
            search_history.append({
                'global_step': global_step_counter,
                'restart': r,
                'stage': 'SA',
                'loss': cur_loss,
                'best_loss': global_best_loss
            })
            global_step_counter += 1
        logger.info(f"[SA Done] Loss after SA: {cur_loss:.6f}")
        # ==============================================================
        # Stage 2: Steepest Ascent Hill Climbing (Local Exploitation)
        # ==============================================================
        for p_it in range(polish_max_iters):
            # 1. gen all neighbor
            neighs = _gen_neighbor_hc(cur_cache, cfg.steps)
            if not neighs: break
            # 2. find Best Improvement
            local_best_cand = None
            local_best_loss = cur_loss
            checked_count = 0
            best_rejected_loss = float('inf')
            polish_neighs_bar = tqdm(neighs, desc="HillClimbing...", leave=False,)
            for cand in polish_neighs_bar:
                k = _cache_key(cand)
                if k not in loss_cache:
                    loss_cache[k] = eval_loss(cand)
                l = loss_cache[k]
                checked_count += 1
                if l < local_best_loss:
                    local_best_loss = l
                    local_best_cand = cand
                    polish_neighs_bar.set_postfix({"Best": f"{local_best_loss:.6f}"})
                if l <= best_rejected_loss and l > cur_loss:
                    best_rejected_loss = l
            # 3. decision
            if local_best_cand is not None:
                diff = cur_loss - local_best_loss
                cur_cache = local_best_cand
                cur_loss = local_best_loss
                if cur_loss < global_best_loss:  # update global best
                    add, rm = cache_delta(cur_cache, global_best_cache)  # log than update
                    global_best_loss = cur_loss
                    global_best_cache = list(cur_cache)
                    logger.info(f"[Polish] iter={p_it} FOUND BETTER: {cur_loss:.6f} (-{diff:.6f}) add+{add}&move-{rm}")
                search_history.append({
                    'global_step': global_step_counter,
                    'restart': r,
                    'stage': 'HC',
                    'loss': cur_loss,
                    'best_loss': global_best_loss
                })
                global_step_counter += 1
            else:  #
                logger.info(f"[Polish] Converged at iter {p_it}. Checked {checked_count} Neighbors")
                break
    total_time = time.time() - _t0
    logger.info(f"[Time] {total_time:.2f}s ({_fmt_hms(total_time)}) | cache_entries={len(loss_cache)}")
    logger.info(f"[Result] best_loss={global_best_loss:.6f}, NFE={cfg.steps - len(global_best_cache)}")
    logger.info(f"[Result] cache_steps={global_best_cache}")

    torch.save(  # Save result
        {
            "cache_step": global_best_cache,
            "best_loss": global_best_loss,
        },
        final_save_path,
    )
    torch.save(search_history, history_save_path)
    search_history = torch.load(history_save_path, map_location="cpu")
    logger.info(f"[Saved] Results saved to {final_save_path}")


# Important function
def _compute_from_cache(cur_cache, total_steps):
    cur_cache_set = set(cur_cache)
    cur_compute = [i for i in range(total_steps) if i not in cur_cache_set]
    cur_compute.sort()
    return cur_compute


def _cache_from_compute(comp, total_steps):
    comp_set = set(comp)
    return [i for i in range(1, total_steps) if i not in comp_set]  # 0 never cached


def _dedup_compute_list(neigh_compute, cur_compute):
    seen = set()
    out = []
    cur_key = tuple(cur_compute)
    for comp in neigh_compute:
        key = tuple(comp)
        if key == cur_key or key in seen: continue
        seen.add(key)
        out.append(comp)
    return out


def _gen_cand_sa(cur_cache, total_steps, nfe):
    prefix = max(1, nfe // 4)
    cur_cache_set = set(cur_cache)
    all_indices = set(range(total_steps))
    cur_compute = sorted(list(all_indices - cur_cache_set))
    movable_compute = [c for c in cur_compute if c >= prefix]
    valid_cache_targets = [c for c in cur_cache if c >= prefix]
    if not movable_compute or not valid_cache_targets:
        return cur_cache
    rand_val = random.random()
    if rand_val < 0.9 or len(movable_compute) < 2 or len(valid_cache_targets) < 2:
        src = random.choice(movable_compute)
        if random.random() < 0.7:
            radius = 3
            local_targets = [t for t in valid_cache_targets if abs(t - src) <= radius]
            if local_targets:
                dst = random.choice(local_targets)
            else:
                dst = random.choice(valid_cache_targets)
        else:
            dst = random.choice(valid_cache_targets)
        new_compute_set = set(cur_compute)
        new_compute_set.remove(src)
        new_compute_set.add(dst)
    else:
        srcs = random.sample(movable_compute, 2)
        dsts = random.sample(valid_cache_targets, 2)
        new_compute_set = set(cur_compute)
        for s in srcs: new_compute_set.remove(s)
        for d in dsts: new_compute_set.add(d)
    new_compute_list = sorted(list(new_compute_set))
    return _cache_from_compute(new_compute_list, total_steps)


def _gen_neighbor_hc(cur_cache, total_steps, W=3, jumps_per_point=2):
    """
    Small-neighborhood (best-practice):
    - Deterministic local relocate within window [k-W, k+W] for each movable compute step
    - Plus a few random far jumps per compute step (helps escape shallow local minima)
    - NFE preserved (single-point replacement in compute set)
    """
    nfe = total_steps - len(cur_cache)
    prefix = max(1, nfe // 4)
    cur_compute = _compute_from_cache(cur_cache, total_steps)
    cur_compute_set = set(cur_compute)
    neigh_compute = []
    # A: local relocate (within window)
    for p, k in enumerate(cur_compute):
        if k < prefix: continue
        lo = max(prefix, k - W)
        hi = min(total_steps - 1, k + W)
        for j in range(lo, hi + 1):
            if j == k or j in cur_compute_set: continue
            new_comp = list(cur_compute)
            new_comp[p] = j
            new_comp.sort()
            neigh_compute.append(new_comp)
    # B: random far jump (outside window preferred)
    all_slots = list(range(prefix, total_steps))
    for p, k in enumerate(cur_compute):
        if k < prefix: continue
        for _ in range(jumps_per_point):
            target = random.choice(all_slots)
            if target in cur_compute_set or target == k: continue
            new_comp = list(cur_compute)
            new_comp[p] = target
            new_comp.sort()
            neigh_compute.append(new_comp)
    neigh_compute = _dedup_compute_list(neigh_compute, cur_compute)
    return [_cache_from_compute(comp, total_steps) for comp in neigh_compute]


def _cache_key(cache_steps):
    return tuple(sorted(set(cache_steps)))


def cache_delta(prev_cache, new_cache):
    a = set(prev_cache); b = set(new_cache)
    add = sorted(b - a)
    rm = sorted(a - b)
    return add, rm


def random_cache_steps(num_steps: int, nfe: int, no_cache_prefix: int = 3,):
    # forced compute prefix: [0, 1, ..., no_cache_prefix-1]
    compute_steps = set(range(no_cache_prefix))
    # remaining compute budget
    remaining = nfe - no_cache_prefix
    # sample additional compute steps from the tail [no_cache_prefix, ..., num_steps-1]
    candidates = list(range(no_cache_prefix, num_steps))
    extra_compute = random.sample(candidates, remaining)
    compute_steps.update(extra_compute)
    compute_steps = sorted(compute_steps)
    cache_step = [i for i in range(num_steps) if i not in compute_steps]
    return cache_step


def _fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


if __name__ == "__main__":
    main()
