import argparse
import hashlib
import math
import os
import random
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from diffusers import FluxPipeline, FluxTransformer2DModel
from tqdm.auto import tqdm

from src.flux.config import FLUX_LORA_REGISTRY, SearchConfig
from src.flux.dit_forward import transformer_budcache_forward
from src.flux.sampling import *
from src.util import create_logger, make_search_exp_name_flux, model_path

def load_config(path: str) -> SearchConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a mapping config at {path}, got {type(raw).__name__}")
    return SearchConfig(**raw)

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
    model_dir = model_path(cfg.model_name)
    output_dir = os.path.join(cfg.output_root, make_search_exp_name_flux(cfg, prompt_hash), os.environ["RUN_ID"])
    os.makedirs(output_dir, exist_ok=True)
    history_save_path = os.path.join(output_dir, f"searchHistory_{cfg.steps}nfe{cfg.nfe}.pt")
    final_save_path = os.path.join(output_dir, f"flux_{cfg.steps}nfe{cfg.nfe}_h{prompt_hash}.pt")

    # Create logger
    logger = create_logger(output_dir, name="flux-search")
    logger.info("Running Hill-Climbing Cache Strategy Search")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Config: steps={cfg.steps}, nfe={cfg.nfe}, prompt_hash={prompt_hash}")
    logger.info(f"Prompts ({len(cfg.prompts)}):")
    for i, p in enumerate(cfg.prompts):
        logger.info(f"  [{i}] {p[:80]}...")

    # [Important] Modify Forward Function
    run_device = "cuda"
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    FluxTransformer2DModel.forward = transformer_budcache_forward
    # Load models
    logger.info("Loading FLUX model...")
    pipe = FluxPipeline.from_pretrained(model_dir, torch_dtype=weight_dtype).to(run_device)
    # Load Hyper-SD LoRA
    hypersd = FLUX_LORA_REGISTRY["hypersd"]
    pipe.load_lora_weights(
        hypersd["path"],
        weight_name=hypersd["weight_name"],
        adapter_name=hypersd["adapter_name"],
    )
    pipe.set_adapters([hypersd["adapter_name"]], adapter_weights=[hypersd["adapter_weight"]]) # important
    logger.info("Model loaded successfully")

    # Prepare inputs
    batch_prompts = cfg.prompts
    bsz = len(batch_prompts)
    noises = get_noise(num_samples=bsz, height=cfg.height, width=cfg.width,
                    device="cuda", dtype=weight_dtype, seed=int(cfg.seed))
    image_seq_len = (noises.shape[-1]*noises.shape[-2])//4
    sigmas = get_schedule(num_steps=int(cfg.steps), image_seq_len=image_seq_len,)
    inp = prepare(img=noises, prompt=batch_prompts, guidance=float(cfg.guidance_scale),
                clip_encoder=pipe.text_encoder, t5_encoder=pipe.text_encoder_2,
                clip_tokenizer=pipe.tokenizer, t5_tokenizer=pipe.tokenizer_2)

    # Generate teacher latent
    logger.info("Generating teacher latent...")
    teacher_latent = denoising_search(
        transformer=pipe.transformer, latents=inp["img"], sigmas=sigmas, guidance=inp["guidance"],
        pooled_prompt_embeds=inp["pooled_prompt_embeds"], prompt_embeds=inp["prompt_embeds"],
        text_ids=inp["text_ids"], image_ids=inp["img_ids"], mode="teacher", cache_step=[])
    logger.info("Teacher latent generated")

    # Initialize search
    mse_loss_fn = torch.nn.MSELoss()
    loss_cache = {}
    global_best_loss = float('inf')
    global_best_cache = []

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
        if k0 not in loss_cache: # init Loss
            loss_cache[k0] = _gen_student_img_loss(
                loss_fn=mse_loss_fn, transformer=pipe.transformer, latents=inp["img"], sigmas=sigmas, guidance=inp["guidance"],
                pooled_prompt_embeds=inp["pooled_prompt_embeds"], prompt_embeds=inp["prompt_embeds"],
                text_ids=inp["text_ids"], image_ids=inp["img_ids"],
                teacher_latent=teacher_latent, cache_steps=cur_cache,)
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
            cand = _gen_cand_sa(cur_cache,cfg.steps,nfe=cfg.nfe)
            k = _cache_key(cand)
            if k not in loss_cache:
                loss_cache[k] = _gen_student_img_loss(
                    loss_fn=mse_loss_fn, transformer=pipe.transformer, latents=inp["img"], sigmas=sigmas, guidance=inp["guidance"],
                    pooled_prompt_embeds=inp["pooled_prompt_embeds"], prompt_embeds=inp["prompt_embeds"],
                    text_ids=inp["text_ids"], image_ids=inp["img_ids"],
                    teacher_latent=teacher_latent, cache_steps=cand,)
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
                    loss_cache[k] = _gen_student_img_loss(
                        loss_fn=mse_loss_fn, transformer=pipe.transformer, latents=inp["img"], sigmas=sigmas, guidance=inp["guidance"],
                        pooled_prompt_embeds=inp["pooled_prompt_embeds"], prompt_embeds=inp["prompt_embeds"],
                        text_ids=inp["text_ids"], image_ids=inp["img_ids"],
                        teacher_latent=teacher_latent, cache_steps=cand,)
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
                if cur_loss < global_best_loss: # update global best
                    add, rm = cache_delta(cur_cache, global_best_cache) # log than update
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
            else: #
                logger.info(f"[Polish] Converged at iter {p_it}. Checked {checked_count} Neighbors")
                break
    total_time = time.time() - _t0
    logger.info(f"[Time] {total_time:.2f}s ({_fmt_hms(total_time)}) | cache_entries={len(loss_cache)}")
    logger.info(f"[Result] best_loss={global_best_loss:.6f}, NFE={cfg.steps - len(global_best_cache)}")
    logger.info(f"[Result] cache_steps={global_best_cache}")

    torch.save( # Save result
        {
            "cache_step": global_best_cache,
            "best_loss": global_best_loss,
        },
        final_save_path,
    )
    torch.save(search_history, history_save_path)
    search_history = torch.load(history_save_path, map_location="cpu")
    plot_search_trajectory(search_history, os.path.join(output_dir, "search_history.png"))
    logger.info(f"[Saved] Results saved to {final_save_path}")
    
# Important function
def _compute_from_cache(cur_cache,total_steps):
	cur_cache_set=set(cur_cache)
	cur_compute=[i for i in range(total_steps) if i not in cur_cache_set]
	cur_compute.sort()
	return cur_compute
def _cache_from_compute(comp,total_steps):
	comp_set=set(comp)
	return [i for i in range(1,total_steps) if i not in comp_set]  #0nevercached
def _dedup_compute_list(neigh_compute,cur_compute):
	seen=set()
	out=[]
	cur_key=tuple(cur_compute)
	for comp in neigh_compute:
		key=tuple(comp)
		if key==cur_key or key in seen:continue
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
def _gen_neighbor_hc(cur_cache,total_steps,W=3,jumps_per_point=2):
	"""
	Small-neighborhood(best-practice):
	-Deterministic local relocate within window [k-W,k+W] for each movable compute step
	-Plus a few random far jumps per compute step (helps escape shallow local minima)
	-NFE preserved (single-point replacement in compute set)
	"""
	nfe=total_steps-len(cur_cache)
	prefix=max(1,nfe//4)
	cur_compute=_compute_from_cache(cur_cache,total_steps)
	cur_compute_set=set(cur_compute)
	neigh_compute=[]
	#A:localrelocate(withinwindow)
	for p,k in enumerate(cur_compute):
		if k<prefix:continue
		lo=max(prefix,k-W)
		hi=min(total_steps-1,k+W)
		for j in range(lo,hi+1):
			if j==k or j in cur_compute_set:continue
			new_comp=list(cur_compute)
			new_comp[p]=j
			new_comp.sort()
			neigh_compute.append(new_comp)
	#B:randomfarjump(outsidewindowpreferred)
	all_slots=list(range(prefix,total_steps))
	for p,k in enumerate(cur_compute):
		if k<prefix:continue
		for _ in range(jumps_per_point):
			target=random.choice(all_slots)
			if target in cur_compute_set or target==k:continue
			new_comp=list(cur_compute)
			new_comp[p]=target
			new_comp.sort()
			neigh_compute.append(new_comp)
	neigh_compute=_dedup_compute_list(neigh_compute,cur_compute)
	return [_cache_from_compute(comp,total_steps) for comp in neigh_compute]
def _cache_key(cache_steps):
    return tuple(sorted(set(cache_steps)))
def cache_delta(prev_cache, new_cache):
    a = set(prev_cache); b = set(new_cache)
    add = sorted(b - a)
    rm  = sorted(a - b)
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

def denoising_search(
    transformer, latents: torch.Tensor, sigmas, guidance,
    pooled_prompt_embeds, prompt_embeds, text_ids, image_ids,
    mode: str,
    cache_step: list = None,
) -> torch.Tensor:
    """
    Denoising loop for search optimization.

    This function is designed for search scenarios where cache_step
    changes frequently. It automatically configures transformer state.

    Args:
        mode: "teacher" (no cache) or "student" (with cache)
        cache_step: List of timestep indices to skip (required for student mode)
    """
    # Configure cache state based on mode
    if mode == "teacher":
        transformer.enable_budcache = False
    else:  # student mode
        if cache_step is None:
            raise ValueError("cache_step required for student mode")
        transformer.enable_budcache = True
        transformer.cache_step = cache_step
        transformer.budcache_cnt = 0
        transformer.previous_residual = None

    x = latents
    for k in range(len(sigmas)-1):
        sigma_k, sigma_next = sigmas[k], sigmas[k+1]
        t_in = torch.full((latents.shape[0],), sigma_k, device=latents.device, dtype=latents.dtype)
        v = transformer(
            hidden_states=x,
            timestep=t_in,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
            return_dict=False,
        )[0]
        x = x + (sigma_next-sigma_k)*v
    return x
@torch.no_grad()
def _gen_student_img_loss(
    loss_fn, transformer, sigmas, latents, 
    guidance, pooled_prompt_embeds, prompt_embeds, text_ids, image_ids,
    teacher_latent,
    cache_steps: List[int],
) -> float:
    student_latent = denoising_search(
        transformer,
        latents=latents, sigmas=sigmas, guidance=guidance,
        pooled_prompt_embeds=pooled_prompt_embeds, prompt_embeds=prompt_embeds,
        text_ids=text_ids, image_ids=image_ids,
        mode="student", cache_step=cache_steps,
    )
    loss = loss_fn(student_latent.float(), teacher_latent.float())
    return float(loss.item())
def _fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"
def plot_search_trajectory(history, save_path, title="Optimization Trajectory"):
    """
    Plot search trajectory with restarts overlaid on same axes.
    Different restarts use different colors, SA uses circles, HC uses triangles.
    """
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Extract data
    data = {}
    for h in history:
        r = h['restart']
        if r not in data:
            data[r] = {'loss': [], 'best_loss': [], 'stage': []}
        data[r]['loss'].append(h['loss'])
        data[r]['best_loss'].append(h['best_loss'])
        data[r]['stage'].append(h['stage'])

    # Generate colors for restarts
    n_restarts = len(data)
    colors = cm.tab10(np.linspace(0, 1, n_restarts)) if n_restarts <= 10 else cm.tab20(np.linspace(0, 1, n_restarts))

    # Plot each restart
    for r_idx, (r, d) in enumerate(sorted(data.items())):
        losses = np.array(d['loss'])
        stages = d['stage']
        color = colors[r_idx]

        # SA points (circles)
        sa_mask = np.array([s == 'SA' for s in stages])
        if sa_mask.any():
            ax.scatter(np.where(sa_mask)[0], losses[sa_mask],
                      s=20, c=[color], marker='o', alpha=0.6,
                      label=f'R{r} SA' if r_idx == 0 else '', zorder=2)

        # HC points (triangles)
        hc_mask = np.array([s == 'HC' for s in stages])
        if hc_mask.any():
            ax.scatter(np.where(hc_mask)[0], losses[hc_mask],
                      s=30, c=[color], marker='^', alpha=0.8,
                      label=f'R{r} HC' if r_idx == 0 else '', zorder=3)

    # Global best line (light)
    all_best = [h['best_loss'] for h in history]
    ax.plot(range(len(all_best)), all_best, color='gray', linewidth=1.5,
            alpha=0.3, label='Global Best', zorder=1)

    # Final annotation
    final_loss = all_best[-1]
    ax.annotate(f"Final: {final_loss:.5f}",
                xy=(len(all_best)-1, final_loss), xycoords='data',
                xytext=(-40, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='black', lw=1),
                fontsize=9, fontweight='bold')

    ax.set_xlabel("Iteration", fontsize=10, fontweight='bold')
    ax.set_ylabel("Latent MSE Loss", fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, pad=15)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend with custom markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, label='SA', alpha=0.6),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=7, label='HC', alpha=0.8),
        Line2D([0], [0], color='gray', linewidth=1.5, alpha=0.3, label='Global Best')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    main()
