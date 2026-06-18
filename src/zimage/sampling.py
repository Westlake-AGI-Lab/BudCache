import math
import torch
from typing import List


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    """Sample the initial Gaussian latent for Z-Image.

    Z-Image latents have 16 channels and spatial size 2 * (pixels // 16), where the
    factor 16 = vae_scale_factor(8) * patch(2) matches ZImagePipeline.prepare_latents.
    Latents are kept in float32 (the denoise loop casts to the transformer dtype per
    step), so the sampler stays numerically identical to the reference pipeline.
    """
    latent_h = 2 * (height // 16)
    latent_w = 2 * (width // 16)
    return torch.randn(
        num_samples,
        16,
        latent_h,
        latent_w,
        dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).to(device)


@torch.no_grad()
def encode_prompt(
    prompts: List[str],
    tokenizer,
    text_encoder,
    max_sequence_length: int = 512,
    device: torch.device = "cuda",
) -> List[torch.Tensor]:
    """Encode prompts into Z-Image caption features (one variable-length tensor per prompt).

    Re-implements ZImagePipeline._encode_prompt directly so there is no pipeline
    nesting. Z-Image uses a single Qwen-style chat LLM as the text encoder:
      1. wrap each prompt in a chat turn and apply the chat template,
      2. tokenize to a fixed max length with an attention mask,
      3. take the second-to-last hidden state as the embedding,
      4. drop padded positions, returning a list of [seq_i, dim] tensors.
    CFG is not used (Turbo runs at guidance_scale=0), so only positive prompts are encoded.
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    templated = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        templated.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        )

    text_inputs = tokenizer(
        templated,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    masks = text_inputs.attention_mask.to(device).bool()

    hidden = text_encoder(
        input_ids=input_ids, attention_mask=masks, output_hidden_states=True
    ).hidden_states[-2]

    # keep only the real (unpadded) tokens for each prompt
    return [hidden[i][masks[i]] for i in range(len(templated))]


def get_schedule(
    num_steps: int,
    shift: float = 3.0,
    use_dynamic_shifting: bool = False,
    image_seq_len: int = None,
    base_image_seq_len: int = 256,
    max_image_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> List[float]:
    """Build the Z-Image (FlowMatchEuler) sigma schedule, EXACTLY mirroring the official
    ZImagePipeline, returning num_steps+1 sigmas. Here `num_steps` == num_inference_steps.

    Reproduces diffusers FlowMatchEulerDiscreteScheduler.set_timesteps with the pipeline's
    `sigma_min = 0.0` override:
      1. linear sigma ramp over num_steps points: linspace(1, 0, num_steps) (last point = 0
         because sigma_min is forced to 0),
      2. apply the shift,
      3. append the always-added terminal sigma 0.
    Because step 1 already ends at 0 and step 3 appends another 0, the final two sigmas are
    both 0, so the LAST denoise step has dt = 0 (a no-op forward: the transformer still runs
    but its output is multiplied by 0 and does not change the latent). This wasted final
    forward is part of the official behavior and is intentionally preserved so the cache-step
    indices line up 1:1 with the official inference path.

    Two shifting modes are supported (both kept so no logic is lost; pick via scheduler config):

      - Static (use_dynamic_shifting=False): resolution-independent, controlled by `shift`
        (Z-Image base=6.0, Z-Image-Turbo=3.0):
            sigma' = shift * sigma / (1 + (shift - 1) * sigma)

      - Dynamic (use_dynamic_shifting=True): resolution-dependent. `mu` is linearly
        interpolated from image_seq_len between (base_image_seq_len, base_shift) and
        (max_image_seq_len, max_shift), then the exponential time-shift is applied:
            sigma' = exp(mu) / (exp(mu) + (1 / sigma - 1))
        Note this is algebraically the static formula with shift = exp(mu); at sigma=0 the
        1/sigma term is +inf so sigma' -> 0 cleanly.

    Z-Image's current checkpoints use the static mode, but the dynamic branch is retained
    for full parity with the official / diffusers pipeline.
    """
    base = torch.linspace(1, 0, num_steps)                       # num_steps points, last = 0 (sigma_min=0)
    if use_dynamic_shifting:
        if image_seq_len is None:
            raise ValueError("image_seq_len required when use_dynamic_shifting=True")
        # mu = calculate_shift(image_seq_len): linear interp of shift vs sequence length
        m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
        b = base_shift - m * base_image_seq_len
        mu = image_seq_len * m + b
        # exponential time-shift (diffusers _time_shift_exponential with sigma exponent = 1.0)
        shifted = math.exp(mu) / (math.exp(mu) + (1.0 / base - 1.0))
    else:
        # static, resolution-independent shift
        shifted = shift * base / (1 + (shift - 1) * base)
    sigmas = torch.cat([shifted, torch.zeros(1)])               # append terminal 0 -> double-0 tail
    return sigmas.tolist()


@torch.no_grad()
def denoising_search(
    transformer,
    latents: torch.Tensor,
    sigmas: List[float],
    cap_feats: List[torch.Tensor],
    mode: str,
    cache_step: list = None,
    neg_cap_feats: List[torch.Tensor] = None,
    guidance_scale: float = 0.0,
) -> torch.Tensor:
    """Run the Z-Image denoising loop for cache-step search (flow-match Euler, with optional CFG).

    Re-implements the ZImagePipeline denoising loop without any scheduler/pipeline
    nesting, and supports both model variants:
      - Z-Image-Turbo: guidance_scale <= 1  -> no CFG, one forward per step,
      - Z-Image (base): guidance_scale  > 1  -> classifier-free guidance.
    Per step i (sigma_i -> sigma_{i+1}):
      - the model timestep is 1 - sigma_i  (pipeline's (1000 - t)/1000 with t = sigma_i*1000),
      - the latent gains an F axis (unsqueeze(2)) and is passed as a list of [C,1,H,W],
      - when CFG is on, the batch is duplicated as [cond; uncond] in a single forward and
        combined as v = v_cond + guidance_scale * (v_cond - v_uncond),
      - the velocity is negated (Z-Image predicts -v),
      - Euler update: x <- x + (sigma_{i+1} - sigma_i) * (-v).
    CFG follows the pipeline defaults (apply every step, no cfg_truncation / cfg_normalization);
    those non-default knobs are intentionally not modeled because they would change the batch
    composition across steps and break the constant-shape residual cache.
    The transformer's cache state is configured here per mode so the search can reuse one
    loaded model across thousands of evaluations.

    Args:
        mode: "teacher" (full compute, no cache) or "student" (apply cache_step).
        cache_step: step indices to skip (required for student mode).
        neg_cap_feats: negative-prompt caption features; required when guidance_scale > 1.
        guidance_scale: CFG scale; <= 1 disables CFG (single forward per step).
    """
    num_steps = len(sigmas) - 1
    transformer.num_steps = num_steps

    # configure cache state on the transformer
    if mode == "teacher":
        transformer.enable_budcache = False
    else:
        if cache_step is None:
            raise ValueError("cache_step required for student mode")
        transformer.enable_budcache = True
        transformer.cache_step = cache_step
        transformer.budcache_cnt = 0
        transformer.previous_residual = None

    bsz = latents.shape[0]
    do_cfg = guidance_scale > 1.0
    if do_cfg:
        if neg_cap_feats is None:
            raise ValueError("neg_cap_feats required when guidance_scale > 1")
        # one forward per step over a doubled batch ordered [cond; uncond]
        cond_feats = list(cap_feats) + list(neg_cap_feats)

    x = latents  # float32 [B, 16, H, W]
    for i in range(num_steps):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Model timestep, computed bit-for-bit like the pipeline. The scheduler timestep is
        # t = sigma_i*1000 and the pipeline feeds (1000 - t)/1000 (all in fp32). This equals
        # 1 - sigma_i mathematically, but evaluating the exact same fp32 expression avoids a
        # ~1e-8 ULP drift (vs computing 1 - sigma_i) that would otherwise perturb the timestep
        # embedding and amplify through the bf16 denoise chain.
        sig32 = torch.tensor(sigma_i, dtype=torch.float32)
        t_val = float(((1000.0 - sig32 * 1000.0) / 1000.0).item())

        if do_cfg:
            latent_in = x.to(transformer.dtype).repeat(2, 1, 1, 1).unsqueeze(2)  # [2B, C, 1, H, W]
            t_in = torch.full((2 * bsz,), t_val, device=x.device, dtype=torch.float32)
            feats = cond_feats
        else:
            latent_in = x.to(transformer.dtype).unsqueeze(2)                     # [B, C, 1, H, W]
            t_in = torch.full((bsz,), t_val, device=x.device, dtype=torch.float32)
            feats = cap_feats

        out_list = transformer(list(latent_in.unbind(dim=0)), t_in, feats, return_dict=False)[0]
        v_all = torch.stack([o.float() for o in out_list], dim=0).squeeze(2)     # [*, C, H, W]

        if do_cfg:
            v_cond = v_all[:bsz]
            v_uncond = v_all[bsz:]
            v = v_cond + guidance_scale * (v_cond - v_uncond)
        else:
            v = v_all

        # Z-Image predicts negated velocity; Euler step toward lower sigma
        x = x + (sigma_next - sigma_i) * (-v)

    return x


@torch.no_grad()
def vae_decode(latents: torch.Tensor, vae, image_processor):
    """Decode Z-Image latents [B, 16, h, w] into PIL images.

    Mirrors ZImagePipeline's decode exactly: unscale/shift the latents into the VAE's
    input space, decode, then postprocess to PIL. Z-Image latents are NOT packed (the 2x2
    patchify happens inside the transformer), so unlike FLUX there is no unpack step here.
    """
    latents = latents.to(vae.dtype)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]
    return image_processor.postprocess(image, output_type="pil")
