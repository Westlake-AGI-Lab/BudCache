# Adapted from Black Forest Labs FLUX (https://github.com/black-forest-labs/flux),
# licensed under the Apache License, Version 2.0.
# Modifications Copyright 2026 BudCache Authors.
import math
from typing import Callable

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).to(device)
    
def prepare(img: Tensor, prompt: str | list[str], guidance, clip_encoder, t5_encoder, clip_tokenizer, t5_tokenizer) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    device = img.device
    weight_dtype = img.dtype
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> (h w) c",)
    # img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    clip_tokens = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length,
                                      truncation=True, return_overflowing_tokens=False, 
                                      return_length=False, return_tensors="pt").input_ids.to(device)
    pooled_prompt_embeds = clip_encoder(clip_tokens, output_hidden_states=False).pooler_output.to(dtype=weight_dtype)
    t5_tokens = t5_tokenizer(prompt, padding="max_length", max_length=512, 
                             truncation=True, return_tensors="pt").input_ids.to(device)
    prompt_embeds = t5_encoder(t5_tokens)[0].to(dtype=weight_dtype)
    txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device)
    
    guidance = torch.full([1], guidance, device=device, dtype=torch.float32)
    guidance = guidance.expand(img.shape[0])
    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "prompt_embeds": prompt_embeds.to(img.device),
        "text_ids": txt_ids.to(img.device),
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "guidance": guidance,
    }
    
def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)   
def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b
def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)
    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()

def denoising_euler(
    transformer, latents, sigmas, guidance,
    pooled_prompt_embeds, prompt_embeds, text_ids, image_ids,
    ) -> torch.Tensor:
    for k in range(len(sigmas)-1):
        sigma_k, sigma_next = sigmas[k], sigmas[k+1]
        t_in = torch.full((latents.shape[0],), sigma_k, device=latents.device, dtype=latents.dtype)
        v = transformer(
            hidden_states=latents,
            timestep=t_in,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
            return_dict=False,
        )[0]
        latents = latents + (sigma_next-sigma_k)*v
    return latents

def denoising_ipndm2(
    transformer, latents, sigmas, guidance,
    pooled_prompt_embeds, prompt_embeds, text_ids, image_ids,
    ) -> torch.Tensor:
    prev_v = None
    for k in range(len(sigmas) - 1):
        sigma_k, sigma_next = sigmas[k], sigmas[k + 1]
        dt = sigma_next - sigma_k
        t_in = torch.full((latents.shape[0],), sigma_k, device=latents.device, dtype=latents.dtype)
        v = transformer(
            hidden_states=latents,
            timestep=t_in,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
            return_dict=False,
        )[0]
        if prev_v is None:
            # Euler warm start for the first step, then switch to the 2nd-order
            # pseudo linear multistep update used by iPNDM.
            latents = latents + dt * v
        else:
            latents = latents + dt * (1.5 * v - 0.5 * prev_v)
        prev_v = v
    return latents

def _flow_lambda(sigma: float, device: torch.device) -> torch.Tensor:
    # Flow matching parameterization uses:
    #   x_t = (1 - sigma_t) * x0 + sigma_t * eps
    # so alpha_t = 1 - sigma_t and lambda_t = log(alpha_t) - log(sigma_t).
    sigma_t = torch.tensor(sigma, device=device, dtype=torch.float32).clamp(1e-12, 1 - 1e-12)
    alpha_t = 1.0 - sigma_t
    return torch.log(alpha_t) - torch.log(sigma_t)

def denoising_dpmpp2m(
    transformer, latents, sigmas, guidance,
    pooled_prompt_embeds, prompt_embeds, text_ids, image_ids,
    ) -> torch.Tensor:
    prev_x0 = None
    prev_lambda = None
    for k in range(len(sigmas) - 1):
        sigma_k, sigma_next = float(sigmas[k]), float(sigmas[k + 1])
        t_in = torch.full((latents.shape[0],), sigma_k, device=latents.device, dtype=latents.dtype)
        v = transformer(
            hidden_states=latents,
            timestep=t_in,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
            return_dict=False,
        )[0]
        # DPM-Solver++ is a data-prediction solver. For flow prediction,
        #   v = eps - x0  and  x_t = (1 - sigma_t) * x0 + sigma_t * eps,
        # hence the corresponding data prediction is
        #   x0_hat = x_t - sigma_t * v_hat.
        x0 = latents.float() - sigma_k * v.float()
        if sigma_next <= 0.0:
            # Final zero-noise step: in the limit sigma_{t+1} -> 0, the sample
            # collapses directly to the data prediction x0_hat.
            latents = x0.to(dtype=latents.dtype)
        else:
            lambda_k = _flow_lambda(sigma_k, latents.device)
            lambda_next = _flow_lambda(sigma_next, latents.device)
            h = lambda_next - lambda_k
            alpha_next = torch.tensor(1.0 - sigma_next, device=latents.device, dtype=torch.float32)
            expm1_neg_h = torch.expm1(-h)
            if prev_x0 is None:
                # First-order DPM-Solver++ update:
                #   x_{t+1} = (sigma_{t+1}/sigma_t) * x_t
                #             - alpha_{t+1} * (exp(-h) - 1) * x0_hat_t
                x_next = (sigma_next / sigma_k) * latents.float() - alpha_next * expm1_neg_h * x0
            else:
                h_0 = lambda_k - prev_lambda
                r0 = h_0 / h
                # Two-step midpoint form used by DPM-Solver++(2M):
                #   D0 = x0_hat_t
                #   D1 = (x0_hat_t - x0_hat_{t-1}) / r0
                #   x_{t+1} = (sigma_{t+1}/sigma_t) * x_t
                #             - alpha_{t+1} * (exp(-h) - 1) * D0
                #             - 0.5 * alpha_{t+1} * (exp(-h) - 1) * D1
                D0 = x0
                D1 = (x0 - prev_x0) / r0
                x_next = (
                    (sigma_next / sigma_k) * latents.float()
                    - alpha_next * expm1_neg_h * D0
                    - 0.5 * alpha_next * expm1_neg_h * D1
                )
            latents = x_next.to(dtype=latents.dtype)
            prev_lambda = lambda_k
        prev_x0 = x0
    return latents

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

def vae_decode(latents, vae, image_processor,):
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    with torch.amp.autocast('cuda', enabled=False):
        vae.to(dtype=torch.float32)
        latents = latents.to(dtype=torch.float32).contiguous()
        image_tensor = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image_tensor.detach().cpu(), output_type="pil")
    return image
