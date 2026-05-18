import torch, numpy as np, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from diffusers.utils import is_torch_version, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from typing import Optional, Dict, Any, Union

class FluxRunner(nn.Module):
    def __init__(self, transformer, learn_sigmas):
        super().__init__()
        self.transformer = transformer
        self.learn_sigmas = learn_sigmas

    def forward(
        self, latents: torch.Tensor,
        teacher_sigmas,
        guidance,
        pooled_prompt_embeds, prompt_embeds,
        text_ids, image_ids,
        mode: str = "teacher", return_traj: bool=False,
    ) -> torch.Tensor:
        if mode == "student":
            return self._student_forward(
                latents=latents,
                sigmas=self.learn_sigmas(),
                guidance=guidance,
                pooled_prompt_embeds=pooled_prompt_embeds,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                image_ids=image_ids,
                return_traj=return_traj,
            )
        elif mode == "teacher":
            with torch.no_grad():
                return self._teacher_forward(
                    latents=latents,
                    sigmas=teacher_sigmas,
                    guidance=guidance,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    image_ids=image_ids,
                    return_traj=return_traj,
                )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    def _teacher_forward(
        self, latents: torch.Tensor, sigmas, guidance,
        pooled_prompt_embeds, prompt_embeds, text_ids, image_ids,
        return_traj=False,
    ) -> torch.Tensor:
        x = latents
        trajectory = [] if return_traj else None
        for k in range(len(sigmas) - 1):
            sigma_k, sigma_next = sigmas[k], sigmas[k+1]
            t_in = sigma_k.expand(latents.shape[0]).to(latents.dtype)
            v = self.transformer(
                hidden_states=x,
                timestep=t_in,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=image_ids,
                return_dict=False,
            )[0]
            x = x + (sigma_next - sigma_k) * v
            if return_traj: trajectory.append(x)
        return trajectory if return_traj else x

    def _student_forward(
        self, latents: torch.Tensor, sigmas: torch.Tensor, guidance,
        pooled_prompt_embeds, prompt_embeds, text_ids, image_ids,
        return_traj=False,
    ) -> torch.Tensor:
        x = latents
        cache_list = getattr(self.transformer, "cache_step", None)
        supports  = getattr(self.transformer, "_supports_cache_flags", False)
        trajectory = [] if return_traj else None
        for k in range(len(sigmas) - 1):
            sigma_k, sigma_next = sigmas[k], sigmas[k + 1]
            t_in = sigma_k.expand(latents.shape[0]).to(latents.dtype)
            enable_cache = cache_list is not None
            calc_now  = (not enable_cache) or (k not in cache_list)
            def _one_step(x_, t_, pooled_, prompt_, txt_, img_, _calc=calc_now, _cache=enable_cache, _supports=supports):
                if _supports:
                    return self.transformer(
                        hidden_states=x_, timestep=t_, guidance=guidance,
                        pooled_projections=pooled_, encoder_hidden_states=prompt_,
                        txt_ids=txt_, img_ids=img_,
                        return_dict=False,
                        should_calc=_calc, enable_cache=_cache,
                    )[0]
                else:
                    return self.transformer(
                        hidden_states=x_, timestep=t_, guidance=guidance,
                        pooled_projections=pooled_, encoder_hidden_states=prompt_,
                        txt_ids=txt_, img_ids=img_,
                        return_dict=False,
                    )[0]
            if calc_now:
                v = checkpoint(
                    _one_step,
                    x, t_in, pooled_prompt_embeds, prompt_embeds, text_ids, image_ids,
                    use_reentrant=False,
                )
            else:
                v = _one_step(x, t_in, pooled_prompt_embeds, prompt_embeds, text_ids, image_ids)
            delta_sigma = (sigma_next - sigma_k).to(x.dtype)
            x = x + delta_sigma * v
            if return_traj: trajectory.append(x)
        return trajectory if return_traj else x

####### Learnable Parameters ######
"""
Trainable timestep schedulers initialized from a standard inference schedule.
The learned sigmas are constrained to stay monotonic during optimization.
"""

class LearnableSigmasV1(nn.Module):
    """Learn K+1 monotonically decreasing sigmas in [sigma_min, sigma_max]."""
    def __init__(self, steps: int, sigma_min: float = 0.0, sigma_max: float = 1.0, eps: float = 1e-8, init_sigmas=None):
        super().__init__()
        assert steps >= 1
        self.K = steps
        self.eps = float(eps)
        if init_sigmas is not None:
            assert self.K == len(init_sigmas)-1
            self.sigma_max = init_sigmas[0].item()
            self.sigma_min = init_sigmas[-1].item()
            p0 = invert_sigmas_to_params(init_sigmas, self.sigma_min, self.sigma_max, s0=1.0/(self.K+1))
            self.params = nn.Parameter(p0)
        else:
            self.sigma_min = float(sigma_min)
            self.sigma_max = float(sigma_max)
            self.params = nn.Parameter(torch.ones(self.K + 1))
    def forward(self) -> torch.Tensor:
        s  = F.softmax(self.params, dim=0)
        c  = torch.cumsum(s, dim=0).flip(0)
        den= (c[0] - c[-1]).clamp_min(self.eps)
        u  = (c - c[-1]) / den                               # [1..0]
        sigmas = u * (self.sigma_max - self.sigma_min) + self.sigma_min
        return sigmas
    
class LearnableSigmasV2_offset(nn.Module):
    """Learn bounded offsets for the selected schedule positions."""
    def __init__(self, steps: int, init_sigmas: torch.Tensor, cache_step: list, bound: float = 0.005, optimize_all = "all"):
        super().__init__()
        self.register_buffer('base', init_sigmas)
        mask = torch.zeros_like(init_sigmas)
        if optimize_all == "all":
            mask [1:-1] = 1.0
        else:
            valid_cache = [i for i in cache_step if 0 < i < steps]
            mask[valid_cache] = 1.0; 
        self.register_buffer('mask', mask)
        self.delta_logits = nn.Parameter(torch.zeros_like(init_sigmas))
        self.bound = bound
    def forward(self):
        offset = torch.tanh(self.delta_logits) * self.bound
        return self.base + offset * self.mask
class LearnableSigmasV2(nn.Module):
    def __init__(self, steps: int, init_sigmas: torch.Tensor, cache_step: list):
        super().__init__()
        self.register_buffer('base', init_sigmas)
        self.segments = []
        self.params = nn.ParameterList()
        compute_idx = get_segments(steps, cache_step)
        for start, end in zip(compute_idx[:-1], compute_idx[1:]):
            n_gaps = end - start
            if n_gaps > 1:
                seg_vals = init_sigmas[start : end+1]
                base_gaps = seg_vals[:-1] - seg_vals[1:]
                self.params.append(nn.Parameter(torch.log(base_gaps.clamp(min=1e-6))))
                self.segments.append((start, end, True))
            else:
                self.params.append(nn.Parameter(torch.empty(0)))
                self.segments.append((start, end, False))
    def forward(self):
        out = self.base.clone()
        for i, (s, e, trainable) in enumerate(self.segments):
            if trainable:
                total_drop = self.base[s] - self.base[e]
                weights = torch.softmax(self.params[i], dim=0) 
                gaps = weights * total_drop
                out[s+1 : e] = self.base[s] - torch.cumsum(gaps, dim=0)[:-1]
        return out
def get_segments(steps, cache_indices):
    """Return fixed compute-step indices used to split trainable segments."""
    all_indices = set(range(steps + 1))
    cache_set = set(cache_indices)
    compute_indices = sorted(list((all_indices - cache_set) | {0, steps}))
    return compute_indices

def invert_sigmas_to_params(
    given_sigmas: torch.Tensor,
    sigma_min: float = 0.0,
    sigma_max: float = 1.0,
    s0: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Invert a monotonic sigma schedule into LearnableSigmasV1 parameters."""
    sig = given_sigmas.detach().to(torch.float64)
    N = sig.shape[0]
    assert N >= 2, "Need at least 2 sigmas"
    denom = max(sigma_max - sigma_min, eps)
    u = (sig - sigma_min) / denom
    if s0 is None:
        s0 = 1.0 / N
    s0 = float(s0)
    assert 0.0 < s0 < 1.0, "s0 must be in (0,1)"
    A = 1.0 - s0
    s = torch.empty_like(u)
    s[0] = s0
    if N >= 2:
        s[1] = A * u[N - 2]
    for i in range(2, N - 1):
        s[i] = A * (u[N - 1 - i] - u[N - i])
    if N >= 2:
        s[N - 1] = A * (1.0 - u[1])
    s = s.clamp_min(eps)
    s = s / s.sum()
    params = torch.log(s)
    return params.to(torch.float32)


# replace transformer.forward with cache's version forward
def transformer_cache_forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None, timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None, txt_ids: torch.Tensor = None, guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None, controlnet_single_block_samples=None,
        return_dict: bool = True, controlnet_blocks_repeat: bool = False,
        enable_cache: bool = False, should_calc: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})
        if enable_cache:
            if not should_calc:
                hidden_states += self.previous_residual.detach()
            else:
                ori_hidden_states = hidden_states.detach().clone()
                for index_block, block in enumerate(self.transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)
                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )
                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )
                    # controlnet residual
                    if controlnet_block_samples is not None:
                        interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        # For Xlabs ControlNet.
                        if controlnet_blocks_repeat:
                            hidden_states = (
                                hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                            )
                        else:
                            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
                for index_block, block in enumerate(self.single_transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)
                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )
                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )
                    # controlnet residual
                    if controlnet_single_block_samples is not None:
                        interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                            hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                            + controlnet_single_block_samples[index_block // interval_control]
                        )
                self.previous_residual = (hidden_states - ori_hidden_states).detach()
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                # controlnet residual
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    # For Xlabs ControlNet.
                    if controlnet_blocks_repeat:
                        hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                        )
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
            for index_block, block in enumerate(self.single_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                # controlnet residual
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        if USE_PEFT_BACKEND: # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
