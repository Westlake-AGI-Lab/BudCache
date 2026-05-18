# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Adapted from Wan2.1 (https://github.com/Wan-Video/Wan2.1) WanModel.forward,
# licensed under the Apache License, Version 2.0.
# Modifications Copyright 2026 BudCache Authors: cache-step injection (budcache_forward).
# `teacache_forward` below is adapted from TeaCache (https://github.com/ali-vilab/TeaCache)
# and retained here as a reference baseline.
import math
import torch
import torch.nn as nn
import numpy as np
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from .modules.attention import flash_attention

def budcache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    enable_cache: bool = False,
    should_calc: bool = True,
    cache_branch: str | None = None,
):
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
    ])

    # time embeddings
    with torch.amp.autocast("cuda", dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
        
    if cache_branch is not None:
        if cache_branch not in {"cond", "uncond"}:
            raise ValueError(f"Unsupported cache_branch: {cache_branch}")
        residual_attr = f"previous_residual_{cache_branch}"
        if enable_cache:
            if not should_calc:
                cached = getattr(self, residual_attr, None)
                if cached is None:
                    raise RuntimeError(f"Missing cached residual for branch={cache_branch}.")
                x += cached.detach()
            else:
                ori_x = x
                for block in self.blocks:
                    x = block(x, **kwargs)
                setattr(self, residual_attr, (x - ori_x).detach())
        else:
            for block in self.blocks:
                x = block(x, **kwargs)
    elif self.enable_budcache:
        # ours
        should_calc_even = True
        should_calc_odd = True
        true_cnt = self.cnt//2
        if self.cnt%2 == 0: self.is_even=True
        else: self.is_even=False
        if self.cache_schedule[true_cnt] == 0:
            should_calc_even = False
            should_calc_odd = False
        else:
            should_calc_even = True
            should_calc_odd = True
            
        # if self.cnt%2==0: # even -> conditon
        # else: # odd -> unconditon

    if cache_branch is None and self.enable_budcache: 
        if self.is_even:
            if not should_calc_even:
                x += self.previous_residual_even
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_even = x - ori_x
        else:
            if not should_calc_odd:
                x += self.previous_residual_odd
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_odd = x - ori_x
    
    elif cache_branch is None:
        for block in self.blocks:
            x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    if cache_branch is None:
        self.cnt += 1
        if self.cnt >= self.num_steps:
            self.cnt = 0
    return [u.float() for u in x]

def teacache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
    ])

    # time embeddings
    with torch.amp.autocast("cuda", dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
        
    if self.enable_teacache:
        modulated_inp = e0 if self.use_ref_steps else e
        # teacache
        if self.cnt%2==0: # even -> conditon
            self.is_even = True
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(((modulated_inp-self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                    should_calc_even = False
                else:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_e0_even = modulated_inp.clone()

        else: # odd -> unconditon
            self.is_even = False
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            else: 
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp-self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    should_calc_odd = False
                else:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()

    if self.enable_teacache: 
        if self.is_even:
            if not should_calc_even:
                x += self.previous_residual_even
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_even = x - ori_x
        else:
            if not should_calc_odd:
                x += self.previous_residual_odd
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_odd = x - ori_x
    
    else:
        for block in self.blocks:
            x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
    return [u.float() for u in x]

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()
