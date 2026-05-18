from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.wan.utils.fm_solvers import get_sampling_sigmas


def build_sampling_sigmas(num_steps: int, shift: float, device: torch.device) -> torch.Tensor:
    sigmas = get_sampling_sigmas(int(num_steps), float(shift))
    sigmas = np.concatenate([sigmas, np.array([0.0], dtype=np.float32)])
    return torch.tensor(sigmas, device=device, dtype=torch.float32)


def pack_contexts(contexts: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([ctx.shape[0] for ctx in contexts], device=contexts[0].device, dtype=torch.long)
    max_len = int(lengths.max().item())
    hidden_dim = contexts[0].shape[-1]
    padded = contexts[0].new_zeros(len(contexts), max_len, hidden_dim)
    for idx, ctx in enumerate(contexts):
        padded[idx, : ctx.shape[0]] = ctx
    return padded, lengths


class LearnableSigmasV1(nn.Module):
    def __init__(
        self,
        steps: int,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        eps: float = 1e-8,
        init_sigmas: torch.Tensor | None = None,
    ):
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.steps = int(steps)
        self.eps = float(eps)
        if init_sigmas is not None:
            if len(init_sigmas) != self.steps + 1:
                raise ValueError(f"Expected {self.steps + 1} init sigmas, got {len(init_sigmas)}")
            self.sigma_max = float(init_sigmas[0].item())
            self.sigma_min = float(init_sigmas[-1].item())
            gaps = (init_sigmas[:-1] - init_sigmas[1:]).clamp_min(self.eps)
            self.params = nn.Parameter(torch.log(gaps))
        else:
            self.sigma_min = float(sigma_min)
            self.sigma_max = float(sigma_max)
            self.params = nn.Parameter(torch.zeros(self.steps))

    def forward(self) -> torch.Tensor:
        weights = F.softmax(self.params, dim=0)
        total = self.sigma_max - self.sigma_min
        gaps = weights * total
        interior = self.sigma_max - torch.cumsum(gaps, dim=0)
        return torch.cat(
            [
                interior.new_tensor([self.sigma_max]),
                interior,
            ]
        )


class WanRunner(nn.Module):
    def __init__(
        self,
        model,
        learn_sigmas: nn.Module,
        num_train_timesteps: int,
        seq_len: int,
        guidance_scale: float,
        param_dtype: torch.dtype,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.model = model
        self.learn_sigmas = learn_sigmas
        self.num_train_timesteps = int(num_train_timesteps)
        self.seq_len = int(seq_len)
        self.guidance_scale = float(guidance_scale)
        self.param_dtype = param_dtype
        self.use_checkpoint = bool(use_checkpoint)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @staticmethod
    def _unpack_contexts(padded: torch.Tensor, lengths: torch.Tensor) -> List[torch.Tensor]:
        return [padded[idx, : int(lengths[idx].item())] for idx in range(padded.shape[0])]

    def _cfg_step(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        context_null: torch.Tensor,
        context_null_lens: torch.Tensor,
        enable_cache: bool = False,
        should_calc: bool = True,
    ) -> torch.Tensor:
        latents_list = list(latents.unbind(0))
        context_list = self._unpack_contexts(context, context_lens)
        context_null_list = self._unpack_contexts(context_null, context_null_lens)
        use_autocast = latents.device.type == "cuda" and self.param_dtype in (torch.float16, torch.bfloat16)
        with torch.amp.autocast("cuda", dtype=self.param_dtype, enabled=use_autocast):
            if getattr(self.model, "_supports_cache_flags", False):
                pred_cond = torch.stack(
                    self.model(
                        latents_list,
                        t=timestep,
                        context=context_list,
                        seq_len=self.seq_len,
                        enable_cache=enable_cache,
                        should_calc=should_calc,
                        cache_branch="cond",
                    )
                )
                pred_uncond = torch.stack(
                    self.model(
                        latents_list,
                        t=timestep,
                        context=context_null_list,
                        seq_len=self.seq_len,
                        enable_cache=enable_cache,
                        should_calc=should_calc,
                        cache_branch="uncond",
                    )
                )
            else:
                pred_cond = torch.stack(self.model(latents_list, t=timestep, context=context_list, seq_len=self.seq_len))
                pred_uncond = torch.stack(
                    self.model(latents_list, t=timestep, context=context_null_list, seq_len=self.seq_len)
                )
        return pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)

    def _rollout(
        self,
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        context_null: torch.Tensor,
        context_null_lens: torch.Tensor,
        use_checkpoint: bool,
        enable_cache: bool,
    ) -> torch.Tensor:
        x = latents
        cache_schedule = getattr(self.model, "cache_schedule", None) if enable_cache else None
        enable_cache = bool(cache_schedule)
        supports_cache = getattr(self.model, "_supports_cache_flags", False)
        for step_idx in range(len(sigmas) - 1):
            sigma_k = sigmas[step_idx]
            sigma_next = sigmas[step_idx + 1]
            timestep = sigma_k.expand(x.shape[0]).to(device=x.device, dtype=torch.float32)
            timestep = timestep * float(self.num_train_timesteps)
            calc_now = (not enable_cache) or bool(cache_schedule[step_idx])
            def _one_step(
                x_,
                t_,
                context_,
                context_lens_,
                context_null_,
                context_null_lens_,
                _calc=calc_now,
                _cache=enable_cache,
                _supports=supports_cache,
            ):
                return self._cfg_step(
                    x_,
                    t_,
                    context_,
                    context_lens_,
                    context_null_,
                    context_null_lens_,
                    enable_cache=_cache if _supports else False,
                    should_calc=_calc,
                )
            if use_checkpoint and calc_now:
                velocity = checkpoint(
                    _one_step,
                    x,
                    timestep,
                    context,
                    context_lens,
                    context_null,
                    context_null_lens,
                    use_reentrant=False,
                )
            else:
                velocity = _one_step(x, timestep, context, context_lens, context_null, context_null_lens)
            delta = (sigma_next - sigma_k).to(dtype=x.dtype).view(1, 1, 1, 1, 1)
            x = x + delta * velocity
        return x

    def forward(
        self,
        latents: torch.Tensor,
        teacher_sigmas: torch.Tensor | None,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        context_null: torch.Tensor,
        context_null_lens: torch.Tensor,
        mode: str = "teacher",
    ) -> torch.Tensor:
        if mode == "teacher":
            if teacher_sigmas is None:
                raise ValueError("teacher_sigmas is required in teacher mode")
            with torch.no_grad():
                return self._rollout(
                    latents=latents,
                    sigmas=teacher_sigmas,
                    context=context,
                    context_lens=context_lens,
                    context_null=context_null,
                    context_null_lens=context_null_lens,
                    use_checkpoint=False,
                    enable_cache=False,
                )
        if mode == "student":
            return self._rollout(
                latents=latents,
                sigmas=self.learn_sigmas(),
                context=context,
                context_lens=context_lens,
                context_null=context_null,
                context_null_lens=context_null_lens,
                use_checkpoint=self.use_checkpoint,
                enable_cache=True,
            )
        raise ValueError(f"Unknown mode: {mode}")
