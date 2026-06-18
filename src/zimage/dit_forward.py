import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence
from diffusers.models.modeling_outputs import Transformer2DModelOutput


# replace ZImageTransformer2DModel.forward with the cache-aware version.
# Cache protocol mirrors the FLUX budcache forward (src/flux/dit_forward.py):
#   - self.enable_budcache : bool  -> whether caching is active
#   - self.cache_step      : List[int] -> step indices whose compute is skipped
#   - self.budcache_cnt    : int   -> current denoising-step counter
#   - self.previous_residual : Tensor -> cached delta across the main blocks
#   - self.num_steps       : int   -> total denoising steps (counter wraps here)
#
# The residual is captured around the 30 main `self.layers` (the dominant cost),
# exactly like FLUX caches the delta across its transformer blocks. The cheap
# per-step pieces (patchify_and_embed, noise_refiner, context_refiner, building
# the `unified` sequence) always run so the cached delta is added on top of the
# current latent's embedded input:
#   compute step : previous_residual = unified_after_layers - unified_before_layers
#   cache   step : unified = unified + previous_residual   (skip the 30 layers)
def transformer_zimage_budcache_forward(
    self,
    x: List[torch.Tensor],
    t,
    cap_feats: List[torch.Tensor],
    patch_size=2,
    f_patch_size=1,
    return_dict: bool = True,
):
    assert patch_size in self.all_patch_size
    assert f_patch_size in self.all_f_patch_size

    bsz = len(x)
    device = x[0].device
    t = t * self.t_scale
    t = self.t_embedder(t)

    (
        x,
        cap_feats,
        x_size,
        x_pos_ids,
        cap_pos_ids,
        x_inner_pad_mask,
        cap_inner_pad_mask,
    ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

    # x embed & refine
    x_item_seqlens = [len(_) for _ in x]
    assert all(_ % 32 == 0 for _ in x_item_seqlens)  # SEQ_MULTI_OF
    x_max_item_seqlen = max(x_item_seqlens)

    x = torch.cat(x, dim=0)
    x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

    # Match t_embedder output dtype to x for layerwise casting compatibility
    adaln_input = t.type_as(x)
    x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
    x = list(x.split(x_item_seqlens, dim=0))
    x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(_) for _ in x_pos_ids], dim=0))

    x = pad_sequence(x, batch_first=True, padding_value=0.0)
    x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
    x_freqs_cis = x_freqs_cis[:, : x.shape[1]]

    x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(x_item_seqlens):
        x_attn_mask[i, :seq_len] = 1

    for layer in self.noise_refiner:
        x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

    # cap embed & refine
    cap_item_seqlens = [len(_) for _ in cap_feats]
    cap_max_item_seqlen = max(cap_item_seqlens)

    cap_feats = torch.cat(cap_feats, dim=0)
    cap_feats = self.cap_embedder(cap_feats)
    cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
    cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
    cap_freqs_cis = list(
        self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split([len(_) for _ in cap_pos_ids], dim=0)
    )

    cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
    cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
    cap_freqs_cis = cap_freqs_cis[:, : cap_feats.shape[1]]

    cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(cap_item_seqlens):
        cap_attn_mask[i, :seq_len] = 1

    for layer in self.context_refiner:
        cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

    # unified = concat(image tokens, caption tokens) per item, then padded
    unified = []
    unified_freqs_cis = []
    for i in range(bsz):
        x_len = x_item_seqlens[i]
        cap_len = cap_item_seqlens[i]
        unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
        unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
    unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
    assert unified_item_seqlens == [len(_) for _ in unified]
    unified_max_item_seqlen = max(unified_item_seqlens)

    unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
    unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
    unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(unified_item_seqlens):
        unified_attn_mask[i, :seq_len] = 1

    # ===== BudCache: cache the residual across the main `self.layers` =====
    enable_budcache = getattr(self, "enable_budcache", False)
    if enable_budcache:
        should_calc = self.budcache_cnt not in self.cache_step
        if not should_calc:
            # reuse cached delta: skip the 30 main blocks entirely
            unified = unified + self.previous_residual.detach()
        else:
            ori_unified = unified.detach().clone()
            for layer in self.layers:
                unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)
            self.previous_residual = (unified - ori_unified).detach()
        # advance step counter, wrap at num_steps
        self.budcache_cnt = self.budcache_cnt + 1
        if self.budcache_cnt == self.num_steps:
            self.budcache_cnt = 0
    else:
        for layer in self.layers:
            unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

    unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
    unified = list(unified.unbind(dim=0))
    x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

    if not return_dict:
        return (x,)
    return Transformer2DModelOutput(sample=x)
