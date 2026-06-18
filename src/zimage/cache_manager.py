import torch
from diffusers import ZImageTransformer2DModel

from .dit_forward import transformer_zimage_budcache_forward

# Capture the pristine (native diffusers) forward at import time, BEFORE anything patches it.
# Importing this module before any monkey-patching guarantees ORIGINAL_FORWARD is the real one.
ORIGINAL_FORWARD = ZImageTransformer2DModel.forward

# Instance/class attributes our cache forward attaches; cleared on reset so a transformer
# can be flipped back to exact native behavior.
_CACHE_ATTRS = (
    "enable_budcache",
    "cache_step",
    "budcache_cnt",
    "num_steps",
    "previous_residual",
)


def reset2original(transformer):
    """Restore the native ZImageTransformer2DModel.forward and strip all cache state.

    After this call the transformer behaves exactly like a freshly loaded diffusers model,
    which is what the `diffusers` reference path and the no-cache `ours` path both require.
    """
    for attr in _CACHE_ATTRS:
        if hasattr(transformer, attr):
            delattr(transformer, attr)
    cls = transformer.__class__
    for attr in _CACHE_ATTRS:
        if attr in cls.__dict__:
            delattr(cls, attr)
    ZImageTransformer2DModel.forward = ORIGINAL_FORWARD


def set_budcache(transformer, cache_step, num_steps):
    """Enable BudCache: patch in the cache-aware forward and initialize cache state.

    cache_step is the list of step indices whose transformer compute is skipped (residual
    reused). num_steps must equal the denoising loop length so the per-step counter wraps
    correctly. denoising() re-zeros budcache_cnt / previous_residual at the start of every
    generation, so this is safe to call once and reuse across a DDP eval loop.
    """
    reset2original(transformer)
    ZImageTransformer2DModel.forward = transformer_zimage_budcache_forward
    transformer.enable_budcache = True
    transformer.cache_step = list(cache_step)
    transformer.budcache_cnt = 0
    transformer.num_steps = int(num_steps)
    transformer.previous_residual = None
