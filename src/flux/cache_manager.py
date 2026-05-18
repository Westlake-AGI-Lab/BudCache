from diffusers import FluxTransformer2DModel

from .dit_forward import transformer_budcache_forward

ORIGINAL_FORWARD = FluxTransformer2DModel.forward

_CACHE_ATTRS = (
    "enable_budcache",
    "cache_step",
    "budcache_cnt",
    "num_steps",
    "previous_residual",
)


def reset2original(transformer):
    for attr in _CACHE_ATTRS:
        if hasattr(transformer, attr):
            delattr(transformer, attr)
    cls = transformer.__class__
    for attr in _CACHE_ATTRS:
        if attr in cls.__dict__:
            delattr(cls, attr)
    FluxTransformer2DModel.forward = ORIGINAL_FORWARD


def set_budcache(transformer, cache_step, num_steps):
    reset2original(transformer)
    FluxTransformer2DModel.forward = transformer_budcache_forward
    transformer.enable_budcache = True
    transformer.cache_step = list(cache_step)
    transformer.budcache_cnt = 0
    transformer.num_steps = int(num_steps)
    transformer.previous_residual = None
