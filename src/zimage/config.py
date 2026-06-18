from dataclasses import dataclass, field
from typing import List


@dataclass
class ZImageInferenceConfig:
    # Z-Image / Z-Image-Turbo eval; model resolved as MODEL_ROOT + model_name.
    model_name: str = "Z-Image-Turbo"
    dataset_name: str = "debug"
    dataset_path: str = "dataset/evaluation/debug.txt"
    output_root: str = "outputs/Eval"

    height: int = 1024
    width: int = 1024
    steps: int = 9               # = num_inference_steps (mirrors pipeline; last step is dt=0 no-op)
    guidance_scale: float = 0.0  # Turbo: 0 (no CFG); base Z-Image: >1 turns CFG on
    seed: int = 42
    batch_size: int = 1
    samples_per_prompt: int = 1
    max_sequence_length: int = 512


@dataclass
class ZImageSearchConfig:
    model_name: str = "Z-Image"
    output_root: str = "outputs/search"
    weight_dtype: str = "bf16"

    height: int = 1024
    width: int = 1024
    steps: int = 9               # = num_inference_steps (mirrors pipeline; last step is dt=0 no-op)
    guidance_scale: float = 0.0  # Turbo: 0 (no CFG); base Z-Image: >1 turns CFG on
    seed: int = 42
    max_sequence_length: int = 512  # Z-Image text encoder max tokens
    prompts: List[str] = field(default_factory=lambda: ["A cat sitting on a windowsill"])

    # search: target NFE (number of computed steps; the other steps reuse the cached residual)
    nfe: int = 5
    # search schedule: Random restarts -> Simulated Annealing -> Hill Climbing
    stage0_multiStart: int = 2
    stage1_sa_iters: int = 200
    stage1_sa_t_max: float = 0.05
    stage1_sa_t_min: float = 1e-5
    stage2_hc_iters: int = 20
