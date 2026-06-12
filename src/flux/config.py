import os
from dataclasses import dataclass, field
from typing import List, Optional


def _model_path(*parts: str) -> str:
    model_root = os.environ.get("MODEL_ROOT") or os.environ.get("MODEL_DIR") or ""
    return os.path.join(model_root, *parts) if model_root else os.path.join(*parts)


FLUX_LORA_REGISTRY = {
    "hypersd": {
        "path": _model_path("lora_weights", "Hyper-SD"),
        "weight_name": "Hyper-FLUX.1-dev-8steps-lora.safetensors",
        "adapter_name": "hyper",
        "adapter_weight": 0.125,
        "default_steps": 8,
    },
}


@dataclass
class FluxInferenceConfig:
    model_name: str = "black-forest-labs/FLUX.1-dev"
    dataset_name: str = "GenEval"
    dataset_path: str = "data/eval/geneval.txt"
    output_root: str = "outputs/Eval"

    height: int = 1024
    width: int = 1024
    steps: int = 28
    guidance_scale: float = 3.5
    seed: int = 42
    batch_size: int = 1
    samples_per_prompt: int = 1
    use_hypersd: bool = False


@dataclass
class SearchConfig:
    # basic — model resolved as MODEL_ROOT + model_name; run dir under output_root
    model_name: str = "FLUX.1-dev"
    output_root: str = "outputs/search"

    height: int = 1024
    width: int = 1024
    steps: int = 28
    guidance_scale: float = 3.5
    seed: int = 42
    weight_dtype: str = "bf16"
    prompts: List[str] = field(default_factory=lambda: ["A cat sitting on a windowsill"])

    nfe: int = 10
    stage0_multiStart: int = 1
    stage1_sa_iters: int = 200
    stage1_sa_t_max: float = 0.05
    stage1_sa_t_min: float = 1e-5
    stage2_hc_iters: int = 20


@dataclass
class TrainingConfig:
    # basic — model resolved as MODEL_ROOT + model_name; run dir under output_root
    model_name: str = "FLUX.1-dev"
    output_root: str = "outputs/train"
    mixed_precision: str = "bf16"
    # dataset & dataloader
    dataset: str = "coco-5000"
    prompts_file: str = "data/train/val2014_N100.jsonl"
    train_batch_size: int = 1
    dataloader_num_workers: int = 4
    # loss function
    loss: str="lpips" # lpips | mse
    # optimizer
    gradient_accumulation_steps: int = 4
    optimizer: str="adamw"
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    # learning scheduler
    learning_rate: float = 5e-4
    lr_scheduler_type: str = "cosine" #Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    lr_warmup_ratio: float = 0.05
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    # training & checkpoints
    max_train_steps: Optional[int] = None
    num_train_epochs: int = 1
    ckpt_save_steps: int = 64
    visual_steps: int = 32
    # teacher / scheduler
    height: int = 1024
    width: int = 1024
    scheduler_mu: float = 1.15
    guidance_scale: float = 3.5
    teacher_steps: int = 28
    student_steps: int = 7
    student_nfe: int = 10
    use_traj_loss: bool=False
    sigmas_learn_mode: str="ld3" # ld3 | offset
    sigmas_optimize_range: str="all" # "all" | "cache"
    sigmas_resume_path: Optional[str] = None
    sigmas_init_mode: str="flux" # "uniform" | "flux"
    init_cache_mode: str = "" # "base" | "stage1"
    stage1_cacheStep: Optional[str] = None
