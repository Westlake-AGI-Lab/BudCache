from dataclasses import dataclass
from typing import Optional


@dataclass
class WanSearchConfig:
    # basic — model resolved as MODEL_ROOT + model_name; run dir under output_root
    task: str = "t2v-1.3B"
    model_name: str = "wan2.1-t2v-1.3B"
    output_root: str = "outputs/search"
    offload_model: bool = True

    # generation
    size: str = "832*480"
    frame_num: int = 81
    seed: int = 42
    steps: int = 50
    solver: str = "unipc"
    shift: float = 5.0
    guidance_scale: float = 5.0

    # search
    nfe: int = 20
    stage0_multiStart: int = 1
    stage1_sa_iters: int = 200
    stage1_sa_t_max: float = 0.1
    stage1_sa_t_min: float = 1e-4
    stage2_hc_iters: int = 5

    # input
    prompt: str = "A detailed close-up of a mechanical eye opening, gears turning inside, highly realistic, 8k resolution, cinematic lighting."


@dataclass
class WanInferenceConfig:
    # basic — model resolved as MODEL_ROOT + model_name; run dir under output_root
    task: str = "t2v-1.3B"
    model_name: str = "wan2.1-t2v-1.3B"
    output_root: str = "outputs/eval"

    # generation
    size: str = "832*480"
    frame_num: int = 81
    seed: int = 42
    steps: int = 50
    solver: str = "euler"
    shift: float = 5.0
    guidance_scale: float = 5.0
    offload_model: bool = True

    # dataset
    dataset_path: str = "data/eval/video_prompt_100.txt"
    dataset_name: str = "gpt2"
    samples_per_prompt: int = 1


@dataclass
class WanTrainingConfig:
    # basic — model resolved as MODEL_ROOT + model_name; run dir under output_root
    task: str = "t2v-1.3B"
    model_name: str = "wan2.1-t2v-1.3B"
    output_root: str = "outputs/train"
    mixed_precision: str = "bf16"
    seed: int = 42

    dataset: str = "coco-5000"
    prompts_file: str = "data/train/video_prompt_200.txt"
    train_batch_size: int = 1
    dataloader_num_workers: int = 4
    t5_cpu: bool = True

    loss: str = "mse"
    gradient_accumulation_steps: int = 1
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    learning_rate: float = 5e-4
    lr_scheduler_type: str = "cosine"
    lr_warmup_ratio: float = 0.05
    lr_num_cycles: int = 1
    lr_power: float = 1.0

    max_train_steps: Optional[int] = None
    num_train_epochs: int = 1
    ckpt_save_steps: int = 64
    visual_steps: int = 0

    size: str = "832*480"
    frame_num: int = 81
    shift: float = 5.0
    guidance_scale: float = 5.0
    teacher_steps: int = 50
    student_steps: int = 20
    cache_step_path: Optional[str] = None

    sigmas_init_mode: str = "wan"
    sigmas_resume_path: Optional[str] = None
    desc: str = ""
