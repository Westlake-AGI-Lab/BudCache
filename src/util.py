import json
import logging
import os
import re
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image

######## input io ########
def read_prompts(path: str) -> List[str]:
    if path.endswith(".jsonl"):
        out: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                out.append(json.loads(ln)["prompt"])
        return out
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    if path.endswith(".csv"):
        df = pd.read_csv(path)  # default "caption"
        return df["caption"].astype(str).tolist()
    raise ValueError("prompts_path must be .jsonl or .txt")


def model_root() -> str:
    return os.environ.get("MODEL_ROOT") or os.environ.get("MODEL_DIR") or ""


def model_path(model_name: str) -> str:
    root = model_root()
    return os.path.join(root, model_name) if root else model_name

######## output ########
_SCIENTIFIC_EXPONENT = re.compile(r"e([+-])0+(\d+)")
def _format_slugify(value) -> str:
    """Compact path-safe stringify for cfg field values."""
    if isinstance(value, float):
        s = f"{value:g}"
        return _SCIENTIFIC_EXPONENT.sub(r"e\1\2", s)  # 1e-05 -> 1e-5
    if isinstance(value, str):
        return value.lower().replace("/", "-").replace(" ", "-")
    return str(value)


def make_eval_exp_name_flux(cfg) -> str:
    """*evaluation* Deterministic experiment name for FLUX sampling."""
    res_str = str(cfg.width) if cfg.height == cfg.width else f"{cfg.width}x{cfg.height}"
    tokens = [
        _format_slugify(cfg.model_name),
        _format_slugify(cfg.dataset_name),
        f"steps{cfg.steps}",
        f"cfg{_format_slugify(cfg.guidance_scale)}",
        f"res{res_str}",
        f"seed{cfg.seed}",
    ]
    if getattr(cfg, "use_hypersd", False):
        tokens.append("hypersd")
    return "_".join(tokens)


def make_eval_exp_name_wan(cfg) -> str:
    """*evaluation* Deterministic experiment name for WAN sampling."""
    tokens = [
        _format_slugify(cfg.task),
        _format_slugify(cfg.dataset_name),
        f"steps{cfg.steps}",
        f"cfg{_format_slugify(cfg.guidance_scale)}",
        f"res{cfg.size.replace('*', 'x')}",
        f"frames{cfg.frame_num}",
        f"seed{cfg.seed}",
    ]
    return "_".join(tokens)


def make_train_exp_name_flux(cfg) -> str:
    """*training* Deterministic experiment name — only the axes you'd compare across runs.
    Infrastructure constants (res, bsz, teacher_steps) live in 00_run_config.yaml instead."""
    tokens = [
        "flux-stage2",
        _format_slugify(cfg.dataset),
        f"S{cfg.student_steps}",
        f"nfe{cfg.student_nfe}",
        _format_slugify(cfg.loss),
        _format_slugify(cfg.sigmas_learn_mode),
        f"lr{_format_slugify(cfg.learning_rate)}",
    ]
    return "_".join(tokens)


def make_search_exp_name_flux(cfg, prompt_hash: str) -> str:
    """*search* Deterministic experiment name for FLUX cache-step search."""
    tokens = [
        "flux-search",
        f"h{prompt_hash}",
        f"steps{cfg.steps}",
        f"nfe{cfg.nfe}",
        f"rs{cfg.stage0_multiStart}",
        f"sa{cfg.stage1_sa_iters}",
        f"hc{cfg.stage2_hc_iters}",
    ]
    return "_".join(tokens)


def make_search_exp_name_wan(cfg, prompt_hash: str) -> str:
    """*search* Deterministic experiment name for WAN cache-step search."""
    tokens = [
        "wan-search",
        _format_slugify(cfg.task),
        f"h{prompt_hash}",
        f"steps{cfg.steps}",
        f"nfe{cfg.nfe}",
        f"rs{cfg.stage0_multiStart}",
        f"sa{cfg.stage1_sa_iters}",
        f"hc{cfg.stage2_hc_iters}",
    ]
    return "_".join(tokens)


def make_train_exp_name_wan(cfg) -> str:
    """*training* Deterministic experiment name — only the axes you'd compare across runs.
    Infrastructure constants (res, frames, bsz, teacher_steps) live in 00_run_config.yaml instead."""
    tokens = [
        "wan-stage2",
        _format_slugify(cfg.task),
        _format_slugify(cfg.dataset),
        f"S{cfg.student_steps}",
        _format_slugify(cfg.loss),
        f"lr{_format_slugify(cfg.learning_rate)}",
    ]
    return "_".join(tokens)


######## log ########
def create_logger(logging_dir, rank=0, name="research", filename="00_run_info.log", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # Rebuild handlers on each call so the same logger name can safely target
    # a new run directory without leaking the previous file handler.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    if rank == 0:
        console_formatter = logging.Formatter(
            fmt="[\033[34m%(asctime)s\033[0m][%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_formatter = logging.Formatter(
            fmt="[%(asctime)s][%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        file_handler = logging.FileHandler(os.path.join(logging_dir, filename))
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        def _log_uncaught(exc_type, exc_value, exc_tb):
            logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        sys.excepthook = _log_uncaught
    else:
        logger.addHandler(logging.NullHandler())

    return logger


######## visual ########
def get_preview_indices(frame_num: int, num_previews: int = 5) -> List[int]:
    if frame_num <= 1:
        return [0]
    indices = np.linspace(0, frame_num - 1, num=min(num_previews, frame_num))
    return sorted({int(round(idx)) for idx in indices})

def extract_frames(video_tensor: torch.Tensor, indices: List[int]) -> List[Image.Image]:
    video = video_tensor.permute(1, 2, 3, 0).detach().cpu().numpy()
    video = ((video + 1) * 127.5).clip(0, 255).astype(np.uint8)
    frames: List[Image.Image] = []
    for idx in indices:
        if 0 <= idx < video.shape[0]:
            frames.append(Image.fromarray(video[idx]))
    return frames

def create_comparison_grid(
    frame_lists: List[List[Image.Image]],
    gap: int = 5,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image | None:
    if not frame_lists or not frame_lists[0]:
        return None
    width, height = frame_lists[0][0].size
    row_width = len(frame_lists[0]) * width + (len(frame_lists[0]) - 1) * gap
    total_height = len(frame_lists) * height + (len(frame_lists) - 1) * gap
    canvas = Image.new("RGB", (row_width, total_height), bg_color)
    for row_idx, frames in enumerate(frame_lists):
        y_offset = row_idx * (height + gap)
        for col_idx, frame in enumerate(frames):
            x_offset = col_idx * (width + gap)
            canvas.paste(frame, (x_offset, y_offset))
    return canvas
