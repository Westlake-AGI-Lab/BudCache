import csv
import glob
import os
import re
from pathlib import Path
from typing import Sequence

import imageio
import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.figsize": (3.4, 2.4),
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "mathtext.fontset": "stix",
    }
)

COMPUTE_COLOR = "#1E88E5"
CACHE_COLOR = "#D81B60"
BASE_COLOR = "#9AA0A6"
INIT_COLOR = "#6D4C41"

def _to_1d_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)

def _sanitize_indices(indices: Sequence[int] | None, upper: int) -> np.ndarray:
    if not indices:
        return np.asarray([], dtype=int)
    filtered = sorted({int(idx) for idx in indices if 0 <= int(idx) < int(upper)})
    return np.asarray(filtered, dtype=int)

def _atomic_write_csv(csv_path: str, rows: Sequence[Sequence]) -> None:
    """Write CSV via tmp + rename so a crash mid-write can't corrupt the target."""
    tmp_path = f"{csv_path}.tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)


def plot_loss_curve(out_dir: str, loss_values: Sequence[float], beta: float = 0.98) -> None:
    if not loss_values:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    rows = [["step", "loss"]] + [[idx, float(value)] for idx, value in enumerate(loss_values, 1)]
    _atomic_write_csv(os.path.join(out_dir, "train_loss.csv"), rows)

    loss = _to_1d_float_array(loss_values)
    x = np.arange(1, len(loss) + 1)
    ema = np.zeros_like(loss)
    value = 0.0
    for idx, loss_value in enumerate(loss):
        value = beta * value + (1 - beta) * loss_value
        ema[idx] = value / (1 - beta ** (idx + 1))

    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(x, loss, color=COMPUTE_COLOR, linewidth=0.3, alpha=0.18, label="Raw")
    plt.plot(x, ema, color="#0D47A1", linewidth=1.2, alpha=0.95, label="EMA")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend(loc="upper right", frameon=False)
    plt.savefig(os.path.join(out_dir, "train_loss.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_grad_norm_curve(out_dir: str, grad_norm_values: Sequence[float]) -> None:
    if not grad_norm_values:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(grad_norm_values, label="Grad Norm", color=CACHE_COLOR, linewidth=1.0, alpha=0.9)
    plt.xlabel("Step")
    plt.ylabel("Norm")
    plt.grid(True)
    plt.legend(frameon=False)
    plt.savefig(os.path.join(out_dir, "grad_norm.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_sigma_evolution(
    learned_sigmas,
    out_png: str,
    init_sigmas=None,
    base_sigmas=None,
    cache_steps: Sequence[int] | None = None,
    title: str | None = None,
) -> None:
    learned = _to_1d_float_array(learned_sigmas)
    x = np.arange(len(learned))
    cache_indices = _sanitize_indices(cache_steps, len(learned))
    cache_index_set = set(cache_indices.tolist())
    compute_indices = np.asarray([idx for idx in range(len(learned)) if idx not in cache_index_set], dtype=int)

    plt.figure(figsize=(5.6, 3.6), dpi=300)

    if base_sigmas is not None:
        base = _to_1d_float_array(base_sigmas)
        length = min(len(base), len(learned))
        plt.plot(
            np.arange(length),
            base[:length],
            linestyle="--",
            linewidth=1.2,
            color=BASE_COLOR,
            alpha=0.85,
            label="Base sigmas",
            zorder=1,
        )
    if init_sigmas is not None:
        init = _to_1d_float_array(init_sigmas)
        length = min(len(init), len(learned))
        should_show = base_sigmas is None or not np.allclose(init[:length], _to_1d_float_array(base_sigmas)[:length])
        if should_show:
            plt.plot(
                np.arange(length),
                init[:length],
                linestyle=":",
                linewidth=1.4,
                color=INIT_COLOR,
                alpha=0.9,
                label="Init sigmas",
                zorder=2,
            )

    plt.plot(x, learned, color=COMPUTE_COLOR, linewidth=1.3, alpha=0.75, zorder=3)
    if len(compute_indices) > 0:
        plt.scatter(
            x[compute_indices],
            learned[compute_indices],
            color=COMPUTE_COLOR,
            edgecolors="white",
            linewidths=0.6,
            s=24,
            alpha=0.95,
            label="Compute step",
            zorder=4,
        )
    if len(cache_indices) > 0:
        plt.scatter(
            x[cache_indices],
            learned[cache_indices],
            color=CACHE_COLOR,
            edgecolors="white",
            linewidths=0.6,
            s=24,
            alpha=0.3,
            label="Cache step",
            zorder=5,
        )

    plt.xlabel("Student step index")
    plt.ylabel(r"$\sigma$")
    plt.xlim(-0.5, max(len(learned) - 0.5, 0.5))
    plt.grid(True, linestyle=":", alpha=0.35)
    if title:
        plt.title(title)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False, loc="best")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_sigma_diff_bar(learned_sigmas, init_sigmas, out_png: str, cache_steps: Sequence[int] | None = None) -> None:
    learned = _to_1d_float_array(learned_sigmas)
    init = _to_1d_float_array(init_sigmas)
    length = min(len(learned), len(init))
    diff = learned[:length] - init[:length]
    steps = np.arange(length)
    cache_indices = set(_sanitize_indices(cache_steps, length).tolist())
    colors = [
        mcolors.to_rgba(CACHE_COLOR, alpha=0.35) if idx in cache_indices else mcolors.to_rgba(COMPUTE_COLOR, alpha=0.95)
        for idx in steps
    ]

    plt.figure(figsize=(5.6, 3.4), dpi=300)
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.75)
    plt.bar(steps, diff, color=colors, width=0.65, zorder=3)
    plt.xlabel("Student step index")
    plt.ylabel(r"$\Delta \sigma$")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(
        handles=[
            mpatches.Patch(color=mcolors.to_rgba(COMPUTE_COLOR, alpha=0.95), label="Compute step"),
            mpatches.Patch(color=mcolors.to_rgba(CACHE_COLOR, alpha=0.35), label="Cache step"),
        ],
        frameon=False,
        loc="best",
        fontsize=8,
    )
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def make_png2gif(frames_dir: str, out_gif: str, pattern: str = "step*.png", fps: int = 8) -> str | None:
    """Stitch numbered step PNGs into a GIF via a streaming writer — never holds
    more than one decoded frame in memory, so long training runs are safe."""
    step_re = re.compile(r"step(\d+)")

    def step_index(path: str) -> int:
        match = step_re.search(os.path.basename(path))
        return int(match.group(1)) if match else 0

    paths = sorted(glob.glob(os.path.join(str(frames_dir), pattern)), key=step_index)
    if not paths:
        return None
    Path(out_gif).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_gif, mode="I", duration=1.0 / fps) as writer:
        for path in paths:
            writer.append_data(imageio.v2.imread(path))
    return out_gif

def _prepare_cells(images: Sequence[Image.Image], cell_size: tuple[int, int]) -> list[Image.Image]:
    cell_w, cell_h = cell_size
    cells = []
    for image in images:
        rgb = image.convert("RGB")
        cell = Image.new("RGB", (cell_w, cell_h), color=(255, 255, 255))
        offset_x = (cell_w - rgb.width) // 2
        offset_y = (cell_h - rgb.height) // 2
        cell.paste(rgb, (offset_x, offset_y))
        cells.append(cell)
    return cells

def build_teacher_student_panel(
    teacher_images: Sequence[Image.Image],
    student_images: Sequence[Image.Image],
    step_label: str | None = None,
) -> Image.Image | None:
    if not teacher_images or not student_images:
        return None

    teacher_images = list(teacher_images)
    student_images = list(student_images)
    cell_w = max(image.width for image in teacher_images + student_images)
    cell_h = max(image.height for image in teacher_images + student_images)
    label_w = 72
    pad = 14
    col_gap = 10
    row_gap = 12
    header_h = 24 if step_label else 0
    num_cols = max(len(teacher_images), len(student_images))
    canvas_w = pad + label_w + num_cols * cell_w + max(num_cols - 1, 0) * col_gap + pad
    canvas_h = pad + header_h + 2 * cell_h + row_gap + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    y_offset = pad
    if step_label:
        draw.text((pad, y_offset), step_label, fill=(40, 40, 40), font=font)
        y_offset += header_h

    rows = [
        ("Teacher", _prepare_cells(teacher_images, (cell_w, cell_h))),
        ("Student", _prepare_cells(student_images, (cell_w, cell_h))),
    ]
    for row_idx, (label, row_images) in enumerate(rows):
        row_top = y_offset + row_idx * (cell_h + row_gap)
        draw.text((pad, row_top + 4), label, fill=(40, 40, 40), font=font)
        x_offset = pad + label_w
        for image_idx, image in enumerate(row_images):
            left = x_offset + image_idx * (cell_w + col_gap)
            canvas.paste(image, (left, row_top))

    return canvas

def append_image_panel(out_png: str, panel: Image.Image, gap: int = 12) -> None:
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(out_png):
        panel.save(out_png)
        return

    existing = Image.open(out_png).convert("RGB")
    width = max(existing.width, panel.width)
    height = existing.height + gap + panel.height
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    existing_x = (width - existing.width) // 2
    panel_x = (width - panel.width) // 2
    canvas.paste(existing, (existing_x, 0))
    canvas.paste(panel, (panel_x, existing.height + gap))

    draw = ImageDraw.Draw(canvas)
    divider_y = existing.height + gap // 2
    draw.line((16, divider_y, width - 16, divider_y), fill=(220, 220, 220), width=1)
    canvas.save(out_png)