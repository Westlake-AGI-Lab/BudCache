from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.metric._metric_io import (
    discover_strategy_dirs,
    is_image_file,
    print_metric_table,
    save_per_metric,
    update_top_summary,
    write_top_meta,
)
from src.metric.plot_metrics import plot_all


def _imread01(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def psnr(gen_path: str, ref_path: str) -> float:
    gen = _imread01(gen_path)
    ref = _imread01(ref_path)
    mse = float(np.mean((gen - ref) ** 2))
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim(gen_path: str, ref_path: str) -> float:
    gen = _imread01(gen_path)
    ref = _imread01(ref_path)
    c1, c2 = 0.01**2, 0.03**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    def _channel_ssim(x: np.ndarray, y: np.ndarray) -> float:
        mu_x = cv2.filter2D(x, -1, window)[5:-5, 5:-5]
        mu_y = cv2.filter2D(y, -1, window)[5:-5, 5:-5]
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x_sq = cv2.filter2D(x * x, -1, window)[5:-5, 5:-5] - mu_x_sq
        sigma_y_sq = cv2.filter2D(y * y, -1, window)[5:-5, 5:-5] - mu_y_sq
        sigma_xy = cv2.filter2D(x * y, -1, window)[5:-5, 5:-5] - mu_xy
        return float(
            (
                (2 * mu_xy + c1)
                * (2 * sigma_xy + c2)
                / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
            ).mean()
        )

    return float(
        np.mean(
            [
                _channel_ssim(gen[..., channel].astype(np.float64), ref[..., channel].astype(np.float64))
                for channel in range(3)
            ]
        )
    )


@torch.no_grad()
def lpips_score(gen_path: str, ref_path: str, model) -> float:
    gen = _imread01(gen_path)
    ref = _imread01(ref_path)
    gen_tensor = torch.from_numpy(gen).permute(2, 0, 1).unsqueeze(0).to("cuda") * 2 - 1
    ref_tensor = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).to("cuda") * 2 - 1
    return float(model(gen_tensor, ref_tensor).item())


def resolve_gt_and_strategies(
    output_root: Path, gt_dir_arg: str | None,
) -> tuple[dict[str, Path], Path]:
    samples_dir = output_root / "samples"
    if samples_dir.is_dir():
        if gt_dir_arg is None:
            raise ValueError("samples mode requires --gt_dir")
        return {"samples": samples_dir}, Path(gt_dir_arg)
    strategy_dirs = discover_strategy_dirs(output_root)
    if "full" not in strategy_dirs:
        raise ValueError(f"Evaluation mode requires full/ as GT. Available: {list(strategy_dirs.keys())}")
    gt_dir = strategy_dirs.pop("full")
    return strategy_dirs, gt_dir


def build_index_against_gt(
    strategy_dirs: dict[str, Path], gt_dir: Path,
) -> tuple[dict[str, dict[str, Path]], dict[str, Path], list[int], list[int]]:
    gt_images = sorted(p for p in gt_dir.iterdir() if is_image_file(p))
    if not gt_images:
        raise ValueError(f"No images found in {gt_dir}")
    gt_map = {p.stem: p for p in gt_images}

    reference_prefix = "p00000_s"
    sample_indices = sorted(
        int(p.stem[len(reference_prefix):])
        for p in gt_images
        if p.stem.startswith(reference_prefix)
    )
    if not sample_indices:
        raise ValueError(f"{gt_dir} does not contain files like p00000_s00.png")
    prompt_indices = sorted(
        {int(p.stem[1:6]) for p in gt_images if p.stem.startswith("p") and "_s" in p.stem}
    )

    strategy_to_image_map: dict[str, dict[str, Path]] = {}
    for strategy, strategy_dir in strategy_dirs.items():
        images = sorted(p for p in strategy_dir.iterdir() if is_image_file(p))
        if not images:
            raise ValueError(f"No images found in {strategy_dir}")
        strategy_to_image_map[strategy] = {p.stem: p for p in images}
    return strategy_to_image_map, gt_map, sample_indices, prompt_indices


def _sweep(
    metric_label: str,
    score_fn,
    strategy_to_image_map: dict[str, dict[str, Path]],
    gt_map: dict[str, Path],
    prompt_indices: list[int],
    sample_indices: list[int],
) -> tuple[dict[str, list[float]], list[dict]]:
    records: list[dict] = []
    by_strategy: dict[str, list[float]] = {}
    for strategy, image_map in strategy_to_image_map.items():
        scores: list[float] = []
        for prompt_idx in tqdm(prompt_indices, desc=f"{metric_label}:{strategy}"):
            for sample_idx in sample_indices:
                sample_id = f"p{prompt_idx:05d}_s{sample_idx:02d}"
                if sample_id not in image_map:
                    raise ValueError(f"{strategy} missing {sample_id}")
                if sample_id not in gt_map:
                    raise ValueError(f"gt missing {sample_id}")
                score = float(score_fn(str(image_map[sample_id]), str(gt_map[sample_id])))
                scores.append(score)
                records.append({
                    "strategy": strategy,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "image": str(image_map[sample_id]),
                    "gt": str(gt_map[sample_id]),
                    "score": score,
                })
        by_strategy[strategy] = scores
    return by_strategy, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--gt_dir", help="External GT directory; required only in samples-mode")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim", "lpips"],
        choices=["psnr", "ssim", "lpips"],
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    metric_dir = output_root / "0_metric"
    metric_dir.mkdir(parents=True, exist_ok=True)

    lpips_model = lpips.LPIPS(net="alex", spatial=False).to("cuda").eval()

    strategy_dirs, gt_dir = resolve_gt_and_strategies(output_root, args.gt_dir)
    if not strategy_dirs:
        raise ValueError("No test strategies found")
    strategy_to_image_map, gt_map, sample_indices, prompt_indices = build_index_against_gt(
        strategy_dirs, gt_dir,
    )

    print(f"GT: {gt_dir} ({len(gt_map)} images)")
    print(f"Test strategies: {list(strategy_to_image_map.keys())}")
    print(f"Detected samples_per_prompt: {len(sample_indices)}")
    for strategy, image_map in strategy_to_image_map.items():
        print(f"{strategy}: {len(image_map)} images")

    write_top_meta(output_root, {
        "recon_gt_dir": str(gt_dir),
        "recon_strategies": list(strategy_to_image_map.keys()),
        "num_prompts": len(prompt_indices),
        "samples_per_prompt": len(sample_indices),
    })

    if "psnr" in args.metrics:
        by_strategy, records = _sweep("PSNR", psnr, strategy_to_image_map, gt_map, prompt_indices, sample_indices)
        print_metric_table("PSNR", by_strategy)
        save_per_metric(metric_dir, "psnr.json", "psnr", "pixel-domain", records, by_strategy)
        update_top_summary(output_root, "psnr", by_strategy)

    if "ssim" in args.metrics:
        by_strategy, records = _sweep("SSIM", ssim, strategy_to_image_map, gt_map, prompt_indices, sample_indices)
        print_metric_table("SSIM", by_strategy)
        save_per_metric(metric_dir, "ssim.json", "ssim", "pixel-domain", records, by_strategy)
        update_top_summary(output_root, "ssim", by_strategy)

    if "lpips" in args.metrics:
        score_fn = lambda gen, ref: lpips_score(gen, ref, lpips_model)
        by_strategy, records = _sweep("LPIPS", score_fn, strategy_to_image_map, gt_map, prompt_indices, sample_indices)
        print_metric_table("LPIPS", by_strategy)
        save_per_metric(metric_dir, "lpips.json", "lpips", "lpips-alex", records, by_strategy)
        update_top_summary(output_root, "lpips", by_strategy)

    plot_all(output_root)


if __name__ == "__main__":
    main()
