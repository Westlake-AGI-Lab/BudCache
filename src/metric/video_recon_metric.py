"""Video reconstruction metrics comparing test strategies against GT (full)."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import imageio
import lpips
import numpy as np
import torch
from tqdm import tqdm

from src.metric._metric_io import (
    print_metric_table,
    save_per_metric,
    update_top_summary,
    write_top_meta,
)
from src.metric.plot_metrics import plot_all


def list_mp4_files(directory: str) -> list[str]:
    return sorted(
        file_name
        for file_name in os.listdir(directory)
        if file_name.endswith(".mp4")
        and os.path.isfile(os.path.join(directory, file_name))
    )


def detect_layout(gen_dir: str) -> str:
    return "prefix" if list_mp4_files(gen_dir) else "folder"


def load_manifest_strategies(gen_dir: str) -> list[str] | None:
    manifest_path = os.path.join(gen_dir, "00_prompt_manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    strategies = manifest.get("strategies")
    if not isinstance(strategies, list):
        return None
    names = []
    for item in strategies:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
    return names or None


def collect_strategy_files(gen_dir: str, layout: str) -> dict[str, dict[str, str]]:
    strategy_to_files: dict[str, dict[str, str]] = {}
    if layout == "prefix":
        for file_name in list_mp4_files(gen_dir):
            if "_" not in file_name:
                continue
            strategy, sample_id = file_name.split("_", 1)
            strategy_to_files.setdefault(strategy, {})[sample_id] = os.path.join(gen_dir, file_name)
    else:
        for strategy in sorted(
            d for d in os.listdir(gen_dir)
            if os.path.isdir(os.path.join(gen_dir, d)) and not d.startswith("00") and d != "0_metric"
        ):
            file_map = {
                file_name: os.path.join(gen_dir, strategy, file_name)
                for file_name in list_mp4_files(os.path.join(gen_dir, strategy))
            }
            if file_map:
                strategy_to_files[strategy] = file_map
    return strategy_to_files


def resolve_gt_strategy(strategy_to_files: dict, manifest_strategies: list[str] | None) -> str:
    candidates = list(strategy_to_files.keys())
    if manifest_strategies:
        for strategy in manifest_strategies:
            if strategy in strategy_to_files and strategy.lower() == "full":
                return strategy
    for strategy in candidates:
        if strategy.lower() == "full":
            return strategy
    raise ValueError(f"Unable to find GT strategy `full`. Available: {candidates}")


def order_test_strategies(strategy_to_files: dict, gt_strategy: str, manifest_strategies: list[str] | None) -> list[str]:
    if manifest_strategies:
        ordered = [s for s in manifest_strategies if s in strategy_to_files and s != gt_strategy]
        if ordered:
            return ordered
    return sorted([s for s in strategy_to_files if s != gt_strategy])


def load_video(video_path: str) -> torch.Tensor:
    """Load video as tensor [T, C, H, W], normalized to [0, 1]."""
    reader = imageio.get_reader(video_path, "ffmpeg")
    frames = []
    for frame in reader:
        frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
        frames.append(frame_tensor)
    reader.close()
    return torch.stack(frames)


def calculate_psnr(video1: torch.Tensor, video2: torch.Tensor) -> float:
    """Frame-wise PSNR averaged over frames."""
    psnr_vals = []
    for t in range(video1.shape[0]):
        img1 = video1[t].cpu().numpy()
        img2 = video2[t].cpu().numpy()
        mse = np.mean((img1 - img2) ** 2)
        if mse < 1e-10:
            psnr_vals.append(100.0)
        else:
            psnr_vals.append(20.0 * math.log10(1.0 / math.sqrt(mse)))
    return float(np.mean(psnr_vals))


def calculate_ssim(video1: torch.Tensor, video2: torch.Tensor, window_size: int = 11) -> float:
    """Frame- and channel-wise SSIM, matching cv2.filter2D valid-mode reference."""
    C1, C2 = (0.01 ** 2), (0.03 ** 2)

    def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return (g.view(-1, 1) * g.view(1, -1)).view(1, 1, size, size)

    window = gaussian_window(window_size).to(video1.device)
    crop = window_size // 2

    ssim_vals = []
    for t in range(video1.shape[0]):
        for c in range(video1.shape[1]):
            img1 = video1[t, c].unsqueeze(0).unsqueeze(0)
            img2 = video2[t, c].unsqueeze(0).unsqueeze(0)
            mu1 = torch.nn.functional.conv2d(img1, window, padding=crop)[:, :, crop:-crop, crop:-crop]
            mu2 = torch.nn.functional.conv2d(img2, window, padding=crop)[:, :, crop:-crop, crop:-crop]
            mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
            sigma1_sq = torch.nn.functional.conv2d(img1**2, window, padding=crop)[:, :, crop:-crop, crop:-crop] - mu1_sq
            sigma2_sq = torch.nn.functional.conv2d(img2**2, window, padding=crop)[:, :, crop:-crop, crop:-crop] - mu2_sq
            sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=crop)[:, :, crop:-crop, crop:-crop] - mu1_mu2
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            ssim_vals.append(ssim_map.mean().item())
    return float(np.mean(ssim_vals))


@torch.no_grad()
def calculate_lpips(video1: torch.Tensor, video2: torch.Tensor, model) -> float:
    frame_vals = []
    for t in range(video1.shape[0]):
        gt_frame = video1[t:t + 1] * 2 - 1
        test_frame = video2[t:t + 1] * 2 - 1
        frame_vals.append(model.forward(gt_frame, test_frame).mean().detach().cpu().item())
    return float(np.mean(frame_vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gen_dir", help="Directory with strategy videos")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim", "lpips"],
        choices=["psnr", "ssim", "lpips"],
    )
    args = parser.parse_args()

    gen_dir = args.gen_dir
    output_root = Path(gen_dir)
    metric_dir = output_root / "0_metric"
    metric_dir.mkdir(parents=True, exist_ok=True)

    layout = detect_layout(gen_dir)
    manifest_strategies = load_manifest_strategies(gen_dir)
    strategy_to_files = collect_strategy_files(gen_dir, layout)
    gt_strategy = resolve_gt_strategy(strategy_to_files, manifest_strategies)
    test_strategies = order_test_strategies(strategy_to_files, gt_strategy, manifest_strategies)
    gt_files = strategy_to_files[gt_strategy]

    print(f"Layout: {layout}")
    print(f"GT: {gt_strategy} ({len(gt_files)} videos)")
    print(f"Test strategies: {test_strategies}")

    write_top_meta(output_root, {
        "video_layout": layout,
        "video_gt_strategy": gt_strategy,
        "video_gt_count": len(gt_files),
        "video_test_strategies": test_strategies,
    })

    lpips_model = lpips.LPIPS(net="alex", spatial=True).to("cuda").eval()

    psnr_records, ssim_records, lpips_records = [], [], []
    psnr_by, ssim_by, lpips_by = {}, {}, {}

    for strategy in test_strategies:
        test_files = strategy_to_files[strategy]
        shared_ids = sorted(sid for sid in test_files if sid in gt_files)

        psnr_scores, ssim_scores, lpips_scores = [], [], []
        for sample_id in tqdm(shared_ids, desc=strategy):
            gt_video = load_video(gt_files[sample_id]).to("cuda")
            test_video = load_video(test_files[sample_id]).to("cuda")

            if "psnr" in args.metrics:
                v = calculate_psnr(gt_video, test_video)
                psnr_scores.append(v)
                psnr_records.append({
                    "strategy": strategy, "sample_id": sample_id,
                    "video": test_files[sample_id], "gt": gt_files[sample_id], "score": v,
                })
            if "ssim" in args.metrics:
                v = calculate_ssim(gt_video, test_video)
                ssim_scores.append(v)
                ssim_records.append({
                    "strategy": strategy, "sample_id": sample_id,
                    "video": test_files[sample_id], "gt": gt_files[sample_id], "score": v,
                })
            if "lpips" in args.metrics:
                v = calculate_lpips(gt_video, test_video, lpips_model)
                lpips_scores.append(v)
                lpips_records.append({
                    "strategy": strategy, "sample_id": sample_id,
                    "video": test_files[sample_id], "gt": gt_files[sample_id], "score": v,
                })

        if "psnr" in args.metrics:
            psnr_by[strategy] = psnr_scores
        if "ssim" in args.metrics:
            ssim_by[strategy] = ssim_scores
        if "lpips" in args.metrics:
            lpips_by[strategy] = lpips_scores

    if "psnr" in args.metrics:
        print_metric_table("PSNR", psnr_by)
        save_per_metric(metric_dir, "psnr.json", "psnr", "pixel-domain", psnr_records, psnr_by)
        update_top_summary(output_root, "psnr", psnr_by)
    if "ssim" in args.metrics:
        print_metric_table("SSIM", ssim_by)
        save_per_metric(metric_dir, "ssim.json", "ssim", "pixel-domain", ssim_records, ssim_by)
        update_top_summary(output_root, "ssim", ssim_by)
    if "lpips" in args.metrics:
        print_metric_table("LPIPS", lpips_by)
        save_per_metric(metric_dir, "lpips.json", "lpips", "lpips-alex-spatial", lpips_records, lpips_by)
        update_top_summary(output_root, "lpips", lpips_by)

    plot_all(output_root)


if __name__ == "__main__":
    main()
