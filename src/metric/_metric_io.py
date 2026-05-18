"""Shared IO + summary helpers for the metric scripts.

Layout written into output_dir:
    0_metric.json            # meta + summary table for all metrics (append-only, never overwritten)
    0_metric/<name>.json     # per-metric file with full per-sample records + its own summary
    0_metric/plots/          # auto-generated figures from plot_metrics.plot_all
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SKIP_DIRS = {"grid", "0_grid", "0_metric"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def mean_std(scores: list[float]) -> tuple[float, float]:
    xs = np.asarray(scores, dtype=np.float64)
    if xs.size == 0:
        return float("nan"), float("nan")
    if xs.size == 1:
        return float(xs[0]), 0.0
    return float(xs.mean()), float(xs.std(ddof=1))


def summarize(by_strategy: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    out = {}
    for strategy, scores in by_strategy.items():
        mean, std = mean_std(scores)
        out[strategy] = {"n": len(scores), "mean": mean, "std": std}
    return out


def print_metric_table(metric_name: str, strategy_to_scores: dict[str, list[float]]) -> None:
    print(f"\n{metric_name}")
    print(f"{'Strategy':<24} {'Count':>6}  {'Mean':>10}  {'Std':>10}")
    print("-" * 56)
    for strategy, scores in strategy_to_scores.items():
        mean, std = mean_std(scores)
        print(f"{strategy:<24} {len(scores):>6}  {mean:>10.4f}  {std:>10.4f}")


def _load_top_summary(output_dir: Path) -> dict:
    path = output_dir / "0_metric.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"meta": {}, "summary": {}}


def _save_top_summary(output_dir: Path, payload: dict) -> None:
    with open(output_dir / "0_metric.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_top_meta(output_dir: Path, meta: dict) -> None:
    payload = _load_top_summary(output_dir)
    payload.setdefault("meta", {}).update(meta)
    _save_top_summary(output_dir, payload)


def update_top_summary(
    output_dir: Path,
    metric_name: str,
    by_strategy: dict[str, list[float]],
) -> None:
    payload = _load_top_summary(output_dir)
    payload.setdefault("summary", {})[metric_name] = summarize(by_strategy)
    _save_top_summary(output_dir, payload)


def save_per_metric(
    metric_dir: Path,
    filename: str,
    metric_name: str,
    model_id: str,
    records: list[dict],
    by_strategy: dict[str, list[float]],
) -> Path:
    payload = {
        "metric": metric_name,
        "model": model_id,
        "summary": summarize(by_strategy),
        "records": records,
    }
    save_path = metric_dir / filename
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  ↳ saved {save_path}")
    return save_path


def discover_strategy_dirs(output_root: Path) -> dict[str, Path]:
    """Find subdirs containing images, skipping known non-strategy folders."""
    samples_dir = output_root / "samples"
    if samples_dir.is_dir():
        return {"samples": samples_dir}
    strategy_dirs: dict[str, Path] = {}
    for path in sorted(output_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name in SKIP_DIRS or path.name.startswith("00") or path.name.startswith("."):
            continue
        if any(is_image_file(child) for child in path.iterdir()):
            strategy_dirs[path.name] = path
    return strategy_dirs
