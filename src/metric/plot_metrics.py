"""Plot summary visualizations from `0_metric.json`.

Usage:
  python -m src.metric.plot_metrics <output_dir_or_json_path>
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.metric._plot_style import (
    BASELINE_COLOR,
    apply_style,
    get_palette,
)


NUMERIC_SUFFIX_RE = re.compile(r"^([A-Za-z_\-]+?)(\d+(?:\.\d+)?)$")


def detect_numeric_sweep(strategies: list[str]) -> tuple[list[float], str] | None:
    if len(strategies) < 2:
        return None
    matches = [NUMERIC_SUFFIX_RE.match(s) for s in strategies]
    if not all(matches):
        return None
    prefixes = {m.group(1) for m in matches}
    if len(prefixes) > 1:
        return None
    return [float(m.group(2)) for m in matches], prefixes.pop()


def error_half_width(stats: dict[str, float], kind: str) -> float:
    n, std = stats["n"], stats["std"]
    if kind == "std":
        return std
    if kind == "sem":
        return std / math.sqrt(max(n, 1))
    if kind == "ci95":
        return 1.96 * std / math.sqrt(max(n, 1))
    return 0.0


def needs_rotation(labels: list[str]) -> bool:
    return any(len(s) > 8 for s in labels) or len(labels) > 5


def fmt_value(value: float) -> str:
    abs_v = abs(value)
    if abs_v >= 10:
        return f"{value:.2f}"
    if abs_v >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _dot_on_axis(
    ax,
    strategies: list[str],
    means: np.ndarray,
    errs: np.ndarray,
    colors: list[str],
    error_kind: str,
) -> None:
    """Render a Cleveland-style dot plot (mean ± CI) on an existing axis."""
    x_positions = np.arange(len(strategies))
    if error_kind != "none":
        for i, (m, e, c) in enumerate(zip(means, errs, colors)):
            ax.errorbar(
                i, m, yerr=e, fmt="none",
                ecolor=c, elinewidth=2.0, capsize=5, capthick=1.5, alpha=0.9,
            )
    ax.scatter(
        x_positions, means, s=140, c=colors,
        edgecolor="black", linewidth=0.6, zorder=3,
    )
    for i, m in enumerate(means):
        ax.text(
            i + 0.14, m, fmt_value(m),
            ha="left", va="center", fontsize=9, color="#222222",
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        strategies,
        rotation=20 if needs_rotation(strategies) else 0,
        ha="right" if needs_rotation(strategies) else "center",
    )
    ax.set_xlim(-0.5, len(strategies) - 0.5)

    y_top = float((means + errs).max())
    y_bot = float((means - errs).min())
    y_range = max(y_top - y_bot, 1e-9)
    y_pad = max(y_range * 0.25, 1e-3)
    ax.set_ylim(y_bot - y_pad, y_top + y_pad)


def plot_dot(
    metric_name: str,
    by_strategy: dict[str, dict[str, float]],
    palette: dict[str, str],
    save_dir: Path,
    error_kind: str,
) -> None:
    strategies = list(by_strategy.keys())
    means = np.array([by_strategy[s]["mean"] for s in strategies])
    errs = np.array([error_half_width(by_strategy[s], error_kind) for s in strategies])
    colors = [palette[s] for s in strategies]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    _dot_on_axis(ax, strategies, means, errs, colors, error_kind)
    ax.set_ylabel(metric_name)
    err_label = {"ci95": "95% CI", "sem": "SEM", "std": "std"}.get(error_kind)
    if err_label:
        ax.text(
            0.99, 0.97, f"error: {err_label}", ha="right", va="top",
            transform=ax.transAxes, fontsize=8, color="#666666",
        )
    fig.savefig(save_dir / f"{metric_name}.png")
    plt.close(fig)


def plot_line(
    metric_name: str,
    by_strategy: dict[str, dict[str, float]],
    palette: dict[str, str],
    save_dir: Path,
    x_values: list[float],
    x_label: str,
    error_kind: str,
) -> None:
    strategies = list(by_strategy.keys())
    paired = sorted(zip(x_values, strategies), key=lambda t: t[0])
    xs = np.array([p[0] for p in paired])
    ordered = [p[1] for p in paired]
    means = np.array([by_strategy[s]["mean"] for s in ordered])
    errs = np.array([error_half_width(by_strategy[s], error_kind) for s in ordered])

    line_color = next(
        (palette[s] for s in ordered if palette[s] != BASELINE_COLOR),
        "#1b9e77",
    )

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    if error_kind != "none":
        ax.fill_between(xs, means - errs, means + errs,
                        color=line_color, alpha=0.18, linewidth=0)
    ax.plot(
        xs, means, color=line_color, linewidth=2.0,
        marker="o", markersize=7,
        markerfacecolor="white", markeredgecolor=line_color, markeredgewidth=1.6,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_name)
    err_label = {"ci95": "95% CI", "sem": "SEM", "std": "std"}.get(error_kind)
    if err_label:
        ax.text(
            0.99, 0.97, f"error: {err_label}", ha="right", va="top",
            transform=ax.transAxes, fontsize=8, color="#666666",
        )
    fig.savefig(save_dir / f"line_{metric_name}.png")
    plt.close(fig)


def plot_panel(
    summary: dict[str, dict[str, dict[str, float]]],
    palette: dict[str, str],
    save_dir: Path,
    error_kind: str,
    sweep: tuple[list[float], str] | None,
) -> None:
    metrics = list(summary.keys())
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.5), squeeze=False)

    for ax, metric_name in zip(axes[0], metrics):
        by_strategy = summary[metric_name]
        strategies = list(by_strategy.keys())
        means = np.array([by_strategy[s]["mean"] for s in strategies])
        errs = np.array([error_half_width(by_strategy[s], error_kind) for s in strategies])

        if sweep is not None:
            x_values, x_label = sweep
            paired = sorted(zip(x_values, strategies), key=lambda t: t[0])
            xs = np.array([p[0] for p in paired])
            ordered = [p[1] for p in paired]
            means = np.array([by_strategy[s]["mean"] for s in ordered])
            errs = np.array([error_half_width(by_strategy[s], error_kind) for s in ordered])
            line_color = next(
                (palette[s] for s in ordered if palette[s] != BASELINE_COLOR),
                "#1b9e77",
            )
            if error_kind != "none":
                ax.fill_between(xs, means - errs, means + errs,
                                color=line_color, alpha=0.18, linewidth=0)
            ax.plot(
                xs, means, color=line_color, linewidth=2.0,
                marker="o", markersize=6,
                markerfacecolor="white", markeredgecolor=line_color, markeredgewidth=1.5,
            )
            ax.set_xlabel(x_label)
        else:
            colors = [palette[s] for s in strategies]
            _dot_on_axis(ax, strategies, means, errs, colors, error_kind)

        ax.set_ylabel(metric_name)

    err_label = {"ci95": "95% CI", "sem": "SEM", "std": "std"}.get(error_kind)
    if err_label:
        fig.text(0.99, 0.98, f"error: {err_label}", ha="right", va="top",
                 fontsize=9, color="#666666")
    fig.savefig(save_dir / "panel.png")
    plt.close(fig)


def plot_scatter_matrix(
    summary: dict[str, dict[str, dict[str, float]]],
    palette: dict[str, str],
    save_dir: Path,
    error_kind: str,
) -> None:
    metrics = list(summary.keys())
    if len(metrics) < 2:
        return
    n = len(metrics)
    strategies = list(next(iter(summary.values())).keys())

    fig, axes = plt.subplots(n, n, figsize=(2.8 * n, 2.8 * n), squeeze=False)
    for i, m_y in enumerate(metrics):
        for j, m_x in enumerate(metrics):
            ax = axes[i, j]
            if i == j:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.5, 0.5, m_x, ha="center", va="center",
                        fontsize=14, fontweight="bold", transform=ax.transAxes)
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.grid(False)
                continue
            for s in strategies:
                x = summary[m_x][s]["mean"]
                y = summary[m_y][s]["mean"]
                xerr = error_half_width(summary[m_x][s], error_kind)
                yerr = error_half_width(summary[m_y][s], error_kind)
                color = palette[s]
                if error_kind != "none":
                    ax.errorbar(
                        x, y, xerr=xerr, yerr=yerr, fmt="none",
                        ecolor=color, elinewidth=0.8, alpha=0.45, capsize=2,
                    )
                ax.scatter(
                    [x], [y], s=110, color=color, edgecolor="black",
                    linewidth=0.5, alpha=0.92, zorder=3,
                )
            if i == n - 1:
                ax.set_xlabel(m_x)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(m_y)
            else:
                ax.set_yticklabels([])

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=palette[s],
            markeredgecolor="black", markersize=10, label=s,
        )
        for s in strategies
    ]
    fig.legend(
        handles=handles, loc="upper center",
        ncol=min(len(strategies), 6), bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_dir / "scatter_matrix.png")
    plt.close(fig)


def plot_all(
    output_dir: Path | str,
    error_kind: str = "ci95",
    palette_scheme: str = "google",
) -> None:
    """Read `0_metric.json` under output_dir and produce all plots.

    Safe to call repeatedly — overwrites previous figures in `metric/plots/`.
    """
    apply_style()
    output_root = Path(output_dir)
    json_path = output_root / "0_metric.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Cannot find {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary", {})
    if not summary:
        raise ValueError(f"{json_path} has no `summary` block")

    plots_dir = output_root / "0_metric" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    strategies = list(next(iter(summary.values())).keys())
    palette = get_palette(strategies, palette_scheme)
    sweep = detect_numeric_sweep(strategies)

    print(f"Plotting {len(summary)} metric(s) × {len(strategies)} strategy(ies) "
          f"[{ 'line:' + sweep[1] if sweep else 'bar' }, error={error_kind}] → {plots_dir}")

    for metric_name, by_strategy in summary.items():
        if sweep is not None:
            plot_line(metric_name, by_strategy, palette, plots_dir,
                      sweep[0], sweep[1], error_kind)
        else:
            plot_dot(metric_name, by_strategy, palette, plots_dir, error_kind)
    plot_panel(summary, palette, plots_dir, error_kind, sweep)
    plot_scatter_matrix(summary, palette, plots_dir, error_kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to 0_metric.json or its parent directory")
    parser.add_argument(
        "--palette", default="google",
        choices=["google", "tab10", "set2", "dark2"],
    )
    parser.add_argument("--error_bar", default="ci95",
                        choices=["ci95", "sem", "std", "none"])
    args = parser.parse_args()

    target = Path(args.path)
    output_root = target.parent if target.is_file() else target
    plot_all(output_root, error_kind=args.error_bar, palette_scheme=args.palette)
    print("Done.")


if __name__ == "__main__":
    main()
