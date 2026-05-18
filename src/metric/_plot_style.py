"""Shared plotting style for metric visualizations."""
from __future__ import annotations

import colorsys

import matplotlib as mpl

# Google brand 4-color + Material-Design extensions (same brightness level)
# First 4 are Google's primary colors; next 6 are Material 400-level hues that
# match in saturation/brightness so they look like one family.
GOOGLE = [
    "#4285F4",  # blue       (Google)
    "#EA4335",  # red        (Google)
    "#FBBC04",  # yellow     (Google)
    "#34A853",  # green      (Google)
    "#AB47BC",  # purple     (Material 400)
    "#00ACC1",  # cyan       (Material 600)
    "#FF7043",  # deep-orange(Material 400)
    "#EC407A",  # pink       (Material 400)
    "#5C6BC0",  # indigo     (Material 400)
    "#8D6E63",  # brown      (Material 400)
]
# Matplotlib's classic 10-class categorical (Tableau-derived) — backup when >10 strategies
TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
# ColorBrewer Set2 — soft, colorblind-aware
SET2 = [
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
]
# ColorBrewer Dark2 — saturated alternative
DARK2 = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
    "#66a61e", "#e6ab02", "#a6761d", "#666666",
]

PALETTES = {"google": GOOGLE, "tab10": TAB10, "set2": SET2, "dark2": DARK2}

BASELINE_COLOR = "#666666"
BASELINE_NAMES = {"full", "gt", "baseline", "reference"}


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _hsl_supplement(n_extra: int) -> list[str]:
    """Evenly-spaced HSL hues at consistent lightness/saturation. Used when a
    chosen palette doesn't have enough colors for all non-baseline strategies."""
    hues = [(i + 0.5) / (n_extra + 1) for i in range(n_extra)]
    out = []
    for h in hues:
        r, g, b = colorsys.hls_to_rgb(h, 0.55, 0.65)
        out.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return out


def get_palette(strategies: list[str], scheme: str = "google") -> dict[str, str]:
    palette = list(PALETTES.get(scheme, GOOGLE))
    non_baseline = [s for s in strategies if s.lower() not in BASELINE_NAMES]
    if len(non_baseline) > len(palette):
        palette = palette + _hsl_supplement(len(non_baseline) - len(palette))
    result: dict[str, str] = {}
    color_idx = 0
    for strategy in strategies:
        if strategy.lower() in BASELINE_NAMES:
            result[strategy] = BASELINE_COLOR
        else:
            result[strategy] = palette[color_idx]
            color_idx += 1
    return result


def darken(hex_color: str, factor: float = 0.8) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"
