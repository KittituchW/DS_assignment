"""Shared chart style.

Categorical colours come in a fixed order and follow the entity, never its rank.
The palette passes lightness, chroma, colour-blind separation and normal-vision
checks on adjacent pairs. Three slots sit below 3:1 contrast on the surface, so
bars carry direct value labels.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

TYPE_COLOURS = {"movie": SERIES[0], "tv": SERIES[1]}
MODEL_COLOURS = {
    "two_tower": SERIES[0],
    "popularity": SERIES[1],
    "most_watched": SERIES[2],
    "content_knn": SERIES[3],
    "popularity_by_type": SERIES[4],
}
BAR = {"edgecolor": SURFACE, "linewidth": 2}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "text.color": INK,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def label_bars(ax, bars, fmt: str = "{:.0f}", offset: float = 3) -> None:
    """Direct value labels in text ink, above positive bars and below negative ones."""
    for bar in bars:
        value = bar.get_height()
        y = bar.get_y() + value
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, y),
            xytext=(0, offset if value >= 0 else -offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
            color=INK_SECONDARY,
        )


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
