"""Shared design system for all manuscript figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed method order and Okabe-Ito colours (identity follows the method, never its rank).
METHODS = ("Default", "Random", "TPE", "GWO", "WOA", "PSO")
OPTIMIZERS = METHODS[1:]
COLORS = {"Default": "#000000", "Random": "#7F7F7F", "TPE": "#0072B2", "GWO": "#D55E00", "WOA": "#009E73",
          "PSO": "#CC79A7"}
# Secondary encoding so identity never rests on colour alone.
MARKERS = {"Random": "o", "TPE": "s", "GWO": "^", "WOA": "D", "PSO": "v"}
DEFAULT_LINE = dict(color="black", linestyle=(0, (4, 2)), linewidth=0.8)
GRID = dict(color="#E6E6E6", linewidth=0.4)
INK_MUTED = "#555555"

SINGLE_COL, DOUBLE_COL = 3.5, 7.2  # inches
DPI = 600
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def apply() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "lines.linewidth": 0.8, "axes.linewidth": 0.8, "patch.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.edgecolor": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
        "legend.frameon": False, "legend.handlelength": 1.5,
        "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": DPI,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",  # editable text in vector output
    })


def panel_label(ax, index: int, x: float = -0.14, y: float = 1.04) -> None:
    ax.text(x, y, PANEL_LETTERS[index], transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def watermark(fig, text: str) -> None:
    fig.text(0.5, 0.5, text, ha="center", va="center", rotation=30, fontsize=34, color="#9A9A9A", alpha=0.16,
             fontweight="bold", zorder=1000)


def save(fig, name: str, out_dir: Path) -> list[Path]:
    """Vector PDF + 600 dpi PNG + 600 dpi TIFF (LZW)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png", out_dir / f"{name}.tiff"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=DPI)
    fig.savefig(paths[2], dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return paths
