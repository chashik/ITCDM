"""Common figure style for the paper.

Figures are vector PDFs sized for the 160 mm text block of the GSU layout
(A4, left 30 mm, right 20 mm).  Colours follow a validated colour-blind-safe
categorical order (never cycled); magnitudes use a single-hue blue ramp and
signed quantities a blue-grey-red diverging ramp.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config as C  # noqa: E402

TEXT_W = 6.3  # inches (160 mm)

# categorical slots, fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec",
                     "#f2a7a6", "#e34948", "#9c1f1f"])
# heatmaps with values in the cells: pale ramps, on which INK has a contrast of at least 7:1 in every cell
SEQ_CELL = LinearSegmentedColormap.from_list(
    "seq_cell", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec"])
DIV_CELL = LinearSegmentedColormap.from_list(
    "div_cell", ["#6da7ec", "#9ec5f4", "#f0efec", "#f2a7a6", "#ec7877"])


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.prop_cycle": matplotlib.cycler(color=CAT),
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    })


def fig(h: float = 3.0, w: float = TEXT_W, **kw):
    setup()
    return plt.subplots(figsize=(w, h), **kw)


def _typeset(f):
    """Texts of a figure written in LaTeX notation (labels shared with the tables): ``---`` and ``--`` (ranges such as
    2003--2018) become en dashes, the dash of the text (spaced) and of ranges (closed up), and the escaped characters ``\\&``, ``\\%`` and ``\\_`` become plain ones."""
    from matplotlib.text import Text
    for t in f.findobj(Text):
        s = t.get_text()
        if "$" not in s and ("--" in s or "\\" in s):
            s = s.replace("---", "\u2013").replace("--", "\u2013")
            t.set_text(s.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_"))


def save(f, name: str):
    path = C.FIG_DIR / f"{name}.pdf"
    _typeset(f)
    f.savefig(path)
    plt.close(f)
    return path


def per_cent(v: float) -> str:
    """A share or a correlation as a whole number per 100 for a heatmap cell (true minus sign, no minus zero)."""
    return f"{int(round(100 * float(v)))}".replace("-", "\u2212")


def label_points(ax, xs, ys, labels, fontsize: float = 9, color: str = INK2):
    """Labels of scatter points, each placed at the first free position around its point so that no two labels
    overlap each other or the legend; a label with no free position is left out. Call it after the layout of the figure
    is fixed."""
    r = ax.figure.canvas.get_renderer()
    taken = [ax.get_legend().get_window_extent(r)] if ax.get_legend() else []
    for x, y, s in zip(xs, ys, labels):
        for dx, dy, ha, va in ((3, 2, "left", "bottom"), (3, -2, "left", "top"), (-3, 2, "right", "bottom"),
                               (-3, -2, "right", "top"), (0, 5, "center", "bottom"), (0, -5, "center", "top"),
                               (3, 11, "left", "bottom"), (3, -11, "left", "top"), (-3, 11, "right", "bottom"),
                               (-3, -11, "right", "top"), (0, 16, "center", "bottom"), (0, -16, "center", "top")):
            t = ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, color=color)
            b = t.get_window_extent(r)
            if all(not b.overlaps(o) for o in taken):
                taken.append(b)
                break
            t.remove()


def shade_splits(ax, ymax_frac: float = 1.0, label: bool = True):
    """Grey bands for the validation and test blocks of the chronological split."""
    import pandas as pd
    v0, v1 = pd.Timestamp(C.TRAIN_END), pd.Timestamp(C.VAL_END)
    t1 = pd.Timestamp(C.TEST_END)
    ax.axvspan(v0, v1, color="#d9d8d3", alpha=0.45, lw=0, zorder=0)
    ax.axvspan(v1, t1, color="#bdbcb6", alpha=0.35, lw=0, zorder=0)
    if label:
        tr = ax.get_xaxis_transform()
        ax.text(pd.Timestamp("2007-06-30"), 0.97, "train", transform=tr, ha="center",
                va="top", fontsize=8.5, color=INK2)
        ax.text(pd.Timestamp("2017-01-01"), 0.97, "validation", transform=tr, ha="center",
                va="top", fontsize=8.5, color=INK2)
        ax.text(pd.Timestamp("2022-10-01"), 0.97, "test", transform=tr, ha="center",
                va="top", fontsize=8.5, color=INK2)


def shade_recessions(ax, rec):
    """Hatched-free light bands for NBER recessions (rec: 0/1 daily series)."""
    import numpy as np
    r = rec.fillna(0).astype(int).to_numpy()
    idx = rec.index
    d = np.diff(np.r_[0, r, 0])
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        ax.axvspan(idx[s], idx[e - 1], color="#f2d7d5", alpha=0.6, lw=0, zorder=0)
