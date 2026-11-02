"""One consistent, colour-blind-safe style for every figure.

Categorical colours are used in a fixed order (never cycled); the first three slots
were checked with a CVD validator (worst adjacent colour-blind Delta E 9.2, normal-vision
Delta E 27.6 in OKLab x100). Diagnosis always uses the same two colours: ASD = slot 1
(blue), TDC = slot 2 (orange). Text never wears a series colour. Figures are saved as
PNG (300 dpi) and PDF.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")  # headless: CI, Kaggle, servers
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b9b8b3"  # "excluded" / "other": a neutral, never a categorical hue
TARGET_LINE = "#52514e"

CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua (below 3:1 on the surface: always direct-label or tabulate)
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#6250d6",  # 7 violet
    "#e34948",  # 8 red
)
DIAGNOSIS_COLORS = {"ASD": CATEGORICAL[0], "TDC": CATEGORICAL[1]}


def apply_style() -> None:
    """Set matplotlib defaults: thin marks, recessive axes, sans-serif text."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": TEXT_PRIMARY,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "pdf.fonttype": 42,  # embed TrueType: editable text in the PDF
            "svg.fonttype": "none",
        }
    )


def _overlap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Overlapping area of two (x0, y0, x1, y1) boxes in pixels."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def label_points(
    ax: Axes,
    xs: Sequence[float],
    ys: Sequence[float],
    labels: Sequence[str],
    *,
    fontsize: float = 6.0,
    color: str = TEXT_SECONDARY,
    avoid: tuple[Sequence[float], Sequence[float]] | None = None,
) -> None:
    """Label each point with its name, moving labels so that they do not overlap.

    Greedy and deterministic: each label tries positions on rings of growing distance
    around its point and keeps the first one that overlaps no earlier label, no point
    (including the extra ``avoid`` points, e.g. samples along error bars) and stays
    inside the axes; otherwise the least-overlapping one. Labels placed away from the
    nearest ring get a thin leader line. Call after the axes limits and the layout are
    final (after ``tight_layout``).
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    xs, ys = list(xs), list(ys)
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    box = ax.get_window_extent(renderer)
    r = 3.0 * px_per_pt  # marker radius
    point_boxes = [(x - r, y - r, x + r, y + r) for x, y in pts]
    if avoid is not None:
        extra = ax.transData.transform(np.column_stack([list(avoid[0]), list(avoid[1])]))
        point_boxes += [(x - r, y - r, x + r, y + r) for x, y in extra]
    sizes = []
    for text in labels:
        t = ax.text(0, 0, text, fontsize=fontsize)
        bb = t.get_window_extent(renderer)
        sizes.append((bb.width, bb.height))
        t.remove()
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (0, 1), (0, -1), (1, 0), (-1, 0)]
    rings = (3.0, 8.0, 14.0, 22.0)
    placed: list[tuple[float, ...]] = []
    for i, (text, (w, h)) in enumerate(zip(labels, sizes, strict=True)):
        x, y = pts[i]
        best = None
        for d in rings:
            for dx, dy in directions:
                ox, oy = dx * d * px_per_pt, dy * d * px_per_pt
                x0 = x + ox if dx > 0 else x + ox - w if dx < 0 else x - w / 2
                y0 = y + oy if dy > 0 else y + oy - h if dy < 0 else y - h / 2
                cand = (x0, y0, x0 + w, y0 + h)
                cost = sum(_overlap(cand, p) for p in placed)
                cost += sum(_overlap(cand, p) for j, p in enumerate(point_boxes) if j != i)
                outside = (
                    max(box.x0 - cand[0], 0)
                    + max(cand[2] - box.x1, 0)
                    + max(box.y0 - cand[1], 0)
                    + max(cand[3] - box.y1, 0)
                )
                cost += outside * h * 4
                if best is None or cost < best[0]:
                    best = (cost, d, dx, dy, cand)
                if cost == 0:
                    break
            if best is not None and best[0] == 0:
                break
        assert best is not None
        _, d, dx, dy, cand = best
        placed.append(cand)
        ax.annotate(
            text,
            (xs[i], ys[i]),
            xytext=(dx * d, dy * d),
            textcoords="offset points",
            ha="left" if dx > 0 else "right" if dx < 0 else "center",
            va="bottom" if dy > 0 else "top" if dy < 0 else "center",
            fontsize=fontsize,
            color=color,
            arrowprops={"arrowstyle": "-", "color": NEUTRAL, "linewidth": 0.6, "shrinkB": 3}
            if d > rings[0]
            else None,
        )


def spread_positions(values: Sequence[float], gap: float, lo: float, hi: float) -> np.ndarray:
    """Move 1-D positions apart by at least ``gap`` within [lo, hi], keeping their order.

    Used for a column of labels next to clustered points: each label stays as close to
    its point's height as the spacing allows, so leader lines stay short and in order.
    """
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="stable")
    out = np.maximum(v[order], lo)
    for i in range(1, len(out)):  # push up
        out[i] = max(out[i], out[i - 1] + gap)
    if len(out) and out[-1] > hi:  # then push down from the top
        out[-1] = hi
        for i in range(len(out) - 2, -1, -1):
            out[i] = min(out[i], out[i + 1] - gap)
    res = np.empty_like(out)
    res[order] = out
    return res


def save_figure(fig: Figure, stem: str | Path, dpi: int = 300) -> tuple[Path, Path]:
    """Save ``fig`` as ``<stem>.png`` (``dpi``) and ``<stem>.pdf``, then close it."""
    base = Path(stem)
    base.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = base.with_suffix(".png"), base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    return png, pdf
