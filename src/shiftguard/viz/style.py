"""One consistent, colour-blind-safe style for every figure.

Categorical colours are used in a fixed order (never cycled); the first three slots
were checked with a CVD validator (worst adjacent colour-blind Delta E 9.2, normal-vision
Delta E 27.6 in OKLab x100). Diagnosis always uses the same two colours: ASD = slot 1
(blue), TDC = slot 2 (orange). Text never wears a series colour. Figures are saved as
PNG (300 dpi) and PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # headless: CI, Kaggle, servers
import matplotlib.pyplot as plt
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


def save_figure(fig: Figure, stem: str | Path, dpi: int = 300) -> tuple[Path, Path]:
    """Save ``fig`` as ``<stem>.png`` (``dpi``) and ``<stem>.pdf``, then close it."""
    base = Path(stem)
    base.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = base.with_suffix(".png"), base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    return png, pdf
