"""Figure 2: the site effect on the connectome, before and after ComBat."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shiftguard.evaluation.site_effect import SiteEffect
from shiftguard.viz.style import (
    DIAGNOSIS_COLORS,
    GRID,
    NEUTRAL,
    TEXT_SECONDARY,
    apply_style,
    save_figure,
)

POINT = DIAGNOSIS_COLORS["ASD"]


def _embedding_panel(ax: plt.Axes, eff: SiteEffect, sites: np.ndarray, title: str) -> None:
    emb = eff.embedding
    ax.scatter(emb[:, 0], emb[:, 1], s=4, color=NEUTRAL, alpha=0.5, linewidths=0, zorder=1)
    cent = pd.DataFrame(emb, columns=["x", "y"]).groupby(pd.Series(sites, name="site")).mean()
    ax.scatter(cent["x"], cent["y"], s=28, color=POINT, edgecolor="white", linewidth=0.8, zorder=3)
    ev = eff.per_component["explained_variance_ratio"].to_numpy()
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    ax.set_title(title, fontsize=9)
    ax.grid(False)


def figure_site_effect(
    raw: SiteEffect, combat: SiteEffect, sites: np.ndarray, stem: str | Path
) -> tuple[Path, Path]:
    """Draw Figure 2.

    Left and middle: subjects (grey) and site centroids (blue) on the first two
    principal components, before and after ComBat. Right: site balanced accuracy and the
    variance of the leading components explained by site, before (hollow) and after
    (filled) ComBat.
    """
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    _embedding_panel(axes[0], raw, sites, "No harmonisation")
    _embedding_panel(axes[1], combat, sites, "ComBat (age, sex; no diagnosis)")
    ax = axes[2]
    rows = [
        (
            "Site balanced accuracy\n(5-fold CV)",
            raw.site_balanced_accuracy,
            combat.site_balanced_accuracy,
        ),
        ("R² of site\n(PC1-10, weighted)", raw.r2_site, combat.r2_site),
        ("Partial R² of site\n(beyond age, sex, dx)", raw.r2_site_partial, combat.r2_site_partial),
    ]
    for i, (_, before, after) in enumerate(rows):
        ax.plot([before, after], [i, i], color=GRID, linewidth=2.0, zorder=1)
        ax.scatter(before, i, s=40, facecolor="white", edgecolor=POINT, linewidth=1.4, zorder=3)
        ax.scatter(after, i, s=40, color=POINT, zorder=3)
        for v, dy in ((before, -0.28), (after, 0.32)):
            ax.annotate(
                f"{v:.2f}",
                (v, i),
                xytext=(0, dy * 30),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=TEXT_SECONDARY,
            )
    ax.vlines(raw.chance, -0.4, 0.4, color=TEXT_SECONDARY, linewidth=0.9, linestyles=":")
    ax.annotate(
        f"chance {raw.chance:.2f}",
        (raw.chance, -0.45),
        ha="left",
        va="top",
        fontsize=6.5,
        color=TEXT_SECONDARY,
    )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("hollow = no harmonisation, filled = ComBat")
    ax.set_title("Site signal in the connectome", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save_figure(fig, stem)
