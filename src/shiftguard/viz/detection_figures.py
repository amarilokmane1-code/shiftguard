"""Figure 7: label-free signals against the real coverage drop (phase 6)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shiftguard.evaluation.detection import SIGNAL_LABELS, TARGET_LABELS, UNSIGNED
from shiftguard.viz.style import (
    DIAGNOSIS_COLORS,
    GRID,
    TEXT_SECONDARY,
    apply_style,
    label_points,
    save_figure,
)

POINT = DIAGNOSIS_COLORS["ASD"]


def _panel(
    ax: plt.Axes,
    g: pd.DataFrame,
    signal: str,
    target: str,
    corr: pd.Series,
    reference: float | None,
) -> None:
    if reference is not None and np.isfinite(reference):
        ax.axvline(reference, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":", zorder=1)
    ax.axhline(0, color=GRID, linewidth=1.0, zorder=1)
    ax.scatter(g[signal], g[target], s=20, color=POINT, zorder=3)
    q = corr["q_value"]
    q_txt = "q < 0.01" if q < 0.01 else f"q = {q:.2f}"
    ax.set_title(
        f"{SIGNAL_LABELS[signal]}\n\N{GREEK SMALL LETTER RHO} = {corr['rho']:.2f} "
        f"[{corr['ci_low']:.2f}, {corr['ci_high']:.2f}], {q_txt}\n"
        f"size-adjusted {corr['rho_partial']:.2f}",
        fontsize=7.5,
    )
    ax.tick_params(labelsize=7)


def figure_detection(
    site_sig: pd.DataFrame,
    corr: pd.DataFrame,
    reference: dict[str, float],
    stem: str | Path,
    model: str = "logreg",
    variant: str = "none",
) -> tuple[Path, Path]:
    """Draw Figure 7.

    Each label-free signal against the coverage drop of split CP, one dot per deployment
    site; dotted line = the signal's mean on within-site (no-shift) folds.
    """
    apply_style()
    g = site_sig[(site_sig["model"] == model) & (site_sig["variant"] == variant)]
    c = corr[(corr["model"] == model) & (corr["variant"] == variant)]
    c = c[c["target"] == "coverage_drop"].set_index("signal")
    signals = [s for s in UNSIGNED if s in c.index and g[s].notna().any()]
    ncol = 4
    nrow = int(np.ceil(len(signals) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 2.7 * nrow), sharey=True, squeeze=False)
    for ax, s in zip(axes.flat, signals, strict=False):
        _panel(ax, g, s, "coverage_drop", c.loc[s], reference.get(s))
    for ax in axes.flat[len(signals) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Coverage drop\n(target \N{MINUS SIGN} site coverage)")
    fig.suptitle(
        "Label-free signals vs real coverage drop per deployment site (split CP, LOSO; "
        "dotted = no-shift reference)",
        fontsize=9,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, stem)


def figure_class_gap(
    site_sig: pd.DataFrame,
    corr: pd.DataFrame,
    stem: str | Path,
    model: str = "logreg",
    variant: str = "none",
) -> tuple[Path, Path]:
    """Figure 7b: signed signals against the per-class coverage gap (ASD minus TDC)."""
    apply_style()
    g = site_sig[(site_sig["model"] == model) & (site_sig["variant"] == variant)]
    c = corr[(corr["model"] == model) & (corr["variant"] == variant)]
    c = c[c["target"] == "class_gap"].set_index("signal")
    fig, axes = plt.subplots(1, len(c), figsize=(3.3 * len(c), 3.0), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], c.index, strict=True):
        ax.axvline(0, color=GRID, linewidth=1.0, zorder=1)
        _panel(ax, g, s, "class_gap", c.loc[s], None)
    axes[0, 0].set_ylabel(TARGET_LABELS["class_gap"])
    fig.tight_layout()
    for ax, s in zip(axes[0], c.index, strict=True):
        label_points(ax, g[s], g["class_gap"], g["test_site"].tolist(), fontsize=5.5)
    return save_figure(fig, stem)
