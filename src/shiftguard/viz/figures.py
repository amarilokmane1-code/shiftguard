"""Publication figures. Each function takes tidy tables and returns the saved paths."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shiftguard.viz.style import (
    DIAGNOSIS_COLORS,
    NEUTRAL,
    SURFACE,
    TEXT_SECONDARY,
    apply_style,
    save_figure,
)


def figure_data_map(
    qc_summary: pd.DataFrame, stem: str | Path, min_per_class: int | None = None
) -> tuple[Path, Path]:
    """Figure 1: subjects per site and diagnosis after QC, with excluded subjects.

    One horizontal stacked bar per site (sorted by size after QC): ASD and TDC kept after
    QC, then the subjects excluded by QC in neutral grey. Sites dropped entirely by the
    minimum-size rule are marked. The total kept is labelled at the end of each bar.

    Parameters
    ----------
    qc_summary
        Output of :func:`shiftguard.data.qc.qc_summary` (one row per site plus TOTAL).
    stem
        Output path without extension.
    min_per_class
        The minimum-size rule, quoted in the label of dropped sites.
    """
    apply_style()
    df = qc_summary[qc_summary["SITE_ID"] != "TOTAL"].copy()
    df["n_excluded"] = df["n_before"] - df["n_after"]
    df = df.sort_values(["n_after", "n_before"], ascending=True).reset_index(drop=True)
    total = qc_summary.loc[qc_summary["SITE_ID"] == "TOTAL"].iloc[0]
    n_sites_kept = int((df["n_after"] > 0).sum())

    fig, ax = plt.subplots(figsize=(7.0, 0.26 * len(df) + 1.4))
    y = np.arange(len(df))
    left = np.zeros(len(df))
    segments = [
        ("n_after_ASD", f"ASD (kept, n = {int(total['n_after_ASD'])})", DIAGNOSIS_COLORS["ASD"]),
        ("n_after_TDC", f"TDC (kept, n = {int(total['n_after_TDC'])})", DIAGNOSIS_COLORS["TDC"]),
        (
            "n_excluded",
            f"Excluded by QC (n = {int(total['n_before'] - total['n_after'])})",
            NEUTRAL,
        ),
    ]
    for column, label, color in segments:
        values = df[column].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            height=0.68,
            color=color,
            label=label,
            edgecolor=SURFACE,
            linewidth=1.0,  # surface gap between stacked segments
        )
        left += values

    for yi, row in df.iterrows():
        kept = int(row["n_after"])
        rule = f"< {min_per_class} per class after QC" if min_per_class else "site too small"
        text = f"{kept}" if kept > 0 else f"site dropped ({rule})"
        ax.text(
            row["n_before"] + 1.5,
            yi,
            text,
            va="center",
            ha="left",
            fontsize=7.5,
            color=TEXT_SECONDARY,
        )

    ax.set_yticks(y, df["SITE_ID"])
    ax.set_xlabel("Subjects")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, df["n_before"].max() * 1.22)
    ax.tick_params(axis="y", length=0)
    ax.set_title(
        f"ABIDE I after QC: {int(total['n_after'])} of {int(total['n_before'])} subjects "
        f"from {n_sites_kept} sites"
    )
    ax.legend(loc="lower right", handlelength=1.2)
    return save_figure(fig, stem)
