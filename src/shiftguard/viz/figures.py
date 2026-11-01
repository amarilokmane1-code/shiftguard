"""Publication figures. Each function takes tidy tables and returns the saved paths."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shiftguard.viz.style import (
    DIAGNOSIS_COLORS,
    GRID,
    NEUTRAL,
    SURFACE,
    TEXT_SECONDARY,
    apply_style,
    save_figure,
)

RAW_COLOR = "#7a7974"  # uncalibrated: dark neutral + dashed line (identity not by colour alone)
GRID_LINE = GRID


def figure_reliability(
    preds: pd.DataFrame, stem: str | Path, n_bins: int = 10
) -> tuple[Path, Path]:
    """Figure 6: reliability diagrams, and per-site calibration under LOSO.

    One row per model. Reliability panels (one per protocol): observed ASD frequency
    against mean predicted ASD probability, pooled over all test predictions and seeds,
    for raw (dashed, grey) and temperature-scaled (solid, blue) probabilities. Legend ECE
    values are the mean over seeds of the pooled-test ECE; ``null`` is the ECE a perfectly
    calibrated model would show at the same test size (binned ECE is biased upward in
    small samples).

    LOSO only, last column: calibration-in-the-large per deployment site, mean
    temperature-scaled P(ASD) minus the observed ASD fraction (dot: mean over seeds; bar:
    SD over seeds), with the grey band showing the 95% range expected from a perfectly
    calibrated model at that site's size. A pooled curve on the diagonal can hide large
    offsets that cancel between sites.
    """
    from shiftguard.models.calibration import (
        expected_calibration_error,
        null_ece,
        reliability_table,
    )

    apply_style()
    test = preds[preds["part"] == "test"]
    models = list(dict.fromkeys(test["model"]))
    protocols = [p for p in ("within_site", "loso") if p in set(test["protocol"])]
    with_sites = "loso" in protocols
    titles = {"within_site": "Within-site (no shift)", "loso": "Leave-one-site-out"}
    n_cols = len(protocols) + int(with_sites)
    fig, axes = plt.subplots(
        len(models),
        n_cols,
        figsize=(3.3 * n_cols, 3.3 * len(models)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0] * len(protocols) + [0.9] * int(with_sites)},
    )
    styles = {
        "p_raw": ("Uncalibrated", RAW_COLOR, "--", "s"),
        "p_cal": ("Temperature-scaled", DIAGNOSIS_COLORS["ASD"], "-", "o"),
    }

    def per_seed_mean(g: pd.DataFrame, fn) -> float:
        return float(np.mean([fn(d) for _, d in g.groupby("seed")]))

    for i, model in enumerate(models):
        for j, protocol in enumerate(protocols):
            ax = axes[i, j]
            g = test[(test["model"] == model) & (test["protocol"] == protocol)]
            ax.plot([0, 1], [0, 1], color=GRID_LINE, linewidth=1.0, zorder=1)
            for col, (label, color, ls, marker) in styles.items():
                rel = reliability_table(g["label"], g[col], n_bins)
                ece = per_seed_mean(
                    g, lambda d, c=col: expected_calibration_error(d["label"], d[c], n_bins)
                )
                ax.plot(
                    rel["mean_pred"],
                    rel["frac_pos"],
                    ls,
                    color=color,
                    marker=marker,
                    markersize=5,
                    linewidth=2.0,
                    label=f"{label} (ECE {ece:.3f})",
                    zorder=3,
                )
            null = per_seed_mean(g, lambda d: null_ece(d["p_cal"], n_bins))
            ax.text(
                0.97,
                0.03,
                f"null ECE at n={len(g) // g['seed'].nunique()}: {null:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color=TEXT_SECONDARY,
            )
            ax.set_title(f"{model} · {titles[protocol]}", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.legend(loc="upper left", fontsize=7)
            ax.set_xlabel("Mean predicted P(ASD)")
            ax.set_ylabel("Observed ASD frequency" if j == 0 else "")
        if with_sites:
            _site_offsets(
                axes[i, -1], test[(test["model"] == model) & (test["protocol"] == "loso")]
            )
            axes[i, -1].set_title(f"{model} · LOSO, per site", fontsize=9)
    fig.tight_layout()
    return save_figure(fig, stem)


def _site_offsets(ax: plt.Axes, g: pd.DataFrame) -> None:
    """Dot plot of mean P(ASD) minus ASD fraction per test site (calibrated probabilities)."""
    rows = []
    for site, s in g.groupby("test_site"):
        per_seed = s.groupby("seed").apply(
            lambda d: pd.Series(
                {
                    "offset": d["p_cal"].mean() - d["label"].mean(),
                    # SD of mean(p) - mean(y) when y ~ Bernoulli(p): sqrt(sum p(1-p)) / n
                    "null_sd": np.sqrt((d["p_cal"] * (1 - d["p_cal"])).sum()) / len(d),
                }
            ),
            include_groups=False,
        )
        rows.append(
            (
                site,
                per_seed["offset"].mean(),
                per_seed["offset"].std(ddof=1),
                per_seed["null_sd"].mean(),
            )
        )
    tab = pd.DataFrame(rows, columns=["site", "offset", "sd", "null_sd"]).sort_values("offset")
    y = np.arange(len(tab))
    ax.barh(
        y, 2 * 1.96 * tab["null_sd"], left=-1.96 * tab["null_sd"], height=0.6, color=GRID, zorder=1
    )
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8, zorder=2)
    ax.errorbar(
        tab["offset"],
        y,
        xerr=np.nan_to_num(tab["sd"]),
        fmt="o",
        color=DIAGNOSIS_COLORS["ASD"],
        markersize=5,
        elinewidth=1.0,
        capsize=0,
        zorder=3,
    )
    ax.set_yticks(y, tab["site"], fontsize=6.5)
    ax.grid(axis="y", visible=False)
    lim = max(0.3, float(np.nanmax(np.abs(tab["offset"]) + np.nan_to_num(tab["sd"]))) * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(
        "Mean P(ASD) \N{MINUS SIGN} ASD fraction\n(grey: 95% range if perfectly calibrated)"
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
