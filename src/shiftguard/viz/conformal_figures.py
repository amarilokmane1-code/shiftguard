"""Phase 5 figures (primary model, logistic regression, unless stated).

Figure 3 (coverage per deployment site), 3b (per class), 4 (labelled target subjects and
the abstention trade-off), 5 (abstention and accuracy on non-abstained cases per site)
and S1 (prevalence-shift diagnostic).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shiftguard.viz.style import (
    DIAGNOSIS_COLORS,
    GRID,
    NEUTRAL,
    TEXT_SECONDARY,
    apply_style,
    save_figure,
)

PRIMARY = "logreg"
POINT = DIAGNOSIS_COLORS["ASD"]  # single-series marks (the figures' one data colour)
LABELS = {
    "split": "Split CP",
    "class_conditional": "Class-conditional",
    "split_aps": "Split CP (APS)",
    "class_conditional_aps": "Class-cond. (APS)",
    "split_aps_det": "Split CP (det. APS)",
    "weighted": "Weighted (cov. shift)",
    "weighted_clip": "Weighted, clipped",
    "label_shift_em": "Label-shift (EM)",
    "label_shift_oracle": "Label-shift (oracle)",
    "combat+split": "ComBat + split",
    "combat+class_conditional": "ComBat + class-cond.",
    "combat+weighted": "ComBat + weighted",
    "combat+weighted_clip": "ComBat + weighted, clipped",
    "mondrian_k0": "Mondrian, 0 labels",
    "mondrian_k5": "Mondrian, 5 labels",
    "mondrian_k10": "Mondrian, 10 labels",
    "mondrian_k20": "Mondrian, 20 labels",
}


def _loso(sites: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    s = sites[(sites["protocol"] == "loso") & (sites["model"] == PRIMARY)]
    return s[s["method"].isin(methods)]


def _site_order(sites: pd.DataFrame) -> list[str]:
    ref = sites[(sites["method"] == "split")].sort_values("coverage", ascending=False)
    return ref["test_site"].tolist()


def figure_coverage_per_site(
    sites: pd.DataFrame, stem: str | Path, alpha: float
) -> tuple[Path, Path]:
    """Figure 3: coverage per deployment site with Clopper-Pearson CIs, one panel per method.

    Filled markers: under-coverage significant after Benjamini-Hochberg (q < 0.05);
    hollow: not significant. Panel titles give the mean abstention rate.
    """
    apply_style()
    methods = [
        m
        for m in ("split", "class_conditional", "label_shift_em", "weighted_clip", "combat+split")
        if m in set(sites["method"])
    ]
    s = _loso(sites, methods)
    order = _site_order(s)
    y = {site: i for i, site in enumerate(order)}
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(2.15 * len(methods), 0.24 * len(order) + 1.4),
        sharey=True,
        squeeze=False,
    )
    lo_all = float(np.nanmin(s["ci_low"])) if len(s) else 0.5
    for ax, m in zip(axes[0], methods, strict=True):
        g = s[s["method"] == m]
        yy = g["test_site"].map(y).to_numpy()
        ax.axvline(1 - alpha, color=TEXT_SECONDARY, linewidth=0.9, zorder=1)
        ax.hlines(yy, g["ci_low"], g["ci_high"], color=POINT, linewidth=1.2, zorder=2)
        sig = (g["q_under"] < 0.05).to_numpy()
        ax.scatter(g["coverage"][sig], yy[sig], s=26, color=POINT, zorder=3, label="q < 0.05")
        ax.scatter(
            g["coverage"][~sig],
            yy[~sig],
            s=26,
            facecolor="white",
            edgecolor=POINT,
            linewidth=1.2,
            zorder=3,
        )
        ax.set_title(
            f"{LABELS.get(m, m)}\nmean site abstention {g['abstention'].mean():.0%}", fontsize=8
        )
        ax.set_xlim(max(0.0, min(lo_all, 0.6) - 0.02), 1.01)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Coverage")
    axes[0, 0].set_yticks(range(len(order)), order, fontsize=7)
    axes[0, 0].invert_yaxis()
    fig.suptitle(
        f"Coverage per deployment site (LOSO, target {1 - alpha:.0%}; "
        "filled = significantly below, BH q < 0.05)",
        fontsize=9,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    return save_figure(fig, stem)


def figure_class_coverage(sites: pd.DataFrame, stem: str | Path, alpha: float) -> tuple[Path, Path]:
    """Figure 3b: per-class coverage (ASD, TDC) per deployment site."""
    apply_style()
    methods = [
        m for m in ("split", "class_conditional", "label_shift_em") if m in set(sites["method"])
    ]
    s = _loso(sites, methods)
    order = _site_order(s)
    y = {site: i for i, site in enumerate(order)}
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(2.6 * len(methods), 0.24 * len(order) + 1.4),
        sharey=True,
        squeeze=False,
    )
    for ax, m in zip(axes[0], methods, strict=True):
        g = s[s["method"] == m]
        yy = g["test_site"].map(y).to_numpy()
        ax.axvline(1 - alpha, color=TEXT_SECONDARY, linewidth=0.9, zorder=1)
        for cls, dy, marker in (("asd", -0.15, "o"), ("tdc", 0.15, "s")):
            label = cls.upper()
            ax.scatter(
                g[f"coverage_{cls}"],
                yy + dy,
                s=22,
                marker=marker,
                color=DIAGNOSIS_COLORS[label],
                zorder=3,
                label=label,
            )
        n_under = int((g["q_under_asd"] < 0.05).sum() + (g["q_under_tdc"] < 0.05).sum())
        ax.set_title(f"{LABELS.get(m, m)}\n{n_under} site-classes below (BH)", fontsize=8)
        ax.set_xlim(0.0, 1.02)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Coverage within class")
    axes[0, 0].set_yticks(range(len(order)), order, fontsize=7)
    axes[0, 0].invert_yaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, stem)


def figure_tradeoff(
    head: pd.DataFrame, sites: pd.DataFrame, stem: str | Path, alpha: float
) -> tuple[Path, Path]:
    """Figure 4: Mondrian arms against target labels, and the abstention trade-off.

    Left: Mondrian mean-site and worst-site coverage against the number of labelled
    target subjects, annotated with the coverage gap (mean shortfall below target);
    middle: abstention. Right: abstention against worst-site coverage for every LOSO
    method that does not abstain on everyone.
    """
    apply_style()
    h = head[(head["protocol"] == "loso") & (head["model"] == PRIMARY)].copy()
    mond = h[h["method"].str.startswith("mondrian_k")].copy()
    mond["k"] = mond["method"].str.removeprefix("mondrian_k").astype(int)
    mond = mond.sort_values("k")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    ax = axes[0]
    ax.axhline(1 - alpha, color=GRID, linewidth=1.0)
    ax.plot(mond["k"], mond["mean_site_coverage"], "-o", color=TEXT_SECONDARY, label="mean site")
    ax.plot(mond["k"], mond["worst_site_coverage"], "--s", color=TEXT_SECONDARY, label="worst site")
    ax.set_xticks(mond["k"])
    ax.set_xlabel("Labelled target subjects (k)")
    ax.set_ylabel("Coverage")
    gaps = "\n".join(
        f"k = {k:>2}: {g:.3f}" for k, g in zip(mond["k"], mond["coverage_gap"], strict=True)
    )
    ax.text(
        0.03,
        0.04,
        f"coverage gap\n(mean shortfall)\n{gaps}",
        transform=ax.transAxes,
        fontsize=6.5,
        color=TEXT_SECONDARY,
        family="monospace",
        va="bottom",
    )
    ax.set_title("Mondrian: coverage", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    ax = axes[1]
    ax.plot(mond["k"], mond["abstention"], "-o", color=POINT)
    ax.set_xticks(mond["k"])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Labelled target subjects (k)")
    ax.set_ylabel("Abstention rate")
    ax.set_title("Mondrian: abstention", fontsize=9)
    ax = axes[2]
    pts = h[h["abstention"] < 0.999]  # arms that abstain on everyone are left out
    ax.axvline(1 - alpha, color=GRID, linewidth=1.0)
    ax.scatter(pts["worst_site_coverage"], pts["abstention"], s=24, color=POINT, zorder=3)
    # points cluster tightly: labels go in a column right of the axes, in the points'
    # vertical order, with leader lines (no overlapping text)
    pts = pts.sort_values("abstention")
    for i, (_, r) in enumerate(pts.iterrows()):
        ax.annotate(
            LABELS.get(r["method"], r["method"]),
            (r["worst_site_coverage"], r["abstention"]),
            xytext=(1.04, (i + 0.5) / len(pts)),
            textcoords="axes fraction",
            va="center",
            fontsize=6.5,
            color=TEXT_SECONDARY,
            arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.7},
        )
    ax.set_xlabel("Worst-site coverage")
    ax.set_ylabel("Abstention rate")
    ax.set_title("Every LOSO method", fontsize=9)
    fig.tight_layout()
    return save_figure(fig, stem)


def figure_abstention_per_site(sites: pd.DataFrame, stem: str | Path) -> tuple[Path, Path]:
    """Figure 5: abstention and accuracy on non-abstained cases per deployment site.

    One panel per method; grey bar = abstention rate (the set is not a single class),
    dot = accuracy among the subjects that received a single label. Both are rates on
    [0, 1]. For binary LAC sets without empty sets, mean set size = 1 + abstention.
    """
    apply_style()
    methods = [
        m
        for m in ("split", "combat+split", "weighted_clip", "mondrian_k20")
        if m in set(sites["method"])
    ]
    s = _loso(sites, methods)
    order = _site_order(s)
    y = {site: i for i, site in enumerate(order)}
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(2.3 * len(methods), 0.24 * len(order) + 1.5),
        sharey=True,
        squeeze=False,
    )
    for ax, m in zip(axes[0], methods, strict=True):
        g = s[s["method"] == m]
        yy = g["test_site"].map(y).to_numpy()
        ax.barh(yy, g["abstention"], height=0.6, color=NEUTRAL, zorder=2, label="abstention")
        ax.scatter(
            g["selective_accuracy"],
            yy,
            s=22,
            color=POINT,
            zorder=3,
            label="accuracy if not abstained",
        )
        ax.set_title(
            f"{LABELS.get(m, m)}\nsite means: abstention {g['abstention'].mean():.0%}, "
            f"accuracy {g['selective_accuracy'].mean():.0%}",
            fontsize=8,
        )
        ax.set_xlim(0, 1.02)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Rate (mean over seeds)")
    axes[0, 0].set_yticks(range(len(order)), order, fontsize=7)
    axes[0, 0].invert_yaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, stem)


def figure_prevalence(prev: pd.DataFrame, stem: str | Path) -> tuple[Path, Path]:
    """Figure S1: the prevalence-shift diagnostic per deployment site.

    Left: EM-estimated vs true ASD fraction. Right: calibration-in-the-large before and
    after the EM adjustment, against the true prevalence shift.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.0, zorder=1)
    ax.errorbar(
        prev["pi_target_true"],
        prev["pi_target_em"],
        yerr=prev["pi_target_em_sd"],
        fmt="o",
        color=POINT,
        markersize=5,
        elinewidth=1.0,
        zorder=3,
    )
    for _, r in prev.iterrows():
        ax.annotate(
            r["test_site"],
            (r["pi_target_true"], r["pi_target_em"]),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=6,
            color=TEXT_SECONDARY,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("True ASD fraction at the site")
    ax.set_ylabel("EM estimate (no labels)")
    ax.set_title("Prevalence estimated without labels", fontsize=9)
    ax = axes[1]
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    lim = float(np.nanmax(np.abs(prev["true_shift"]))) * 1.15
    xs = np.array([-lim, lim])
    ax.plot(
        xs,
        -xs,
        ":",
        color=TEXT_SECONDARY,
        linewidth=1.0,
        label="pure label shift, uninformative model",
    )
    ax.vlines(
        prev["true_shift"],
        prev["cil_test"],
        prev["cil_test_em"],
        color=GRID,
        linewidth=1.5,
        zorder=1,
    )
    ax.scatter(
        prev["true_shift"],
        prev["cil_test"],
        s=26,
        facecolor="white",
        edgecolor=POINT,
        linewidth=1.2,
        zorder=3,
        label="before EM",
    )
    ax.scatter(
        prev["true_shift"], prev["cil_test_em"], s=26, color=POINT, zorder=3, label="after EM"
    )
    for _, r in prev.iterrows():
        ax.annotate(
            r["test_site"],
            (r["true_shift"], r["cil_test"]),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=6,
            color=TEXT_SECONDARY,
        )
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("True prevalence shift (target \N{MINUS SIGN} source ASD fraction)")
    ax.set_ylabel("Mean P(ASD) \N{MINUS SIGN} ASD fraction")
    ax.set_title("Calibration-in-the-large per site", fontsize=9)
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    return save_figure(fig, stem)
