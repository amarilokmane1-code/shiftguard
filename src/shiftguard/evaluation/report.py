"""Phase 7: rebuild every table and figure from saved runs and write docs/RESULTS.md.

The final runs are listed in a manifest, ``<reports>/final_runs.yaml`` (written by
``shiftguard all``, editable by hand):

.. code-block:: yaml

    simulate: 2026-11-01_175354_simulate
    baseline: 2026-11-01_223647_loso
    baseline_combat: 2026-11-02_101130_loso_combat
    ...

``shiftguard report`` then

1. regenerates the tables and figures of every listed run that is present on disk
   (from saved predictions / outcomes: nothing is refitted);
2. writes ``docs/RESULTS.md`` from the CSV tables in ``<reports>/tables``;
3. replaces the key-results table of the README between the markers
   ``<!-- key-results:start -->`` and ``<!-- key-results:end -->``.

Every number in RESULTS.md and in the README block comes from those CSV files. Run
folders are not tracked by git (``reports/runs/`` is ignored); the tables are. On a
fresh clone, step 1 is skipped for missing runs and steps 2-3 still work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from shiftguard.utils.log import get_logger
from shiftguard.utils.tables import to_markdown

log = get_logger(__name__)

MANIFEST = "final_runs.yaml"
MANIFEST_KEYS = (
    "simulate",
    "baseline",
    "baseline_combat",
    "baseline_droprois",
    "conformal",
    "conformal_alpha0p2",
    "conformal_droprois",
    "site_effect",
    "detection",
)
README_START = "<!-- key-results:start -->"
README_END = "<!-- key-results:end -->"


# --------------------------------------------------------------------------- manifest
def load_manifest(path: Path) -> dict[str, str]:
    """Read the manifest; unknown keys are an error (typos must not pass silently)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(data) - set(MANIFEST_KEYS)
    if unknown:
        raise ValueError(f"unknown manifest keys: {sorted(unknown)}")
    return {k: str(v) for k, v in data.items() if v}


def save_manifest(path: Path, runs: dict[str, str]) -> None:
    """Write the manifest in the canonical key order."""
    lines = [
        "# Run folders (reports/runs/) that the final tables, figures and RESULTS.md come from."
    ]
    lines += [f"{k}: {runs[k]}" for k in MANIFEST_KEYS if runs.get(k)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_provenance(run_dir: Path) -> dict[str, Any]:
    """Creation time, git commit and dirty flag of a run folder (if present)."""
    meta = Path(run_dir) / "metadata.json"
    if not meta.is_file():
        return {"present": False}
    m = json.loads(meta.read_text(encoding="utf-8"))
    git = m.get("git") or {}
    return {
        "present": True,
        "created": m.get("created", ""),
        "commit": (git.get("commit") or "")[:7],
        "dirty": bool(git.get("dirty")),
    }


# --------------------------------------------------------------------------- regeneration
def regenerate(manifest: dict[str, str], reports: Path) -> dict[str, str]:
    """Rebuild tables and figures of every manifest run present on disk.

    Returns ``{key: "regenerated" | "missing"}``.
    """
    from shiftguard.config import load_config

    status: dict[str, str] = {}
    for key in MANIFEST_KEYS:
        name = manifest.get(key)
        if not name:
            continue
        run_dir = reports / "runs" / name
        if not (run_dir / "metadata.json").is_file():
            log.warning("%s: run folder %s not found; keeping the tracked tables", key, run_dir)
            status[key] = "missing"
            continue
        cfg = load_config(run_dir / "config.yaml") if key != "simulate" else {}
        _REGEN[key.split("_")[0]](run_dir, cfg, reports)
        status[key] = "regenerated"
        log.info("%s: regenerated from %s", key, name)
    return status


def _regen_simulate(run_dir: Path, _cfg: dict[str, Any], reports: Path) -> None:
    from shiftguard.evaluation.simulation import write_simulation_tables

    sim = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["simulate"]
    df = pd.read_csv(run_dir / "conformal_synthetic.csv", dtype={"p_below": str})
    write_simulation_tables(df, reports, sim["n_trials"], sim["seed"], run_dir=run_dir)


def _regen_baseline(run_dir: Path, cfg: dict[str, Any], reports: Path) -> None:
    from shiftguard.evaluation.runner import write_baseline_outputs

    preds = pd.read_csv(run_dir / "predictions.csv.gz", dtype={"SITE_ID": str, "test_site": str})
    info = pd.read_csv(run_dir / "fold_info.csv", dtype={"test_site": str})
    write_baseline_outputs(preds, info, cfg, run_dir, reports, save_raw=False)


def _regen_conformal(run_dir: Path, cfg: dict[str, Any], reports: Path) -> None:
    from shiftguard.evaluation.conformal_report import write_outputs

    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    names = {"baseline": meta.get("baseline", ""), "combat": meta.get("combat", "")}
    outcomes = pd.read_csv(run_dir / "outcomes.csv.gz", dtype={"SITE_ID": str, "test_site": str})
    diag = pd.read_csv(run_dir / "fold_diagnostics.csv", dtype={"test_site": str})
    write_outputs(outcomes, diag, cfg, run_dir, reports, names)


def _regen_site(run_dir: Path, cfg: dict[str, Any], reports: Path) -> None:
    from shiftguard.evaluation.site_effect import load_site_effect, write_site_effect_outputs

    effects, sites = load_site_effect(run_dir)
    write_site_effect_outputs(
        effects, sites, run_dir.name, reports, cfg["experiment"].get("report_tag", "")
    )


def _regen_detection(run_dir: Path, cfg: dict[str, Any], reports: Path) -> None:
    from shiftguard.evaluation.detection import write_detection_outputs

    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    site_sig = pd.read_csv(run_dir / "site_signals.csv", dtype={"test_site": str})
    corr = pd.read_csv(run_dir / "correlations.csv")
    ref = json.loads((run_dir / "within_site_reference.json").read_text(encoding="utf-8"))
    write_detection_outputs(
        site_sig, corr, ref, cfg, run_dir.name, meta.get("conformal_run", ""), reports
    )


_REGEN = {
    "simulate": _regen_simulate,
    "baseline": _regen_baseline,
    "conformal": _regen_conformal,
    "site": _regen_site,
    "detection": _regen_detection,
}


# --------------------------------------------------------------------------- formatting
def _read(tables: Path, name: str) -> pd.DataFrame | None:
    path = tables / f"{name}.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path, dtype={"test_site": str})


def _f(v: Any, d: int = 3) -> str:
    """Format a number; tables may also store pre-formatted strings or "-" for missing."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "–"
    return "–" if np.isnan(x) else f"{x:.{d}f}"


def _ci(d: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{d:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


METHOD_LABELS = {
    "split": "Split CP",
    "class_conditional": "Class-conditional CP",
    "label_shift_em": "Label-shift CP (EM prior)",
    "label_shift_oracle": "Label-shift CP (oracle prior, uses labels)",
    "weighted": "Weighted CP (covariate shift)",
    "weighted_clip": "Weighted CP (covariate shift, clipped)",
    "combat+split": "ComBat + split CP",
    "combat+class_conditional": "ComBat + class-conditional CP",
    "combat+weighted_clip": "ComBat + weighted CP (clipped)",
    "mondrian_k10": "Mondrian (10 target labels)",
    "mondrian_k20": "Mondrian (20 target labels)",
    "split_aps": "Split CP (randomised APS)",
}
KEY_METHODS = [
    "split",
    "class_conditional",
    "label_shift_em",
    "weighted_clip",
    "combat+split",
    "combat+weighted_clip",
    "mondrian_k20",
]


def _baseline_row(base: pd.DataFrame | None, protocol: str, model: str = "logreg") -> dict:
    if base is None:
        return {}
    r = base[(base["protocol"] == protocol) & (base["model"] == model)]
    return {} if r.empty else r.iloc[0].to_dict()


def key_results_table(tables: Path) -> pd.DataFrame | None:
    """Build the README key-results table from the headline and baseline tables."""
    head = _read(tables, "conformal_headline")
    if head is None:
        return None
    base, base_c = _read(tables, "baselines"), _read(tables, "baselines_combat")
    h = head[head["model"] == "logreg"].set_index(["protocol", "method"])
    rows = []
    for protocol, method in [("within_site", "split")] + [("loso", m) for m in KEY_METHODS]:
        if (protocol, method) not in h.index:
            continue
        r = h.loc[(protocol, method)]
        b = _baseline_row(base_c if method.startswith("combat+") else base, protocol)
        rows.append(
            {
                "Setting": "Within-site" if protocol == "within_site" else "LOSO",
                "Method": METHOD_LABELS.get(method, method),
                "AUC": _f(_num(b.get("auc"))),
                "Balanced acc.": _f(_num(b.get("balanced_accuracy"))),
                "Marginal coverage": _f(r["coverage"]),
                "Worst-site coverage": _f(r.get("worst_site_coverage")),
                "Worst site-class": _f(r.get("worst_site_class_coverage")),
                "Coverage gap": _f(r.get("coverage_gap")),
                "Abstention": _f(r["abstention"]),
                "Accuracy (kept)": _f(r.get("selective_accuracy")),
            }
        )
    return pd.DataFrame(rows)


def _num(v: Any) -> float:
    """Baseline tables store ``mean ± sd`` strings or floats; return the mean."""
    if v is None:
        return float("nan")
    if isinstance(v, int | float):
        return float(v)
    return float(str(v).split("±")[0])


def update_readme(readme: Path, tables: Path) -> bool:
    """Replace the README key-results block; return False if the markers are missing."""
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    tab = key_results_table(tables)
    if tab is None:
        return False
    a = text.index(README_START) + len(README_START)
    b = text.index(README_END)
    block = "\n" + to_markdown(tab) + "\n"
    readme.write_text(text[:a] + block + text[b:], encoding="utf-8", newline="\n")
    return True


# --------------------------------------------------------------------------- RESULTS.md
def build_results_md(reports: Path, manifest: dict[str, str]) -> str:
    """Assemble docs/RESULTS.md from the CSV tables (no number is typed by hand)."""
    t = reports / "tables"
    out: list[str] = [
        "# Results",
        "",
        "*Generated by `shiftguard report` from the CSV tables in `reports/tables/`, which "
        "come from the run folders listed below. Do not edit by hand; edit the code and "
        "rerun.*",
        "",
        "**Research software. Nothing here supports clinical use.** Primary model: L2 "
        "logistic regression; primary score: LAC; α = 0.1 (target coverage 0.90) unless "
        "stated; 10 seeds; LOSO = leave one site out.",
        "",
        "## Provenance",
        "",
    ]
    prov = []
    for k in MANIFEST_KEYS:
        if k in manifest:
            p = run_provenance(reports / "runs" / manifest[k])
            prov.append(
                {
                    "analysis": k,
                    "run folder": manifest[k],
                    "git commit": p.get("commit", "–") if p["present"] else "not on disk",
                    "uncommitted changes": ("yes" if p.get("dirty") else "no")
                    if p["present"]
                    else "–",
                }
            )
    if prov:
        out += [
            to_markdown(pd.DataFrame(prov)),
            "",
            "*Uncommitted changes* means code, configs or docs differed from the commit "
            "when the run started. Tables and figures under `reports/` rewritten by an "
            "earlier step of the same pipeline are outputs and do not count.",
            "",
        ]
        if any(r["uncommitted changes"] == "yes" for r in prov):
            out += [
                "Runs marked *uncommitted changes* were made from a working tree with edits "
                "not yet committed (development of that phase); the code that produced them "
                "was committed at the end of the phase.",
                "",
            ]

    cohort = _read(t, "cohort_summary")
    if cohort is not None and {"SITE_ID", "n", "pct_ASD", "pct_male"} <= set(cohort.columns):
        c = cohort[~cohort["SITE_ID"].isin(["ALL", "TOTAL"])]  # drop the total row
        n = int(c["n"].sum())
        pct_asd = float((c["n"] * c["pct_ASD"]).sum() / n)
        pct_male = float((c["n"] * c["pct_male"]).sum() / n)
        out += [
            "## Cohort",
            "",
            f"ABIDE I (PCP, CPAC, filt_noglobal, CC200) after QC: **{n} subjects** from "
            f"**{len(c)} sites** ({pct_asd:.1f}% ASD, {pct_male:.1f}% male; "
            f"{int(c['n'].min())}–{int(c['n'].max())} subjects per site). Per site: "
            "[cohort_summary.md](../reports/tables/cohort_summary.md); QC: "
            "[qc_summary.md](../reports/qc_summary.md).",
            "",
        ]

    base = _read(t, "baselines")
    if base is not None:
        cols = [
            c
            for c in (
                "protocol",
                "model",
                "auc",
                "auc_site_worst",
                "balanced_accuracy",
                "ece_cal",
                "ece_null",
                "brier",
            )
            if c in base
        ]
        out += [
            "## Baselines (phase 4)",
            "",
            "Mean ± SD over seeds. ECE after temperature scaling, next to the ECE a "
            "perfectly calibrated model would show at the same size (null).",
            "",
            to_markdown(base[cols]),
            "",
        ]

    se = _read(t, "site_effect")
    if se is not None:
        out += [
            "## Site effect before / after ComBat (Figure 2)",
            "",
            "Site predictability from the connectome (5-fold CV balanced accuracy) and R² of "
            "site on the 10 leading principal components (variance-weighted; partial = "
            "beyond age, sex and diagnosis).",
            "",
            to_markdown(se),
            "",
            "![Figure 2](../reports/figures/fig2_site_effect.png)",
            "",
        ]

    head = _read(t, "conformal_headline")
    if head is not None:
        cols = [
            "protocol",
            "model",
            "method",
            "coverage",
            "coverage_asd",
            "coverage_tdc",
            "worst_site_coverage",
            "worst_site",
            "worst_site_class_coverage",
            "coverage_gap",
            "n_sites_under",
            "n_site_classes_under",
            "abstention",
            "selective_accuracy",
            "between_site_sd",
        ]
        h = head[[c for c in cols if c in head]]
        out += [
            "## Conformal prediction under site shift (phase 5)",
            "",
            "Coverage pooled over sites (mean over seeds); site metrics on seed-averaged "
            "site coverage; `n_*_under` = number of sites (site-classes) significantly "
            "below target (exact binomial test, Benjamini–Hochberg q < 0.05); "
            "`between_site_sd` = empirical-Bayes SD of true site coverage.",
            "",
            to_markdown(h),
            "",
            "![Figure 3](../reports/figures/fig3_coverage_per_site.png)",
            "",
            "Per-class coverage: [Figure 3b](../reports/figures/fig3b_class_coverage.png); "
            "labelled target subjects: [Figure 4](../reports/figures/fig4_target_labels.png); "
            "abstention per site: [Figure 5](../reports/figures/fig5_abstention_per_site.png); "
            "per site and class with Clopper–Pearson intervals: "
            "[conformal_sites.md](../reports/tables/conformal_sites.md).",
            "",
        ]
    comp = _read(t, "conformal_comparisons")
    if comp is not None:
        c = comp[comp["model"] == "logreg"].copy()
        c["difference (site bootstrap)"] = [
            _ci(d, lo, hi) for d, lo, hi in zip(c["diff"], c["ci_low"], c["ci_high"], strict=True)
        ]
        c["q"] = c["q_value"].map(_f)
        cols = ["method", "metric", "difference (site bootstrap)", "q"]
        if "ci_low_2stage" in c:
            c["two-stage 95% CI"] = [
                f"[{lo:+.3f}, {hi:+.3f}]"
                for lo, hi in zip(c["ci_low_2stage"], c["ci_high_2stage"], strict=True)
            ]
            c["q (two-stage)"] = c["q_2stage"].map(_f)
            cols += ["two-stage 95% CI", "q (two-stage)"]
        out += [
            "### Each method minus split CP (logistic regression, LOSO)",
            "",
            "Primary: paired bootstrap over sites (2000 resamples), 95% percentile CI, BH "
            "within model. Secondary: two-stage bootstrap (sites, then subjects within each "
            "drawn site). Shortfall = mean over sites of max(0, 0.90 − coverage).",
            "",
            to_markdown(c[cols]),
            "",
        ]
    prev = _read(t, "prevalence_shift")
    if prev is not None:
        cols = [
            "test_site",
            "n_test",
            "pi_target_true",
            "pi_source",
            "true_shift",
            "pi_target_em",
            "cil_test",
            "cil_label_shift_pred",
            "cil_excess",
            "domain_auc",
        ]
        out += [
            "### Prevalence-shift diagnostic (Figure S1)",
            "",
            "Per deployment site, mean over seeds: true and source ASD fraction, EM estimate "
            "from unlabelled data, calibration-in-the-large (`cil` = mean P(ASD) − ASD "
            "fraction), the offset pure label shift would produce, and the excess.",
            "",
            to_markdown(prev[[c for c in cols if c in prev]]),
            "",
        ]
    for name, title in (
        ("conformal_headline_alpha0p2", "### Sensitivity: α = 0.2 (target 0.80)"),
        ("conformal_headline_droprois", "### Sensitivity: ROI-level exclusion (19 sites)"),
    ):
        d = _read(t, name)
        if d is not None:
            d = d[(d["model"] == "logreg") & (d["protocol"] == "loso")]
            d = d[d["method"].isin(KEY_METHODS)]
            cols = [
                "method",
                "coverage",
                "worst_site_coverage",
                "worst_site_class_coverage",
                "coverage_gap",
                "abstention",
            ]
            out += [title, "", to_markdown(d[cols]), ""]

    det = _read(t, "detection_correlations")
    if det is not None:
        d = det[(det["model"] == "logreg") & (det["variant"] == "none")].copy()
        d["rho [95% CI]"] = [
            _ci(r, lo, hi, 2) for r, lo, hi in zip(d["rho"], d["ci_low"], d["ci_high"], strict=True)
        ]
        d["leave-one-site-out"] = [
            f"{lo:+.2f} to {hi:+.2f}"
            for lo, hi in zip(d["rho_loo_min"], d["rho_loo_max"], strict=True)
        ]
        d["q"] = d["q_value"].map(_f)
        d["size-adjusted rho"] = d["rho_partial"].map(lambda v: _f(v, 2))
        d["q (adjusted)"] = d["q_partial"].map(_f)
        out += [
            "## Label-free shift detection (phase 6, exploratory)",
            "",
            "Spearman ρ across the 18 deployment sites between each label-free signal and "
            "the real coverage loss of split CP; bootstrap CI over sites, permutation p, BH "
            "per setting, range over leave-one-site-out, and partial ρ given site size. MLP "
            "and ComBat settings: "
            "[detection_correlations.md](../reports/tables/detection_correlations.md).",
            "",
            to_markdown(
                d[
                    [
                        "signal",
                        "target",
                        "rho [95% CI]",
                        "leave-one-site-out",
                        "q",
                        "size-adjusted rho",
                        "q (adjusted)",
                    ]
                ]
            ),
            "",
            "![Figure 7](../reports/figures/fig7_detection.png)",
            "",
            "![Figure 7b](../reports/figures/fig7b_detection_class_gap.png)",
            "",
        ]
    sim = _read(t, "conformal_synthetic")
    if sim is not None:
        cols = [
            c for c in ("experiment", "method", "target", "coverage_95ci", "theory") if c in sim
        ]
        out += [
            "## Synthetic validity checks of the conformal core (phase 2)",
            "",
            to_markdown(sim[cols]),
            "",
        ]
    return "\n".join(out).rstrip() + "\n"
