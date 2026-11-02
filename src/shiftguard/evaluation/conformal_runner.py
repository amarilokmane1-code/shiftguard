"""Phase 5: conformal prediction under site shift, from saved baseline predictions.

The models are not refitted. For every (protocol, model, seed, deployment site) of a
baseline run, the stored calibration-split logits are divided into two halves, stratified
by site x diagnosis:

* half **A** fits the temperature (probability calibration) and gives the source class
  prior;
* half **B** is the conformal calibration set of every method.

So no calibration point is used twice, which matters for randomised APS and for the EM
prior adjustment (both depend on the calibrated probability values, not only on their
ranking). Every method is then applied to the same deployment subjects:

``split`` / ``class_conditional``
    LAC sets, marginal and per diagnosis (APS variants: ``*_aps``, randomised;
    ``split_aps_det`` once, as the documented binary failure case).
``weighted`` / ``weighted_clip``
    Covariate-shift weighted CP with domain-classifier weights (``shift.domain``), with
    and without clipping the weights to their [1%, 99%] calibration quantiles.
``label_shift_em`` / ``label_shift_oracle``
    Label-shift weighted CP (``conformal.label_shift``) with class weights
    ``pi_t / pi_s``, the target prior estimated by EM from the unlabelled deployment
    site (or, as an oracle reference only, its true prior). Scores use the EM-adjusted
    probabilities.
``mondrian_k{k}``
    ``k`` labelled deployment subjects (nested random draws) form the deployment site's
    own calibration group; evaluated on the subjects outside the largest draw, so every
    ``k`` is scored on the same people. Fewer than ``ceil((1-alpha)/alpha)`` labels give
    the full set (abstain on everyone).
``combat+split`` / ``combat+class_conditional`` / ``combat+weighted(_clip)``
    The same, from a baseline run refitted on ComBat-harmonised features (weighted arms
    with domain weights estimated on the harmonised features).

Every fold also records **label-free signals** for phase 6 (``evaluation.detection``): mean
P(ASD) and mean max-probability on calibration half B and on the deployment site, KS
two-sample tests between the two, and the split-CP abstention rate on both.

Within-site (no shift) folds get the split / class-conditional / APS methods only.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy import stats

from shiftguard.conformal import (
    ClassConditionalConformal,
    LabelShiftConformal,
    SplitConformal,
    WeightedConformal,
    conformal_quantile,
    lac_score_matrix,
)
from shiftguard.models.calibration import TemperatureScaler
from shiftguard.shift.domain import domain_classifier_weights
from shiftguard.shift.label_shift import adjust_to_prior, em_prior_shift, label_shift_weights
from shiftguard.utils.log import get_logger
from shiftguard.utils.seeding import derive_seed

log = get_logger(__name__)

FOLD_KEYS = ["protocol", "model", "seed", "test_site"]


# --------------------------------------------------------------------------- helpers
def calibration_halves(
    sites: NDArray[np.object_], labels: NDArray[np.intp], seed: int
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Positions of halves A (temperature) and B (conformal), stratified by site x label.

    Within each stratum, ``n // 2`` go to A and the rest to B.
    """
    rng = np.random.default_rng(seed)
    a, b = [], []
    strata = pd.Series([f"{s}|{y}" for s, y in zip(sites, labels, strict=True)])
    for _, pos in strata.groupby(strata, sort=True).groups.items():
        perm = rng.permutation(np.asarray(pos))
        a.append(perm[: len(perm) // 2])
        b.append(perm[len(perm) // 2 :])
    return np.sort(np.concatenate(a)), np.sort(np.concatenate(b))


def _probs(p1: NDArray[np.float64]) -> NDArray[np.float64]:
    p1 = np.clip(p1, 1e-12, 1 - 1e-12)
    return np.c_[1.0 - p1, p1]


def _outcomes(
    sets: NDArray[np.bool_], test: pd.DataFrame, method: str, mask: NDArray[np.bool_] | None = None
) -> pd.DataFrame:
    keep = np.ones(len(test), dtype=bool) if mask is None else mask
    t = test.loc[keep, [*FOLD_KEYS, "SUB_ID", "SITE_ID", "label"]].copy()
    t["method"] = method
    t["in0"] = sets[keep, 0].astype(np.int8)
    t["in1"] = sets[keep, 1].astype(np.int8)
    return t


# --------------------------------------------------------------------------- domain weights
def _domain_fold(
    features: NDArray[np.float64],
    seed: int,
    test_site: str,
    train_rows: NDArray[np.intp],
    cal_rows: NDArray[np.intp],
    test_rows: NDArray[np.intp],
    dc_cfg: dict[str, Any],
    base_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dw = domain_classifier_weights(
        features[train_rows],
        features[cal_rows],
        features[test_rows],
        components_grid=tuple(dc_cfg["pca_components_grid"]),
        C_grid=tuple(dc_cfg["C_grid"]),
        cv_folds=int(dc_cfg["cv_folds"]),
        random_state=derive_seed(base_seed, "domain", seed, test_site),
    )
    rows = pd.DataFrame(
        {
            "seed": seed,
            "test_site": test_site,
            "row": np.r_[cal_rows, test_rows],
            "weight": np.r_[dw.cal_weights, dw.test_weights],
        }
    )
    info = {
        "seed": seed,
        "test_site": test_site,
        "domain_auc": dw.auc,
        "domain_n_components": dw.n_components,
        "domain_C": dw.C,
    }
    return rows, info


def compute_domain_weights(
    features: NDArray[np.float64],
    preds: pd.DataFrame,
    train_rows: dict[tuple[int, str], NDArray[np.intp]],
    config: dict[str, Any],
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Domain-classifier weights for every LOSO (seed, site): rows and fold diagnostics."""
    dc_cfg = config["conformal"]["weighted"]["domain_classifier"]
    base = config["experiment"]["base_seed"]
    loso = preds[preds["protocol"] == "loso"]
    first_model = loso["model"].iloc[0]
    tasks = []
    for (seed, site), g in loso[loso["model"] == first_model].groupby(["seed", "test_site"]):
        cal = g.loc[g["part"] == "cal", "row"].to_numpy()
        test = g.loc[g["part"] == "test", "row"].to_numpy()
        tasks.append((int(seed), str(site), train_rows[(int(seed), str(site))], cal, test))
    log.info("Domain classifier: %d folds (n_jobs=%s)", len(tasks), n_jobs)
    t0 = time.perf_counter()
    out = Parallel(n_jobs=n_jobs)(delayed(_domain_fold)(features, *t, dc_cfg, base) for t in tasks)
    log.info("Domain classifier finished in %.1f min", (time.perf_counter() - t0) / 60)
    return (
        pd.concat([o[0] for o in out], ignore_index=True),
        pd.DataFrame([o[1] for o in out]),
    )


# --------------------------------------------------------------------------- one fold
def label_free_signals(
    pB: NDArray[np.float64], pT: NDArray[np.float64], split: SplitConformal
) -> dict[str, float]:
    """Signals computable at a deployment site without any of its labels.

    Compared with calibration half B (the exchangeable reference): mean P(ASD), mean
    max-probability (confidence), KS two-sample statistics and p-values on P(ASD) and on
    the max-probability, and the split-CP abstention rate. Half B's abstention uses the
    threshold fitted on half B itself, so it is the in-sample rate (slightly optimistic).
    """
    mB, mT = pB.max(axis=1), pT.max(axis=1)
    ks_p = stats.ks_2samp(pB[:, 1], pT[:, 1])
    ks_m = stats.ks_2samp(mB, mT)
    abst = [float(np.mean(split.predict_sets(p).sum(axis=1) != 1)) for p in (pB, pT)]
    return {
        "mean_p_cal": float(pB[:, 1].mean()),
        "mean_p_test": float(pT[:, 1].mean()),
        "mean_maxprob_cal": float(mB.mean()),
        "mean_maxprob_test": float(mT.mean()),
        "ks_prob": float(ks_p.statistic),
        "ks_prob_p": float(ks_p.pvalue),
        "ks_maxprob": float(ks_m.statistic),
        "ks_maxprob_p": float(ks_m.pvalue),
        "abstain_split_cal": abst[0],
        "abstain_split_test": abst[1],
    }


def conformal_fold(
    fold: pd.DataFrame,
    config: dict[str, Any],
    weights: pd.DataFrame | None = None,
    prefix: str = "",
    methods: str = "all",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply every method to one fold of baseline predictions.

    Parameters
    ----------
    fold
        Rows of one (protocol, model, seed, test_site) with ``part`` in {cal, test}.
    weights
        Domain weights of this fold indexed by dataset ``row`` (LOSO only).
    prefix
        Prepended to method names (``"combat+"``).
    methods
        ``"all"`` or ``"basic"`` (split and class-conditional LAC, plus the weighted arms
        when ``weights`` are given).
    """
    ccfg = config["conformal"]
    alpha = float(ccfg["alpha"])
    base = config["experiment"]["base_seed"]
    protocol, model, seed, site = (fold[k].iloc[0] for k in FOLD_KEYS)
    cal = fold[fold["part"] == "cal"].reset_index(drop=True)
    test = fold[fold["part"] == "test"].reset_index(drop=True)
    half_seed = derive_seed(base, "cal_halves", int(seed), str(site))
    pos_a, pos_b = calibration_halves(
        cal["SITE_ID"].to_numpy(dtype=object), cal["label"].to_numpy(), half_seed
    )
    A, B = cal.iloc[pos_a], cal.iloc[pos_b]
    scaler = TemperatureScaler().fit(A["logit"].to_numpy(), A["label"].to_numpy())
    pB = _probs(scaler.predict_proba(B["logit"].to_numpy()))
    pT = _probs(scaler.predict_proba(test["logit"].to_numpy()))
    yB, yT = B["label"].to_numpy(), test["label"].to_numpy()
    out: list[pd.DataFrame] = []

    def add(name: str, sets: NDArray[np.bool_], mask: NDArray[np.bool_] | None = None) -> None:
        out.append(_outcomes(sets, test, prefix + name, mask))

    split = SplitConformal(alpha).fit(pB, yB)
    add("split", split.predict_sets(pT))
    add("class_conditional", ClassConditionalConformal(alpha).fit(pB, yB).predict_sets(pT))
    pi_s1 = float(cal["label"].mean())  # source prior: calibration sites (A and B)
    info: dict[str, Any] = {
        "protocol": protocol,
        "model": model,
        "seed": int(seed),
        "test_site": site,
        "variant": prefix.rstrip("+") or "none",
        "n_cal_A": len(A),
        "n_cal_B": len(B),
        "n_test": len(test),
        "temperature": scaler.temperature_,
        "pi_source": pi_s1,
        "pi_target_true": float(yT.mean()),
        "cil_test": float(pT[:, 1].mean() - yT.mean()),
    }
    info |= label_free_signals(pB, pT, split)
    # Prevalence-shift diagnostic. For a source-calibrated model under PURE label shift,
    # mean P(ASD) - ASD fraction = -(pi_t - pi_s) * (1 - slope), where slope = mean P(ASD)
    # among ASD minus among TDC (on calibration half B). |offset| <= |prevalence shift|.
    slope = float(pB[yB == 1, 1].mean() - pB[yB == 0, 1].mean())
    pi_s = np.array([1.0 - pi_s1, pi_s1])
    em = em_prior_shift(pT, pi_s)  # unlabelled deployment data only
    info |= {
        "discrimination_slope": slope,
        "cil_label_shift_pred": float(-(yT.mean() - pi_s1) * (1.0 - slope)),
        "pi_target_em": float(em.target_prior[1]),
        "em_iter": em.n_iter,
        "em_converged": em.converged,
        "cil_test_em": float(em.adjusted_probs[:, 1].mean() - yT.mean()),
    }

    def weighted_arms() -> None:  # covariate shift: domain-classifier weights
        if weights is None or protocol != "loso":
            return
        wB = weights.loc[B["row"].to_numpy()].to_numpy()
        wT = weights.loc[test["row"].to_numpy()].to_numpy()
        for name, clip in (
            ("weighted", None),
            ("weighted_clip", tuple(ccfg["weighted"]["clip_quantiles"])),
        ):
            cp = WeightedConformal(alpha, clip_quantiles=clip).fit(pB, yB, wB)
            add(name, cp.predict_sets(pT, wT))
            q = cp.quantiles(wT)
            info[f"ess_{name}"] = cp.ess_
            info[f"inf_rate_{name}"] = float(np.mean(~np.isfinite(q)))
        info["median_test_weight"] = float(np.median(wT))

    if methods == "basic":
        weighted_arms()
        return pd.concat(out, ignore_index=True), info

    rs = derive_seed(base, "aps", int(seed), str(site), str(model))
    add(
        "split_aps",
        SplitConformal(alpha, "aps", randomized=True, random_state=rs).fit(pB, yB).predict_sets(pT),
    )
    add(
        "class_conditional_aps",
        ClassConditionalConformal(alpha, "aps", randomized=True, random_state=rs)
        .fit(pB, yB)
        .predict_sets(pT),
    )
    if ccfg.get("report_deterministic_aps_once", True):
        add("split_aps_det", SplitConformal(alpha, "aps").fit(pB, yB).predict_sets(pT))
    if protocol != "loso":
        return pd.concat(out, ignore_index=True), info

    # ---- label shift: EM prior on the unlabelled deployment site, then weighted CP
    pi_true = np.array([1.0 - yT.mean(), yT.mean()])
    for name, pi_t, pT_adj in (
        ("label_shift_em", em.target_prior, em.adjusted_probs),
        ("label_shift_oracle", pi_true, adjust_to_prior(pT, pi_s, pi_true)),
    ):
        pB_adj = adjust_to_prior(pB, pi_s, pi_t)
        cp = LabelShiftConformal(alpha).fit(pB_adj, yB, label_shift_weights(pi_s, pi_t))
        add(name, cp.predict_sets(pT_adj))

    weighted_arms()

    # ---- Mondrian: k labelled deployment subjects as the site's own calibration group
    ks = sorted(int(k) for k in ccfg["n_target_labelled"])
    order = np.random.default_rng(derive_seed(base, "mondrian", int(seed), str(site))).permutation(
        len(test)
    )
    k_max = max(ks)
    eval_mask = np.zeros(len(test), dtype=bool)
    eval_mask[order[k_max:]] = True
    sT = lac_score_matrix(pT)
    for k in ks:
        lab = order[:k]
        q = conformal_quantile(sT[lab, yT[lab]], alpha) if k else np.inf
        add(f"mondrian_k{k}", sT <= q, eval_mask)
    info["n_mondrian_eval"] = int(eval_mask.sum())
    return pd.concat(out, ignore_index=True), info


# --------------------------------------------------------------------------- whole run
def run_conformal(
    preds: pd.DataFrame,
    config: dict[str, Any],
    domain_weights: pd.DataFrame | None = None,
    combat_preds: pd.DataFrame | None = None,
    combat_weights: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All folds of a baseline run (plus an optional ComBat run): outcomes and diagnostics."""
    outs, infos = [], []

    def index(w: pd.DataFrame | None) -> dict:
        if w is None:
            return {}
        return {k: g.set_index("row")["weight"] for k, g in w.groupby(["seed", "test_site"])}

    for run, w_all, prefix, methods in (
        (preds, index(domain_weights), "", "all"),
        (combat_preds, index(combat_weights), "combat+", "basic"),
    ):
        if run is None:
            continue
        if prefix:
            run = run[run["protocol"] == "loso"]
        for key, fold in run.groupby(FOLD_KEYS, sort=True):
            w = w_all.get((int(key[2]), str(key[3]))) if key[0] == "loso" else None
            o, info = conformal_fold(fold, config, w, prefix=prefix, methods=methods)
            outs.append(o)
            infos.append(info)
    return pd.concat(outs, ignore_index=True), pd.DataFrame(infos)


def train_rows_from_splits(
    config: dict[str, Any], sites: NDArray[np.object_], labels: NDArray[np.intp]
) -> dict:
    """Training rows of every LOSO split (they are not stored in the predictions)."""
    from shiftguard.evaluation.protocols import make_splits

    return {
        (sp.seed, str(sp.test_site)): sp.train
        for sp in make_splits("loso", sites, labels, config["experiment"])
    }


def load_run_predictions(run_dir: Path) -> pd.DataFrame:
    """Read ``predictions.csv.gz`` of a baseline run."""
    path = Path(run_dir) / "predictions.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: not a baseline run folder")
    return pd.read_csv(path, dtype={"SITE_ID": str, "test_site": str})
