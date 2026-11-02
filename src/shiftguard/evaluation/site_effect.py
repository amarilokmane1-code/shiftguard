"""Describe the site effect on the connectivity features, before and after ComBat.

Two descriptions, both computed on all subjects of the cohort (descriptive, not part of
any evaluation fold):

**Site predictability.** How well can the acquisition site be predicted from the
connectome? Standardisation, PCA and a multinomial L2 logistic regression are fitted
inside 5-fold cross-validation stratified by site (no fold sees its test subjects).
Reported as balanced accuracy, with chance = 1 / number of sites.

**Variance explained by site.** The features are standardised and reduced to their
leading principal components. Each component is regressed on site dummies, and on
site + age + sex + diagnosis. Reported per component and as a variance-weighted average
over components:

* ``r2_site``: R^2 of site alone;
* ``r2_site_partial``: R^2(site + covariates) - R^2(covariates), the share that only
  site explains once age, sex and diagnosis are accounted for.

The diagnosis enters this *description* only; it is never a ComBat covariate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SiteEffect:
    """Site-effect summary of one feature matrix."""

    site_balanced_accuracy: float
    chance: float
    r2_site: float
    r2_site_partial: float
    per_component: pd.DataFrame
    embedding: NDArray[np.float64]  # first two principal components, one row per subject


def site_predictability(
    X: NDArray[np.float64],
    sites: NDArray[np.object_],
    n_components: int = 50,
    cv_folds: int = 5,
    seed: int = 0,
) -> float:
    """Cross-validated balanced accuracy of predicting the site from the features."""
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, svd_solver="randomized", random_state=seed),
        LogisticRegression(C=1.0, max_iter=2000),
    )
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, X, sites, cv=cv)
    return float(balanced_accuracy_score(sites, pred))


def _r2(y: NDArray[np.float64], design: NDArray[np.float64]) -> float:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return float(1.0 - resid.var() / y.var())


def variance_by_site(
    X: NDArray[np.float64],
    sites: NDArray[np.object_],
    covariates: pd.DataFrame,
    n_components: int = 10,
    seed: int = 0,
) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """R^2 of site (alone and beyond the covariates) for each leading component.

    Returns the per-component table and the subjects' scores on the first two components.
    """
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    scores = pca.fit_transform(Z)
    one = np.ones((len(Z), 1))
    site_d = pd.get_dummies(pd.Series(sites, dtype=str), drop_first=True).to_numpy(float)
    cov = covariates.astype(float).to_numpy()
    cov = (cov - cov.mean(axis=0)) / cov.std(axis=0)
    rows = []
    for j in range(n_components):
        y = scores[:, j]
        r2_site = _r2(y, np.c_[one, site_d])
        r2_cov = _r2(y, np.c_[one, cov])
        r2_full = _r2(y, np.c_[one, cov, site_d])
        rows.append(
            {
                "component": j + 1,
                "explained_variance_ratio": float(pca.explained_variance_ratio_[j]),
                "r2_site": r2_site,
                "r2_covariates": r2_cov,
                "r2_site_partial": r2_full - r2_cov,
            }
        )
    return pd.DataFrame(rows), scores[:, :2]


def describe_site_effect(
    X: NDArray[np.float64],
    sites: NDArray[np.object_],
    covariates: pd.DataFrame,
    n_components_predict: int = 50,
    n_components_variance: int = 10,
    seed: int = 0,
) -> SiteEffect:
    """Site predictability and variance explained by site for one feature matrix."""
    acc = site_predictability(X, sites, n_components=n_components_predict, seed=seed)
    comp, emb = variance_by_site(X, sites, covariates, n_components_variance, seed)
    w = comp["explained_variance_ratio"] / comp["explained_variance_ratio"].sum()
    return SiteEffect(
        site_balanced_accuracy=acc,
        chance=1.0 / len(np.unique(sites)),
        r2_site=float((w * comp["r2_site"]).sum()),
        r2_site_partial=float((w * comp["r2_site_partial"]).sum()),
        per_component=comp,
        embedding=emb,
    )


def effects_table(effects: dict[str, SiteEffect], n_subjects: int) -> pd.DataFrame:
    """One summary row per harmonisation setting."""
    return pd.DataFrame(
        [
            {
                "harmonization": name,
                "n_subjects": n_subjects,
                "n_sites": round(1.0 / e.chance),
                "site_balanced_accuracy": e.site_balanced_accuracy,
                "chance": e.chance,
                "r2_site": e.r2_site,
                "r2_site_partial": e.r2_site_partial,
            }
            for name, e in effects.items()
        ]
    )


def save_site_effect(run_dir: Path, effects: dict[str, SiteEffect], sites: NDArray) -> None:
    """Store everything Figure 2 and the table need in the run folder."""
    run_dir = Path(run_dir)
    effects_table(effects, len(sites)).to_csv(
        run_dir / "site_effect.csv", index=False, lineterminator="\n"
    )
    pd.concat(
        [e.per_component.assign(harmonization=n) for n, e in effects.items()], ignore_index=True
    ).to_csv(run_dir / "site_effect_components.csv", index=False, lineterminator="\n")
    pd.concat(
        [
            pd.DataFrame(
                {
                    "harmonization": n,
                    "site": sites,
                    "pc1": e.embedding[:, 0],
                    "pc2": e.embedding[:, 1],
                }
            )
            for n, e in effects.items()
        ],
        ignore_index=True,
    ).to_csv(run_dir / "site_effect_embedding.csv.gz", index=False, lineterminator="\n")


def load_site_effect(run_dir: Path) -> tuple[dict[str, SiteEffect], NDArray]:
    """Rebuild the :class:`SiteEffect` objects saved by :func:`save_site_effect`."""
    run_dir = Path(run_dir)
    tab = pd.read_csv(run_dir / "site_effect.csv").set_index("harmonization")
    comps = pd.read_csv(run_dir / "site_effect_components.csv")
    emb = pd.read_csv(run_dir / "site_effect_embedding.csv.gz", dtype={"site": str})
    effects, sites = {}, None
    for name, row in tab.iterrows():
        e = emb[emb["harmonization"] == name]
        sites = e["site"].to_numpy(dtype=object)
        effects[str(name)] = SiteEffect(
            site_balanced_accuracy=float(row["site_balanced_accuracy"]),
            chance=float(row["chance"]),
            r2_site=float(row["r2_site"]),
            r2_site_partial=float(row["r2_site_partial"]),
            per_component=comps[comps["harmonization"] == name]
            .drop(columns="harmonization")
            .reset_index(drop=True),
            embedding=e[["pc1", "pc2"]].to_numpy(),
        )
    return effects, sites


def write_site_effect_outputs(
    effects: dict[str, SiteEffect], sites: NDArray, run_name: str, reports: Path, tag: str = ""
) -> pd.DataFrame:
    """Write ``reports/tables/site_effect`` and Figure 2."""
    from shiftguard.utils.tables import write_table
    from shiftguard.viz.site_effect_figure import figure_site_effect

    reports = Path(reports)
    sfx = f"_{tag}" if tag else ""
    table = effects_table(effects, len(sites))
    write_table(
        table,
        reports / "tables" / f"site_effect{sfx}",
        title=(
            f"Site effect on the connectome (run {run_name}). Site balanced accuracy: "
            "5-fold CV (scaler, PCA 50, multinomial L2 logistic regression), chance = "
            "1 / sites. r2_site: R^2 of site dummies on the 10 leading principal components, "
            "weighted by explained variance; r2_site_partial: beyond age, sex and diagnosis. "
            "ComBat uses age and sex only."
        ),
    )
    figure_site_effect(
        effects["none"], effects["combat"], sites, reports / "figures" / f"fig2_site_effect{sfx}"
    )
    return table
