"""Probabilistic classifiers (logistic regression, MLP, optional GNN) and calibration."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator

MODEL_NAMES: tuple[str, ...] = ("logreg", "mlp")


def make_model(name: str, models_cfg: dict[str, Any], seed: int) -> BaseEstimator:
    """Build an unfitted model from the ``models`` config section.

    Parameters
    ----------
    name
        ``"logreg"`` or ``"mlp"``.
    models_cfg
        The ``models`` section of the experiment config.
    seed
        Random seed for the model.
    """
    cfg = models_cfg[name]
    if name == "logreg":
        from shiftguard.models.logreg import NestedLogisticRegression

        return NestedLogisticRegression(
            Cs=tuple(cfg["grid"]["C"]),
            penalty=cfg.get("penalty", "l2"),
            l1_ratios=tuple(cfg["grid"].get("l1_ratio", (0.5,))),
            pca_components=cfg.get("pca_components"),
            standardize=cfg.get("standardize", True),
            inner_cv_folds=cfg.get("inner_cv_folds", 5),
            max_iter=cfg.get("max_iter", 5000),
            random_state=seed,
        )
    if name == "mlp":
        from shiftguard.models.mlp import TorchMLPClassifier

        return TorchMLPClassifier(
            hidden_sizes=tuple(cfg["hidden_sizes"]),
            dropout=cfg["dropout"],
            weight_decay=cfg["weight_decay"],
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            max_epochs=cfg["max_epochs"],
            early_stopping_patience=cfg["early_stopping_patience"],
            inner_val_fraction=cfg["inner_val_fraction"],
            standardize=cfg.get("standardize", True),
            device=cfg.get("device", "cpu"),
            random_state=seed,
        )
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
