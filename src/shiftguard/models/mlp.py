"""Small multilayer perceptron (PyTorch) with dropout and early stopping.

Architecture and optimiser settings are fixed by the config (not tuned) to keep the
pipeline CPU-friendly. The only data-driven choice is the stopping epoch, chosen on an
inner, stratified validation split carved out of the training data passed to ``fit``;
the weights with the lowest validation loss are restored.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The MLP needs PyTorch: pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu"
        ) from exc
    return torch


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Binary MLP classifier with a scikit-learn interface.

    Parameters
    ----------
    hidden_sizes
        Width of each hidden layer (ReLU, then dropout).
    dropout
        Dropout probability after each hidden layer.
    weight_decay
        L2 penalty (Adam ``weight_decay``).
    learning_rate
        Adam learning rate.
    batch_size
        Mini-batch size.
    max_epochs
        Upper bound on training epochs.
    early_stopping_patience
        Stop after this many epochs without validation improvement.
    inner_val_fraction
        Fraction of the training data held out for early stopping.
    standardize
        Z-score features (fitted on the training data passed to ``fit``).
    device
        Torch device.
    random_state
        Seed (weights, dropout, batching and the inner split).
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (256, 64),
        dropout: float = 0.5,
        weight_decay: float = 1e-3,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 200,
        early_stopping_patience: int = 20,
        inner_val_fraction: float = 0.15,
        standardize: bool = True,
        device: str = "cpu",
        random_state: int | None = None,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.inner_val_fraction = inner_val_fraction
        self.standardize = standardize
        self.device = device
        self.random_state = random_state

    def _build(self, n_features: int):
        torch = _require_torch()
        layers: list[object] = []
        width = n_features
        for h in self.hidden_sizes:
            layers += [torch.nn.Linear(width, h), torch.nn.ReLU(), torch.nn.Dropout(self.dropout)]
            width = h
        layers.append(torch.nn.Linear(width, 1))
        return torch.nn.Sequential(*layers).to(self.device)

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Train with early stopping on an inner validation split."""
        torch = _require_torch()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.float32)
        seed = 0 if self.random_state is None else int(self.random_state)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        self.scaler_ = StandardScaler().fit(X) if self.standardize else None
        Xs = self.scaler_.transform(X).astype(np.float32) if self.scaler_ else X
        idx_tr, idx_va = train_test_split(
            np.arange(len(y)), test_size=self.inner_val_fraction, stratify=y, random_state=seed
        )
        Xt = torch.from_numpy(Xs).to(self.device)
        yt = torch.from_numpy(y).to(self.device)

        model = self._build(Xs.shape[1])
        opt = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best, best_state, bad_epochs, n_epochs = np.inf, None, 0, 0
        for epoch in range(1, self.max_epochs + 1):
            n_epochs = epoch
            model.train()
            perm = rng.permutation(idx_tr)
            for start in range(0, len(perm), self.batch_size):
                batch = perm[start : start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(Xt[batch]).squeeze(1), yt[batch])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = float(loss_fn(model(Xt[idx_va]).squeeze(1), yt[idx_va]))
            if val < best - 1e-6:
                best, bad_epochs = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.early_stopping_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.best_val_loss_ = best
        self.n_epochs_ = n_epochs
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Logit of the ASD class."""
        torch = _require_torch()
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler_.transform(X).astype(np.float32) if self.scaler_ else X
        with torch.no_grad():
            out = self.model_(torch.from_numpy(Xs).to(self.device)).squeeze(1)
        return out.cpu().numpy().astype(np.float64)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Class probabilities, shape ``(n, 2)``."""
        p1 = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.c_[1.0 - p1, p1]
