"""Reproducible seeding of every random number generator in use."""

from __future__ import annotations

import os
import random
import sys

import numpy as np


def set_global_seed(seed: int, *, torch_deterministic: bool = True) -> np.random.Generator:
    """Seed Python, NumPy and (if already imported) PyTorch.

    Code in this package draws randomness from explicit ``numpy.random.Generator``
    objects; the global seeds are set as a safety net for third-party code.

    Parameters
    ----------
    seed
        Non-negative integer seed.
    torch_deterministic
        If PyTorch is loaded, also request deterministic algorithms.

    Returns
    -------
    numpy.random.Generator
        A generator seeded with ``seed``.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - deliberate: seeds legacy global state for libraries
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch_deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.default_rng(seed)


def derive_seed(base_seed: int, *keys: int | str) -> int:
    """Derive a stable child seed from a base seed and identifying keys.

    Used to give each (seed, site, method) combination its own independent stream while
    keeping the whole experiment reproducible from one number.

    Parameters
    ----------
    base_seed
        The experiment-level seed.
    *keys
        Identifiers such as a site name or a fold index.

    Returns
    -------
    int
        A 32-bit seed.
    """
    entropy = [base_seed, *[_key_to_int(k) for k in keys]]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def _key_to_int(key: int | str) -> int:
    """Map a key to a non-negative integer deterministically (independent of hash seed)."""
    if isinstance(key, int):
        if key < 0:
            raise ValueError("integer keys must be non-negative")
        return key
    return int.from_bytes(key.encode("utf-8"), "little") % (2**63)
