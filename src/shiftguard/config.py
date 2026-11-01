"""YAML configuration loading with explicit inheritance and dotted overrides.

Every experiment is fully described by one YAML file. A file may list parent files under
the ``inherit`` key (paths relative to the file itself); parents are merged first, in
order, and the child's values win. Nested mappings are merged recursively, everything
else (lists, scalars) is replaced.

Example
-------
``configs/experiment/loso.yaml``::

    inherit:
      - ../data/abide_cc200.yaml
      - ../model/logreg.yaml
    experiment:
      name: loso
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

INHERIT_KEY = "inherit"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Parameters
    ----------
    base
        The mapping providing default values.
    override
        The mapping whose values take precedence. Nested mappings are merged; any other
        value type replaces the corresponding value in ``base``.

    Returns
    -------
    dict
        A new dictionary; neither input is modified.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_recursive(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    """Load ``path`` and its parents, detecting inheritance cycles."""
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"Cyclic config inheritance: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Top level of {path} must be a mapping, got {type(raw).__name__}")

    parents = raw.pop(INHERIT_KEY, [])
    if isinstance(parents, str):
        parents = [parents]

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = (path.parent / parent).resolve()
        merged = deep_merge(merged, _load_recursive(parent_path, (*stack, path)))
    return deep_merge(merged, raw)


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving ``inherit`` chains and applying overrides.

    Parameters
    ----------
    path
        Path to the YAML file.
    overrides
        Optional ``"dotted.key=value"`` strings (see :func:`apply_overrides`).

    Returns
    -------
    dict
        The fully merged configuration. The resolved source path is stored under
        ``_meta.source``.
    """
    resolved = Path(path).expanduser().resolve()
    config = _load_recursive(resolved, stack=())
    if overrides:
        config = apply_overrides(config, overrides)
    config.setdefault("_meta", {})["source"] = str(resolved)
    return config


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``"a.b.c=value"`` overrides to a copy of ``config``.

    Values are parsed with YAML, so ``seeds=[0,1]`` gives a list, ``n=5`` an int and
    ``flag=false`` a boolean. Intermediate mappings are created when missing.

    Parameters
    ----------
    config
        The configuration to modify.
    overrides
        Override strings of the form ``dotted.key=value``.

    Returns
    -------
    dict
        A new configuration dictionary.
    """
    out = copy.deepcopy(dict(config))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like 'key=value', got {item!r}")
        dotted, raw_value = item.split("=", 1)
        keys = [k for k in dotted.strip().split(".") if k]
        if not keys:
            raise ValueError(f"Empty key in override {item!r}")
        node = out
        for key in keys[:-1]:
            child = node.get(key)
            if child is None:
                child = node[key] = {}
            elif not isinstance(child, dict):
                raise ValueError(f"Cannot set {dotted!r}: {key!r} is not a mapping")
            node = child
        node[keys[-1]] = yaml.safe_load(raw_value)
    return out


def get_in(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Return ``config[a][b][c]`` for ``dotted="a.b.c"``, or ``default`` if absent."""
    node: Any = config
    for key in dotted.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``config`` to ``path`` as YAML (key order preserved) and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    return out
