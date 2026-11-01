"""Small helpers for writing result tables (CSV + Markdown) without extra dependencies."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(value: Any, digits: int) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    """Render ``df`` as a GitHub-flavoured Markdown table."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = [
        "| " + " | ".join(_fmt(v, digits) for v in row) + " |" for row in df.itertuples(index=False)
    ]
    return "\n".join([header, rule, *rows]) + "\n"


def write_table(
    df: pd.DataFrame,
    stem: str | Path,
    *,
    markdown_columns: Sequence[str] | None = None,
    digits: int = 3,
    title: str | None = None,
) -> tuple[Path, Path]:
    """Write ``<stem>.csv`` (all columns) and ``<stem>.md`` (selected columns).

    Returns
    -------
    tuple of pathlib.Path
        The CSV and Markdown paths.
    """
    base = Path(stem)
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = base.with_suffix(".csv"), base.with_suffix(".md")
    # Force LF so generated tables are byte-identical on Windows and Linux.
    df.to_csv(csv_path, index=False, lineterminator="\n")
    shown = df[list(markdown_columns)] if markdown_columns else df
    text = (f"# {title}\n\n" if title else "") + to_markdown(shown, digits)
    md_path.write_text(text, encoding="utf-8", newline="\n")
    return csv_path, md_path
