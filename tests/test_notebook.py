"""Phase 8: the Kaggle notebook is generated, valid, and only calls real CLI commands."""

import importlib.util
import json
import re
from pathlib import Path

import typer

from shiftguard.cli import app
from shiftguard.config import load_config

NB = Path("notebooks/kaggle_shiftguard.ipynb")
BUILDER = Path("notebooks/build_kaggle_notebook.py")


def _builder():
    spec = importlib.util.spec_from_file_location("build_kaggle_notebook", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _text(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def test_notebook_matches_its_generator(tmp_path):
    out = tmp_path / "nb.ipynb"
    _builder().build(out)
    assert out.read_bytes() == NB.read_bytes(), "rerun python notebooks/build_kaggle_notebook.py"


def test_notebook_is_valid_and_complete():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["cells"]
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    text = _text(nb)
    assert "Internet must be ON" in text
    assert "not support clinical use" in text or "Nothing here supports clinical use" in text
    assert not re.search(r"__[A-Z]+__", text), "unfilled runtime placeholder"


def test_notebook_calls_only_existing_commands_and_config():
    text = _text(json.loads(NB.read_text(encoding="utf-8")))
    commands = set(typer.main.get_group(app).commands)  # registered command names
    used = set(re.findall(r"shiftguard (\w[\w-]*)", text)) | set(
        re.findall(r'"shiftguard", "(\w[\w-]*)"', text)
    )
    used -= {"on", "is"}  # prose ("ShiftGuard on Kaggle")
    assert used and used <= commands, used - commands
    cfg_path = re.search(r'CONFIG = "([^"]+)"', text).group(1)
    cfg = load_config(cfg_path)
    assert cfg["paths"]["reports_dir"].startswith("/kaggle/working")
    assert cfg["experiment"]["n_seeds"] == 10  # same protocol as the README
