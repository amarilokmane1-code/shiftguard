import json
import logging
import shutil
import subprocess
from datetime import datetime

import numpy as np
import pytest
import yaml

from shiftguard.utils.log import get_logger, setup_logging
from shiftguard.utils.paths import project_root, resolve_path
from shiftguard.utils.provenance import (
    create_run_dir,
    git_info,
    package_versions,
    write_run_metadata,
)
from shiftguard.utils.seeding import derive_seed, set_global_seed


def test_set_global_seed_is_reproducible():
    a = set_global_seed(123).normal(size=5)
    b = set_global_seed(123).normal(size=5)
    np.testing.assert_array_equal(a, b)


def test_derive_seed_is_stable_and_distinct():
    s1 = derive_seed(0, "NYU", 3)
    assert s1 == derive_seed(0, "NYU", 3)
    assert s1 != derive_seed(0, "UCLA_1", 3)
    assert s1 != derive_seed(1, "NYU", 3)
    assert 0 <= s1 < 2**32


def test_create_run_dir_never_overwrites(tmp_path):
    now = datetime(2026, 1, 2, 3, 4, 5)
    first = create_run_dir(tmp_path, "loso test/1", now=now)
    second = create_run_dir(tmp_path, "loso test/1", now=now)
    assert first.name == "2026-01-02_030405_loso-test-1"
    assert second != first
    assert first.parent == tmp_path / "runs"


def test_write_run_metadata(tmp_path):
    cfg = {"experiment": {"name": "x"}}
    meta_path = write_run_metadata(tmp_path, cfg, seeds=[0, 1], extra={"command": "run"})
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["seeds"] == [0, 1]
    assert meta["command"] == "run"
    assert {"commit", "dirty"} <= set(meta["git"])
    assert meta["packages"]["numpy"] == np.__version__
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved == cfg


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_info_separates_outputs_from_source(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "src").mkdir()
    (tmp_path / "reports" / "tables").mkdir(parents=True)
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "reports" / "tables" / "t.csv").write_text("a\n1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")

    clean = git_info(tmp_path)
    assert clean["dirty"] is False and clean["outputs_modified"] == []

    # A table rewritten by an earlier pipeline step is an output, not a code change.
    (tmp_path / "reports" / "tables" / "t.csv").write_text("a\n2\n")
    outputs_only = git_info(tmp_path)
    assert outputs_only["dirty"] is False
    assert outputs_only["outputs_modified"] == ["reports/tables/t.csv"]

    # Any change outside reports/ makes the run dirty, also when called from a subfolder.
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    assert git_info(tmp_path / "src")["dirty"] is True


def test_package_versions_handles_missing():
    versions = package_versions(["numpy", "definitely-not-a-real-package-xyz"])
    assert versions["definitely-not-a-real-package-xyz"] is None
    assert versions["numpy"] is not None


def test_logging_writes_timestamped_file(tmp_path):
    log_file = tmp_path / "run.log"
    setup_logging(logging.INFO, log_file=log_file)
    get_logger("tests").info("hello")
    for handler in logging.getLogger("shiftguard").handlers:
        handler.flush()
    line = log_file.read_text(encoding="utf-8").strip()
    assert line.endswith("hello")
    datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
    setup_logging(logging.INFO)  # detach the file handler


def test_project_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SHIFTGUARD_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()
    assert resolve_path("data") == tmp_path.resolve() / "data"


def test_project_root_finds_pyproject(tmp_path, monkeypatch):
    monkeypatch.delenv("SHIFTGUARD_ROOT", raising=False)
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert project_root(nested) == tmp_path.resolve()
