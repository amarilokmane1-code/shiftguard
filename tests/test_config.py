from pathlib import Path

import pytest

from shiftguard.config import apply_overrides, deep_merge, get_in, load_config, save_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_deep_merge_is_recursive_and_does_not_mutate():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2]}
    override = {"a": {"y": 3}, "b": [9]}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 3}, "b": [9]}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1, 2]}


def test_inheritance_child_wins(tmp_path):
    _write(tmp_path / "parent.yaml", "model:\n  C: 1.0\n  name: logreg\n")
    child = _write(tmp_path / "sub" / "child.yaml", "inherit: ../parent.yaml\nmodel:\n  C: 0.1\n")
    cfg = load_config(child)
    assert cfg["model"] == {"C": 0.1, "name": "logreg"}
    assert "inherit" not in cfg
    assert cfg["_meta"]["source"] == str(child.resolve())


def test_multiple_parents_merge_in_order(tmp_path):
    _write(tmp_path / "a.yaml", "k: a\nonly_a: 1\n")
    _write(tmp_path / "b.yaml", "k: b\n")
    child = _write(tmp_path / "c.yaml", "inherit: [a.yaml, b.yaml]\n")
    cfg = load_config(child)
    assert cfg["k"] == "b"
    assert cfg["only_a"] == 1


def test_cycle_is_detected(tmp_path):
    _write(tmp_path / "a.yaml", "inherit: b.yaml\n")
    _write(tmp_path / "b.yaml", "inherit: a.yaml\n")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(tmp_path / "a.yaml")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_overrides_parse_yaml_values():
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=5", "a.c=[1, 2]", "flag=false", "x.y.z=0.1"])
    assert cfg == {"a": {"b": 5, "c": [1, 2]}, "flag": False, "x": {"y": {"z": 0.1}}}


@pytest.mark.parametrize("bad", ["no_equals", "=3"])
def test_bad_overrides_raise(bad):
    with pytest.raises(ValueError):
        apply_overrides({}, [bad])


def test_override_through_scalar_raises():
    with pytest.raises(ValueError, match="not a mapping"):
        apply_overrides({"a": 1}, ["a.b=2"])


def test_get_in():
    cfg = {"a": {"b": {"c": 3}}}
    assert get_in(cfg, "a.b.c") == 3
    assert get_in(cfg, "a.x", default="d") == "d"


def test_save_roundtrip(tmp_path):
    cfg = {"experiment": {"name": "t", "seeds": [0, 1]}}
    path = save_config(cfg, tmp_path / "out" / "cfg.yaml")
    assert load_config(path)["experiment"] == cfg["experiment"]


@pytest.mark.parametrize("name", ["loso.yaml", "within_site.yaml", "kaggle.yaml", "synthetic.yaml"])
def test_shipped_experiment_configs_resolve(name):
    cfg = load_config(CONFIG_DIR / "experiment" / name)
    for section in ("experiment", "model", "conformal", "paths"):
        assert section in cfg, f"{name} lacks section {section!r}"
    assert 0 < cfg["conformal"]["alpha"] < 1
    assert cfg["experiment"]["protocol"] in {"loso", "within_site"}
    # Diagnosis must never be a harmonisation covariate (it would leak the label).
    assert "DX_GROUP" not in cfg["harmonization"]["covariates"]


def test_kaggle_config_inherits_loso():
    loso = load_config(CONFIG_DIR / "experiment" / "loso.yaml")
    kaggle = load_config(CONFIG_DIR / "experiment" / "kaggle.yaml")
    assert kaggle["experiment"]["protocol"] == loso["experiment"]["protocol"]
    assert kaggle["data"] == loso["data"]
    assert kaggle["paths"]["data_dir"].startswith("/kaggle/")
