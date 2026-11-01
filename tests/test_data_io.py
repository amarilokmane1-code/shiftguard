import json

import numpy as np
import pandas as pd
import pytest

from abide_fixtures import make_phenotypes, write_fake_abide
from shiftguard.config import load_config
from shiftguard.data.dataset import build_dataset, derived_dir, load_dataset
from shiftguard.data.download import (
    AbideSpec,
    OfflineError,
    available_file_ids,
    download_abide,
    download_s3,
    ensure_online,
    fetch_phenotypes,
    missing_files,
    timeseries_path,
)
from shiftguard.data.timeseries import (
    load_timeseries,
    read_1d,
    read_timeseries_cache,
    save_timeseries,
    timeseries_info,
)

SPEC = AbideSpec()


# ----------------------------------------------------------------------------- download
def test_spec_paths_and_flags():
    assert SPEC.tag == "cpac_filt_noglobal_rois_cc200"
    assert SPEC.band_pass_filtering and not SPEC.global_signal_regression
    assert AbideSpec(strategy="nofilt_global").global_signal_regression
    assert SPEC.url("Pitt_0050003").endswith(
        "/Outputs/cpac/filt_noglobal/rois_cc200/Pitt_0050003_rois_cc200.1D"
    )
    for bad in ({"pipeline": "fsl"}, {"strategy": "global"}, {"derivative": "func_preproc"}):
        with pytest.raises(ValueError):
            AbideSpec(**bad)


def test_offline_guard_blocks_every_network_path(tmp_path):
    # conftest sets SHIFTGUARD_OFFLINE=1 for every test
    with pytest.raises(OfflineError):
        ensure_online()
    with pytest.raises(OfflineError):
        fetch_phenotypes(tmp_path)
    with pytest.raises(OfflineError):
        download_s3(tmp_path, SPEC, ["x"])


def test_download_uses_cache_without_network(tmp_path):
    pheno = make_phenotypes({"A": (2, 2)})
    pheno.loc[0, "FILE_ID"] = "no_filename"
    write_fake_abide(tmp_path, pheno, SPEC)
    rep = download_abide(tmp_path, SPEC)  # offline, but nothing is missing
    assert rep.method_used == "cache"
    assert rep.n_expected == rep.n_present == 3
    assert rep.bytes_on_disk > 0


def test_download_with_missing_files_is_blocked_offline(tmp_path):
    pheno = make_phenotypes({"A": (2, 2)})
    write_fake_abide(tmp_path, pheno, SPEC, skip=(pheno["FILE_ID"].iloc[1],))
    assert missing_files(
        tmp_path,
        SPEC,
        available_file_ids(tmp_path / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv"),
    ) == [pheno["FILE_ID"].iloc[1]]
    with pytest.raises(OfflineError):
        download_abide(tmp_path, SPEC)


def test_auto_falls_back_to_s3_when_nilearn_fails(tmp_path, monkeypatch):
    # Reproduces the real run: nilearn aborts (read timeout), the HTTPS fallback finishes.
    import shiftguard.data.download as dl

    pheno = make_phenotypes({"A": (2, 2)})
    write_fake_abide(tmp_path, pheno, SPEC, skip=tuple(pheno["FILE_ID"].iloc[1:]))
    monkeypatch.delenv("SHIFTGUARD_OFFLINE")

    def nilearn_times_out(*args, **kwargs):
        raise TimeoutError("Read timed out")

    fetched = []

    def fake_get(url, dest, **kwargs):
        assert url.startswith(dl.S3_ROOT) and url.endswith(dest.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("#1\t#2\n1\t2\n3\t4\n", encoding="utf-8")
        fetched.append(dest.name)
        return dest.stat().st_size

    monkeypatch.setattr(dl, "download_nilearn", nilearn_times_out)
    monkeypatch.setattr(dl, "_http_get", fake_get)
    rep = download_abide(tmp_path, SPEC, n_jobs=2)
    assert rep.method_used == "s3"
    assert rep.n_present == rep.n_expected == 4 and not rep.missing
    assert len(fetched) == 3  # only the missing files

    with pytest.raises(TimeoutError):  # method="nilearn" means no fallback
        timeseries_path(tmp_path, SPEC, pheno["FILE_ID"].iloc[1]).unlink()
        download_abide(tmp_path, SPEC, method="nilearn")


def test_http_get_retries_then_succeeds_or_reports(tmp_path, monkeypatch):
    import requests

    import shiftguard.data.download as dl

    calls = {"n": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"1\t2\n"

    def flaky_get(url, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("boom")
        return FakeResponse()

    monkeypatch.setattr(requests, "get", flaky_get)
    dest = tmp_path / "sub" / "x.1D"
    assert dl._http_get("https://example.invalid/x.1D", dest, retries=3) == 4
    assert dest.read_bytes() == b"1\t2\n"
    assert not dest.with_name("x.1D.part").exists()

    calls["n"] = -10  # always fails within 2 retries
    with pytest.raises(RuntimeError, match="failed to download"):
        dl._http_get("https://example.invalid/y.1D", tmp_path / "y.1D", retries=2)
    assert not (tmp_path / "y.1D").exists()


def test_download_s3_reports_failures(tmp_path, monkeypatch):
    import shiftguard.data.download as dl

    monkeypatch.delenv("SHIFTGUARD_OFFLINE")

    def get(url, dest, **kwargs):
        if "BAD" in url:
            raise RuntimeError(f"failed to download {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"1\n")
        return 2

    monkeypatch.setattr(dl, "_http_get", get)
    assert download_s3(tmp_path, SPEC, ["OK_1", "BAD_2", "OK_3"], n_jobs=2) == ["BAD_2"]


# ----------------------------------------------------------------------------- time series
def test_read_1d_skips_header_and_load_handles_missing(tmp_path):
    pheno = make_phenotypes({"A": (1, 1)})
    write_fake_abide(tmp_path, pheno, SPEC, n_timepoints=30, n_rois=5)
    fid_ok, fid_missing = pheno["FILE_ID"]
    timeseries_path(tmp_path, SPEC, fid_missing).unlink()
    assert read_1d(timeseries_path(tmp_path, SPEC, fid_ok)).shape == (30, 5)
    ts = load_timeseries(tmp_path, SPEC, [fid_ok, fid_missing])
    assert ts[fid_missing] is None


def test_empty_file_is_unreadable(tmp_path):
    path = tmp_path / "x.1D"
    path.write_text("#1\t#2\n", encoding="utf-8")
    with pytest.raises((ValueError, UserWarning)):
        read_1d(path)


def test_timeseries_info_and_cache(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 4))
    a[:, 2] = 3.0  # constant ROI
    b = rng.normal(size=(40, 4))
    b[0, 0] = np.nan
    info = timeseries_info({"a": a, "b": b, "c": None}).set_index("FILE_ID")
    assert info.loc["a", "n_zero_variance_rois"] == 1
    assert bool(info.loc["b", "has_nonfinite"])
    assert not bool(info.loc["c", "readable"])
    path = save_timeseries(tmp_path / "ts.npz", {"a": a, "z": b})
    back = read_timeseries_cache(path, ["z", "a"])
    assert list(back) == ["z", "a"]
    np.testing.assert_allclose(back["a"], a, atol=1e-5)  # stored as float32


# ----------------------------------------------------------------------------- build stage
@pytest.fixture
def fake_config(tmp_path):
    cfg = load_config("configs/experiment/loso.yaml")
    cfg["paths"] = {"data_dir": str(tmp_path / "data"), "reports_dir": str(tmp_path / "reports")}
    cfg["qc"]["min_subjects_per_class_per_site"] = 3
    return cfg


def test_build_and_load_dataset_end_to_end(fake_config, tmp_path):
    pheno = make_phenotypes({"S1": (4, 4), "S2": (5, 3), "TINY": (2, 1)})
    pheno.loc[0, "qc_rater_1"] = "fail"
    write_fake_abide(tmp_path / "data", pheno, SPEC, n_rois=6)

    qc = build_dataset(fake_config)
    assert qc.small_sites == ["TINY"]
    assert qc.reason_counts["rater_qc_fail"] == 1
    reports = tmp_path / "reports"
    summary = pd.read_csv(reports / "qc_summary.csv")
    assert summary.set_index("SITE_ID").loc["TOTAL", "n_after"] == len(qc.included) == 15
    assert (reports / "figures" / "fig1_data_map.png").stat().st_size > 0
    assert (reports / "figures" / "fig1_data_map.pdf").stat().st_size > 0
    assert json.loads((derived_dir(fake_config) / "qc_config.json").read_text())["small_sites"] == [
        "TINY"
    ]

    ds = load_dataset(fake_config)
    assert ds.features.shape == (15, 15)
    assert len(ds.time_series) == 15
    assert ds.labels.tolist() == qc.included["label"].tolist()
    assert set(ds.sites) == {"S1", "S2"}


def test_build_requires_download(fake_config):
    with pytest.raises(FileNotFoundError, match="download"):
        build_dataset(fake_config)
    with pytest.raises(FileNotFoundError, match="features"):
        load_dataset(fake_config)
