"""Shared pytest configuration.

Tests never download ABIDE: ``SHIFTGUARD_OFFLINE=1`` makes every downloader refuse to
touch the network, so a test that accidentally triggers a download fails immediately.
"""

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("SHIFTGUARD_OFFLINE", "1")
