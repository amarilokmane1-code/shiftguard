"""ShiftGuard: does conformal abstention survive site shift.

A multi-site autism neuroimaging benchmark (ABIDE I) that tests whether the coverage
guarantee of conformal prediction holds when a classifier is deployed on an unseen
acquisition site, and which corrections protect it.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("shiftguard")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
