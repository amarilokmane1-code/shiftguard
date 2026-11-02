"""Write notebooks/kaggle_shiftguard.ipynb (run: python notebooks/build_kaggle_notebook.py).

The notebook is generated from this script so that its cells stay reviewable as plain
Python and the JSON stays clean (no outputs, no execution counts).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/amarilokmane1-code/shiftguard"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        f"""# ShiftGuard on Kaggle: does conformal abstention survive site shift?

**Internet must be ON** (Settings → Internet → On). The notebook installs the package
from GitHub and downloads ABIDE I (about 0.4 GB) from the Preprocessed Connectomes
Project. No GPU is needed; the default pipeline is CPU-only.

Package: [{REPO}]({REPO}). Methods and the full results: `docs/METHODS.md` and
`docs/RESULTS.md` in the repository.

**Research software. Nothing here supports clinical use.** ABIDE data are subject to
their own terms of use (see `docs/DATA_CARD.md`); do not redistribute them, and do not
publish this notebook's `data/` folder.

### Expected runtime (Kaggle CPU, 4 cores)

Times measured on a 4-core laptop CPU (Windows) with `n_jobs = 4`; * = the step shared
the CPU with another run, so it is an upper bound; ~ = estimate, not measured. A Kaggle
CPU session should be of the same order.

| Step | Command | Time |
|---|---|---|
| Clone + install (CPU PyTorch, package, ComBat) | `git clone`, `pip install` | ~8 min |
| Download ABIDE (1035 files, 0.4 GB) | `shiftguard download` | ~7 min |
| QC + connectomes | `shiftguard features` | ~3 min |
| Synthetic validity checks | `shiftguard simulate` | < 1 min |
| Baselines: logistic regression + MLP, within-site + LOSO, 10 seeds (380 fits) | `shiftguard run` | ~2 h |
| ComBat + LOSO refit, logistic regression (180 fits) | `shiftguard run` (ComBat) | ~2 h* |
| Conformal prediction (domain classifiers ~20 min the first time, then cached) | `shiftguard conformal` | ~25 min |
| Site effect (Figure 2) | `shiftguard site-effect` | ~3 min |
| Label-free detection | `shiftguard detect` | ~3.5 min |
| Report | `shiftguard report` | ~2.5 min |

The full pipeline takes about **4.5–5 hours**, within Kaggle's 12-hour session limit.
The `--sensitivity` analyses (α = 0.2 and ROI-level exclusion) add about 2.5–3 hours*.
For a quick look, set `QUICK = True` below: 2 seeds instead of 10, roughly 1 hour in
total (~). The numbers will then differ from the README, which uses 10 seeds.""",
    ),
    (
        "code",
        f"""# Get the code (configs are part of the repository) and install it.
# CPU-only PyTorch first, so pip does not fetch the much larger CUDA build.
!git clone -q --depth 1 {REPO} /kaggle/working/shiftguard
%cd /kaggle/working/shiftguard
!pip install -q torch --index-url https://download.pytorch.org/whl/cpu
!pip install -q -e ".[harmonize]"
!shiftguard info""",
    ),
    (
        "code",
        """from pathlib import Path

QUICK = False        # True: 2 seeds instead of 10 (about 1 hour instead of 5)
SENSITIVITY = False  # True: also alpha = 0.2 and ROI-level exclusion (+2.5 hours)

# The Kaggle config inherits configs/experiment/loso.yaml and writes to /kaggle/working.
CONFIG = "configs/experiment/kaggle.yaml"
OVERRIDES = ["-s", "experiment.n_seeds=2"] if QUICK else []
REPORTS = Path("/kaggle/working/reports")""",
    ),
    (
        "markdown",
        """## Run the whole pipeline

`shiftguard all` runs download → features → synthetic checks → baselines → ComBat refit
→ conformal prediction → site effect → label-free detection, then writes the run
manifest (`reports/final_runs.yaml`) and `reports/RESULTS.md`. Each step writes a
time-stamped run folder under `reports/runs/` with its config, seeds, package versions
and git commit.""",
    ),
    (
        "code",
        """import subprocess

cmd = ["shiftguard", "all", "-c", CONFIG, *OVERRIDES, "--n-jobs", "4"]
if SENSITIVITY:
    cmd.append("--sensitivity")
print(" ".join(cmd))
subprocess.run(cmd, check=True)""",
    ),
    (
        "markdown",
        """## Results

The main table (logistic regression, α = 0.1): coverage pooled over sites, worst-site
coverage, worst coverage within one diagnosis at one site, and abstention.""",
    ),
    (
        "code",
        """import pandas as pd

head = pd.read_csv(REPORTS / "tables" / "conformal_headline.csv")
cols = ["protocol", "model", "method", "coverage", "worst_site_coverage",
        "worst_site_class_coverage", "coverage_gap", "abstention", "selective_accuracy"]
head[head["model"] == "logreg"][cols]""",
    ),
    (
        "code",
        """from IPython.display import Image, Markdown, display

FIGURES = [
    ("fig3_coverage_per_site", "Figure 3: coverage per deployment site"),
    ("fig3b_class_coverage", "Figure 3b: coverage per class and site"),
    ("fig2_site_effect", "Figure 2: site effect before / after ComBat"),
    ("fig4_target_labels", "Figure 4: labelled target subjects and abstention"),
    ("fig5_abstention_per_site", "Figure 5: abstention per site"),
    ("fig6_reliability", "Figure 6: calibration"),
    ("fig7_detection", "Figure 7: label-free signals vs coverage drop"),
    ("fig7b_detection_class_gap", "Figure 7b: which class loses coverage"),
]
for stem, title in FIGURES:
    path = REPORTS / "figures" / f"{stem}.png"
    if path.is_file():
        display(Markdown(f"### {title}"))
        display(Image(filename=str(path), width=900))""",
    ),
    (
        "code",
        """display(Markdown((REPORTS / "RESULTS.md").read_text(encoding="utf-8")))""",
    ),
    (
        "markdown",
        """## Notes

- Seeds are fixed (`experiment.base_seed`), so a rerun with 10 seeds reproduces the
  repository's tables, up to floating-point differences between platforms.
- The deployment site is never used for training, calibration or tuning, except in the
  explicit few-labelled-target (Mondrian) arm. ComBat and the domain classifier use the
  deployment site's **unlabelled** features only.
- Before sharing this notebook publicly, clear `/kaggle/working/data` from its outputs.
  ABIDE data must not be redistributed.""",
    ),
]


def build(path: Path) -> None:
    """Write the notebook JSON (nbformat 4, no outputs)."""
    nb = {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": text.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, text in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")  # LF on every platform


if __name__ == "__main__":
    build(Path(__file__).with_name("kaggle_shiftguard.ipynb"))
