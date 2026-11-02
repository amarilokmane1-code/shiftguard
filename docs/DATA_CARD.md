# Data card: ABIDE I Preprocessed

## Source

- **Dataset:** Autism Brain Imaging Data Exchange I (ABIDE I), a retrospective collection
  of resting-state fMRI, anatomical and phenotypic data from 17 international sites.
- **Derivatives used:** the Preprocessed Connectomes Project (PCP) release of ABIDE I:
  ROI-averaged time series. Defaults: pipeline `cpac`, strategy `filt_noglobal`
  (band-pass filtering, no global signal regression), atlas `rois_cc200`
  (Craddock 200 parcellation). All three are configurable in `configs/data/`.
- **Access:** `nilearn.datasets.fetch_abide_pcp`, with a fallback that downloads the same
  files over HTTPS from the public PCP S3 bucket.

## Terms of use

ABIDE data are released by the International Neuroimaging Data-sharing Initiative (INDI)
for non-commercial research under their own data usage agreement. Users must read and
accept the terms on the ABIDE and PCP websites before using the data.

- This repository **does not contain and does not redistribute** any ABIDE data.
  `data/` is git-ignored, and per-subject run outputs (`reports/runs/`) are git-ignored too.
- Only aggregate results (site-level tables and figures) are committed.
- Do not attempt to re-identify participants.

## Required citations

If you use these data, cite:

- Di Martino A, et al. *The autism brain imaging data exchange: towards a large-scale
  evaluation of the intrinsic brain architecture in autism.* Molecular Psychiatry,
  19(6):659–667, 2014.
- Craddock C, et al. *The Neuro Bureau Preprocessing Initiative: open sharing of
  preprocessed neuroimaging data and derivatives.* Frontiers in Neuroinformatics,
  Conference Abstract: Neuroinformatics 2013.

## Download (as run on 2026-11-01)

- The phenotypic file lists **1112 subjects** under **20 PCP site IDs** (some of the 17
  ABIDE I institutions contributed separate cohorts, e.g. `UM_1`/`UM_2`, `UCLA_1`/`UCLA_2`,
  `LEUVEN_1`/`LEUVEN_2`; each ID is treated as its own site). 77 have no preprocessed data.
- All **1035** available CC200 time-series files were downloaded: **406.5 MB** of raw data.
  The derived caches in `data/derived/` (QC'd time series and correlation features) take
  another 205.2 MB. nilearn's `fetch_abide_pcp` fetched the first 3 files, then stopped on a
  network read timeout (it does not retry). The HTTPS fallback, which retries, fetched the
  remaining 1032 into the same layout.

## Variables kept

`SUB_ID`, `FILE_ID`, `SITE_ID`, `DX_GROUP` (1 = ASD, 2 = control), `AGE_AT_SCAN`, `SEX`
(1 = male, 2 = female), `FIQ` (where available; `-9999` is treated as missing),
`EYE_STATUS_AT_SCAN`, `func_mean_fd`, and the five PCP manual QC fields. In code, `label = 1`
means ASD and `label = 0` means control.

## Quality control

Configured in `configs/data/abide_cc200.yaml` (`qc` section) and implemented in
`src/shiftguard/data/qc.py`. Each excluded subject is attributed to the **first** reason
that applies, in this order, so the reason counts add up:

| # | Reason | Rule | Excluded |
|---|---|---|---|
| 1 | `no_preprocessed_data` | PCP provides no file (`FILE_ID = no_filename`) | 77 |
| 2 | `rater_qc_fail` | rater 1 marked the scan `fail` (rule `rater_1_pass`) | 34 |
| 3 | `missing_covariates` | age or sex missing (needed for ComBat) | 0 |
| 4 | `missing_timeseries` | file absent or unreadable | 0 |
| 5 | `too_few_timepoints` | fewer than 50 volumes | 0 |
| 6 | `bad_rois` | any constant (zero-variance) ROI, or non-finite values | 26 |
| 7 | `small_site` | site left with fewer than 10 subjects in either class | 31 (CALTECH, CMU) |
| | **Kept** | | **944 of 1112** (460 ASD, 484 TDC), **18 sites** |

Rater 1 marked 111 subjects `fail`, but 77 of those exclusions are counted under reason 1
because those subjects have no file. Per-site counts, before and after QC, are in
[`reports/qc_summary.md`](../reports/qc_summary.md) (and `.csv`). Subject-level exclusions
are in `data/derived/<tag>/exclusions.csv`, which is git-ignored.

**Zero-variance ROIs (decision, 2026-11-01).** A constant ROI (outside the subject's field
of view) has an undefined correlation, and Ledoit-Wolf shrinkage would silently turn it
into exact zeros. **Constant ROIs are never filled with zeros.** Such zeros would cluster
at the sites with limited coverage and create an artificial site signal, which would bias
a study whose subject is the site effect.

Two leakage-free options remain: exclude the affected *subjects*, or exclude the affected
*ROIs* for every subject (the same feature set everywhere). The pre-specified decision rule
was to use ROI-level exclusion if the 26 affected subjects together have 5 or fewer
distinct constant ROIs, and the strict subject rule otherwise. They have **121** distinct
constant ROIs (28 if the one broken CALTECH scan with 118 is set aside, and 13 for the CMU
subjects alone). So the **main analysis uses the strict subject rule**
(`max_zero_variance_rois: 0`): 944 subjects, 18 sites, all 200 CC200 ROIs. This costs the
CMU site: 9 of its 27 subjects are excluded, and CMU then fails the minimum-size rule.
ROI-level exclusion (which would keep CMU) was run as a **sensitivity analysis**
(`configs/experiment/loso_droprois.yaml`). It drops the 25 ROIs that are constant in any
kept subject and keeps 979 subjects from 19 sites (CMU: 23 subjects), with 175 ROIs
(15,225 edges); see [cohort_summary_droprois.md](../reports/tables/cohort_summary_droprois.md).
LOSO logistic-regression AUC is 0.725 (main cohort: 0.731). Split CP covers 0.895 (main:
0.898), and CMU is covered at 0.930. Every phase 5 conclusion holds
([conformal_headline_droprois.md](../reports/tables/conformal_headline_droprois.md)).
CALTECH is lost under either option (15 rater-1 failures).

Alternative rater rules (`all_raters_pass`, `nilearn_quality_checked` = nilearn's
`quality_checked=True`, `none`) are one config line away.

## Known biases and composition (after QC)

Generated per site in [`reports/tables/cohort_summary.md`](../reports/tables/cohort_summary.md).

- **Strong male majority:** 85.1% male overall. Five sites (LEUVEN_1, OHSU, SBL, TRINITY,
  USM) are 100% male.
- **Age:** median 14.2 years, range 6.5–64. Few young children. The sites barely overlap in
  age: KKI and STANFORD are children (medians 10.1 and 9.5), while LEUVEN_1, MAX_MUN and SBL
  are adults (medians 22.0, 26.0 and 34.5).
- **Mostly cognitively able participants:** median FIQ 108 (missing for 62 subjects, all
  of LEUVEN_2 and 23 of 28 at SBL). Low-IQ autistic people are under-represented.
- **Case/control ratio varies by site,** from 37.9% ASD (SDSU) to 64.8% ASD (USM). So a
  "site shift" can include label (prior) shift. Phase 5 measured how much of each site's
  miscalibration this explains: about a third of the summed squared offsets. Only USM
  (and UM_2) look like a prior shift; the large offsets at KKI, LEUVEN_1 and PITT do not
  ([prevalence_shift.md](../reports/tables/prevalence_shift.md)).
- Sites also differ in scanner, protocol, eyes-open/closed instructions and diagnostic
  procedure. This *site effect* is what the study is about, not a nuisance to hide.

## Site effect in the features

Described in [site_effect.md](../reports/tables/site_effect.md) and
[Figure 2](../reports/figures/fig2_site_effect.png). On the 944 QC'd subjects:

- The acquisition site can be predicted from the connectome with 5-fold balanced
  accuracy 0.55, against a chance level of 0.06.
- Site explains 13% of the variance of the 10 leading principal components (12% beyond
  age, sex and diagnosis).
- After ComBat (age and sex preserved, diagnosis not used), both are at chance level:
  balanced accuracy 0.05 and R² 0.005.

## Use in this project

- Only the derived ROI time series and phenotypes are used. No images are downloaded.
- Subject-level data, features and predictions stay in `data/` and `reports/runs/`,
  which are git-ignored. Only site-level aggregates are committed.
- The data support a methods benchmark, not a clinical model (see
  [MODEL_CARD.md](MODEL_CARD.md)).
