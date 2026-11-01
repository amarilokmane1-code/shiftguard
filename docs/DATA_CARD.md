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

## Variables kept

`SUB_ID`, `FILE_ID`, `SITE_ID`, `DX_GROUP` (1 = ASD, 2 = control), `AGE_AT_SCAN`, `SEX`
(1 = male, 2 = female), `FIQ` (where available), and the PCP manual QC fields.

## Quality control

Configured in `configs/data/abide_cc200.yaml` (`qc` section). Every exclusion is logged,
with counts per site, in `reports/qc_summary.csv`. Final counts: **TBD (phase 3)**.

## Known biases and composition

- Strong male majority (ASD is diagnosed more often in males, and ABIDE I reflects this).
- Mostly older children, adolescents and adults. Few young children.
- Mostly cognitively able (high-functioning) participants. Low-IQ autistic people are
  under-represented.
- Sites differ in scanner, protocol, eyes-open/closed instructions, age range and
  diagnostic procedure. This *site effect* is the subject of the study, not a nuisance
  to hide.

Per-site and per-group counts: **TBD (phase 3)**.
