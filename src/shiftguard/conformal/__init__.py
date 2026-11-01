"""Conformal prediction implemented from scratch.

Nonconformity scores, split conformal, class-conditional, Mondrian (group-conditional)
and weighted conformal prediction for covariate shift.
"""

from shiftguard.conformal.mondrian import MondrianConformal
from shiftguard.conformal.quantiles import (
    conformal_quantile,
    min_calibration_size,
    weighted_conformal_quantile,
)
from shiftguard.conformal.scores import aps_score_matrix, lac_score_matrix, score_matrix
from shiftguard.conformal.sets import abstained, covered, decisions, set_sizes, summarize_sets
from shiftguard.conformal.split import ClassConditionalConformal, SplitConformal
from shiftguard.conformal.weighted import (
    WeightedConformal,
    clip_weights,
    effective_sample_size,
    weight_clip_bounds,
)

__all__ = [
    "ClassConditionalConformal",
    "MondrianConformal",
    "SplitConformal",
    "WeightedConformal",
    "abstained",
    "aps_score_matrix",
    "clip_weights",
    "conformal_quantile",
    "covered",
    "decisions",
    "effective_sample_size",
    "lac_score_matrix",
    "min_calibration_size",
    "score_matrix",
    "set_sizes",
    "summarize_sets",
    "weight_clip_bounds",
    "weighted_conformal_quantile",
]
