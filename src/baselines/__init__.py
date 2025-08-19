# src/baselines/__init__.py
"""
Baseline methods for FieldCert-Weather experiments.

This package provides:
- Raw deterministic thresholding.
- Global threshold bump tuned on calibration to meet area risk.
- Morphological CP-style shrinkage by erosion/opening.
- Pixelwise split-conformal (per-pixel threshold bump) baseline.
- EMOS (event-probability logistic) fit & evaluation utilities.
"""

from .raw_threshold import make_raw_masks, build_truth_masks
from .global_bump import calibrate_global_bump_by_fpa, apply_global_bump
from .morph_cp import calibrate_morph_radius_by_fpa, apply_morph_radius
from .pixel_cp import calibrate_pixelwise_delta, apply_pixelwise_delta
from .emos_fit import EMOSLogistic
from .emos_eval import (
    prob_to_set_calibrate_threshold,
    brier_score,
    reliability_curve,
    evaluate_prob_event_baseline,
)

__all__ = [
    # Raw
    "make_raw_masks",
    "build_truth_masks",
    # Global bump
    "calibrate_global_bump_by_fpa",
    "apply_global_bump",
    # Morph CP
    "calibrate_morph_radius_by_fpa",
    "apply_morph_radius",
    # Pixel CP
    "calibrate_pixelwise_delta",
    "apply_pixelwise_delta",
    # EMOS (event probability)
    "EMOSLogistic",
    "prob_to_set_calibrate_threshold",
    "brier_score",
    "reliability_curve",
    "evaluate_prob_event_baseline",
]
