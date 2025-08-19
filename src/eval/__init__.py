# src/eval/__init__.py
"""
Evaluation utilities for FieldCert-Weather.

Modules:
- sets: set-level metrics (FPA/FNA/IoU), geometry, bootstrap CIs, functional coverage.
- prob: probabilistic metrics (Brier, reliability), CRPS for ensembles.
- wb2: WeatherBench-2 convenience I/O (open zarr, lead selection, alignment).
"""

from .sets import (
    confusion_areas,
    set_area,
    boundary_length,
    compactness,
    evaluate_set_masks,
    block_bootstrap_ci,
    evaluate_functional_intervals,
)

from .prob import (
    brier_score,
    brier_skill_score,
    reliability_curve,
    crps_ensemble,
)

from .wb2 import (
    open_wb2_dataset,
    select_forecast_lead,
    align_forecast_obs,
    compute_area_weights,
    build_bbox_mask,
)

__all__ = [
    # sets
    "confusion_areas",
    "set_area",
    "boundary_length",
    "compactness",
    "evaluate_set_masks",
    "block_bootstrap_ci",
    "evaluate_functional_intervals",
    # prob
    "brier_score",
    "brier_skill_score",
    "reliability_curve",
    "crps_ensemble",
    # wb2
    "open_wb2_dataset",
    "select_forecast_lead",
    "align_forecast_obs",
    "compute_area_weights",
    "build_bbox_mask",
]
