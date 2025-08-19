# src/calibration/__init__.py
"""
Calibration toolkit for FieldCert-Weather.

This package provides:
- Conformal Risk Control (CRC) calibrator for set-valued maps (CRCCalibrator).
- Physics-aware margin field builders.
- Symmetric morphology filters and threshold-bump set constructors.
- Field-level loss functions (FPA/FNA) and functional errors.
- Regime/block utility helpers and KNN block selection.
"""

from .crc import CRCResult, CRCSettings, CRCCalibrator
from .scores import (
    fpa_loss,
    fna_loss,
    combined_set_loss,
    functional_abs_error,
    jaccard_index,
)
from .margins import (
    build_margin_field,
    gradient_magnitude,
    divergence_plus,
    robust_minmax_scale,
)
from .morphology import (
    threshold_bump_mask,
    morphological_filter,
    symmetric_filter,
)
from .regimes import (
    assign_season,
    assign_blocks,
    make_regime_keys,
    group_indices_by_regime,
    KNNSelector,
)

__all__ = [
    # CRC
    "CRCResult",
    "CRCSettings",
    "CRCCalibrator",
    # Scores
    "fpa_loss",
    "fna_loss",
    "combined_set_loss",
    "functional_abs_error",
    "jaccard_index",
    # Margins
    "build_margin_field",
    "gradient_magnitude",
    "divergence_plus",
    "robust_minmax_scale",
    # Morphology
    "threshold_bump_mask",
    "morphological_filter",
    "symmetric_filter",
    # Regimes
    "assign_season",
    "assign_blocks",
    "make_regime_keys",
    "group_indices_by_regime",
    "KNNSelector",
]
