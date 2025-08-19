# src/baselines/morph_cp.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Literal

from calibration.scores import fpa_loss
from calibration.morphology import morphological_filter
from .raw_threshold import build_truth_masks


def _eroded_prediction(
    preds: np.ndarray, threshold: float, radius: int, element: Literal["disk", "square"] = "disk", iterations: int = 1
) -> np.ndarray:
    """
    Build predicted set via raw thresholding followed by binary erosion of given radius.
    """
    yhat = np.asarray(preds, dtype=np.float64)
    Tval = float(threshold)
    base = (yhat >= Tval)
    out = np.empty_like(base, dtype=bool)
    for t in range(base.shape[0]):
        out[t] = morphological_filter(
            base[t],
            operation="erode",
            radius=radius,
            element=element,
            iterations=iterations,
        )
    return out


def calibrate_morph_radius_by_fpa(
    preds_cal: np.ndarray,    # (T,H,W)
    truths_cal: np.ndarray,   # (T,H,W)
    threshold: float,
    alpha: float = 0.10,
    radii: Optional[np.ndarray] = None,
    element: Literal["disk", "square"] = "disk",
    iterations: int = 1,
    spatial_weights: Optional[np.ndarray] = None,  # (H,W)
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Calibrate an erosion radius r to achieve mean FPA ≤ α on calibration.

    Parameters
    ----------
    preds_cal, truths_cal : (T,H,W)
    threshold : float
    alpha : target FPA
    radii : optional 1D array of candidate radii
    element : structuring element
    iterations : morphology iterations
    spatial_weights : optional weights for FPA

    Returns
    -------
    r_star : selected radius
    risk_curve : FPA per candidate radius
    radii : grid used
    """
    yhat = np.asarray(preds_cal, dtype=np.float64)
    y = np.asarray(truths_cal, dtype=np.float64)
    if yhat.shape != y.shape or yhat.ndim != 3:
        raise ValueError("preds_cal and truths_cal must be (T,H,W) and equal shapes")
    truth_masks = build_truth_masks(y, float(threshold))

    if radii is None:
        radii = np.arange(0, 16, dtype=int)  # 0..15 px
    else:
        radii = np.asarray(radii)
        if radii.ndim != 1 or radii.size == 0:
            raise ValueError("radii must be a non-empty 1D array")
        if radii.dtype.kind not in ("i", "u"):
            raise ValueError("radii must be integer values")

    fpas = np.empty(radii.size, dtype=np.float64)
    for i, r in enumerate(radii):
        pred_masks = _eroded_prediction(yhat, float(threshold), int(r), element=element, iterations=iterations)
        losses_t = []
        for t in range(yhat.shape[0]):
            losses_t.append(fpa_loss(pred_masks[t], truth_masks[t], spatial_weights))
        fpas[i] = float(np.mean(losses_t))

    mask_ok = fpas <= float(alpha) + 1e-12
    if np.any(mask_ok):
        idx = int(np.argmax(mask_ok))
    else:
        idx = radii.size - 1

    return int(radii[idx]), fpas, radii


def apply_morph_radius(
    preds: np.ndarray,
    threshold: float,
    r_star: int,
    element: Literal["disk", "square"] = "disk",
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply a calibrated erosion radius r* to produce prediction masks on (T,H,W).
    """
    return _eroded_prediction(preds, float(threshold), int(r_star), element=element, iterations=iterations)
