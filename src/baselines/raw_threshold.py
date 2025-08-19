# src/baselines/raw_threshold.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Literal

from calibration.morphology import symmetric_filter


def build_truth_masks(
    truths: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Build binary ground-truth exceedance masks for a 3D field stack.

    Parameters
    ----------
    truths : array-like (T,H,W)
    threshold : float

    Returns
    -------
    mask_truth : boolean array (T,H,W)
    """
    y = np.asarray(truths, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(f"truths must be 3D (T,H,W), got {y.shape}")
    return (y >= float(threshold))


def make_raw_masks(
    preds: np.ndarray,
    threshold: float,
    *,
    truths: Optional[np.ndarray] = None,
    morph_operation: Literal["none", "open", "close", "erode", "dilate"] = "none",
    morph_radius: int = 0,
    morph_element: Literal["disk", "square"] = "disk",
    morph_iterations: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Raw deterministic exceedance masks with optional symmetric morphology.

    If `truths` is provided and morphology != "none", applies the same morphological
    operator to predicted and truth masks to preserve fair comparison.

    Parameters
    ----------
    preds : array-like (T,H,W)
    threshold : float
    truths : optional array-like (T,H,W)
    morph_operation : morphological op to apply symmetrically
    morph_radius : int
    morph_element : {"disk","square"}
    morph_iterations : int

    Returns
    -------
    pred_mask : (T,H,W) boolean
    truth_mask : (T,H,W) boolean or None if truths not supplied
    """
    yhat = np.asarray(preds, dtype=np.float64)
    if yhat.ndim != 3:
        raise ValueError(f"preds must be 3D (T,H,W), got {yhat.shape}")
    Tval = float(threshold)

    pred_mask = (yhat >= Tval)

    truth_mask = None
    if truths is not None:
        truth_mask = build_truth_masks(truths, Tval)

    if morph_operation != "none" and morph_radius > 0 and morph_iterations > 0:
        if truths is None:
            # Morph only on predictions if no truth given (e.g., pre-processing)
            pm_out = np.empty_like(pred_mask, dtype=bool)
            for t in range(pred_mask.shape[0]):
                pm, _ = symmetric_filter(
                    pred_mask[t],
                    pred_mask[t],  # dummy, same op applied
                    operation=morph_operation,
                    radius=morph_radius,
                    element=morph_element,
                    iterations=morph_iterations,
                )
                pm_out[t] = pm
            pred_mask = pm_out
        else:
            pm_out = np.empty_like(pred_mask, dtype=bool)
            gm_out = np.empty_like(truth_mask, dtype=bool)
            for t in range(pred_mask.shape[0]):
                pm, gm = symmetric_filter(
                    pred_mask[t],
                    truth_mask[t],
                    operation=morph_operation,
                    radius=morph_radius,
                    element=morph_element,
                    iterations=morph_iterations,
                )
                pm_out[t] = pm
                gm_out[t] = gm
            pred_mask, truth_mask = pm_out, gm_out

    return pred_mask, truth_mask
