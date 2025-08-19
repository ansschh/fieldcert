# src/baselines/global_bump.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from calibration.scores import fpa_loss
from .raw_threshold import build_truth_masks


def _apply_global_bump_mask(
    preds: np.ndarray,
    threshold: float,
    tau: float,
) -> np.ndarray:
    yhat = np.asarray(preds, dtype=np.float64)
    Tval = float(threshold)
    return (yhat >= (Tval + float(tau)))


def calibrate_global_bump_by_fpa(
    preds_cal: np.ndarray,     # (T,H,W)
    truths_cal: np.ndarray,    # (T,H,W)
    threshold: float,
    alpha: float = 0.10,
    tau_grid: Optional[np.ndarray] = None,
    spatial_weights: Optional[np.ndarray] = None,  # (H,W)
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calibrate a *global* threshold bump τ to achieve mean FPA ≤ α on the calibration set.

    Parameters
    ----------
    preds_cal, truths_cal : arrays (T,H,W)
    threshold : float
    alpha : target FPA
    tau_grid : optional 1D array of τ values (ascending); default 0..5 in 0.1 steps
    spatial_weights : optional (H,W) weights

    Returns
    -------
    tau_star : selected τ
    risk_curve : FPA per τ (uncorrected empirical mean)
    tau_grid : grid used
    """
    yhat = np.asarray(preds_cal, dtype=np.float64)
    y = np.asarray(truths_cal, dtype=np.float64)
    if yhat.shape != y.shape or yhat.ndim != 3:
        raise ValueError("preds_cal and truths_cal must be 3D with identical shapes (T,H,W)")
    if tau_grid is None:
        tau_grid = np.arange(0.0, 5.0001, 0.1, dtype=np.float64)
    else:
        tau_grid = np.asarray(tau_grid, dtype=np.float64)
        if tau_grid.ndim != 1 or tau_grid.size == 0:
            raise ValueError("tau_grid must be a non-empty 1D array")
        if not np.all(np.diff(tau_grid) >= 0):
            raise ValueError("tau_grid must be sorted ascending")

    truth_masks = build_truth_masks(y, float(threshold))

    # Compute mean FPA for each τ (time-wise average)
    fpas = np.empty(tau_grid.size, dtype=np.float64)
    for i, tau in enumerate(tau_grid):
        pred_masks = _apply_global_bump_mask(yhat, float(threshold), float(tau))
        # average over time
        losses_t = []
        for t in range(yhat.shape[0]):
            losses_t.append(fpa_loss(pred_masks[t], truth_masks[t], spatial_weights))
        fpas[i] = float(np.mean(losses_t))

    # Select smallest τ achieving ≤ α; if none, pick largest τ
    mask_ok = fpas <= float(alpha) + 1e-12
    if np.any(mask_ok):
        idx = int(np.argmax(mask_ok))
    else:
        idx = tau_grid.size - 1

    return float(tau_grid[idx]), fpas, tau_grid


def apply_global_bump(
    preds: np.ndarray,
    threshold: float,
    tau_star: float,
) -> np.ndarray:
    """
    Apply a previously calibrated global bump τ* to get prediction masks on (T,H,W).
    """
    return _apply_global_bump_mask(preds, float(threshold), float(tau_star))
