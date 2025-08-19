# src/baselines/pixel_cp.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from calibration.morphology import symmetric_filter


def calibrate_pixelwise_delta(
    preds_cal: np.ndarray,    # (T,H,W)
    truths_cal: np.ndarray,   # (T,H,W)
    threshold: float,
    alpha_pixel: float = 0.10,
    *,
    symmetric_morph: Optional[dict] = None,
) -> np.ndarray:
    """
    Pixelwise split-conformal style baseline (per-pixel threshold bump).

    For each pixel (i,j), compute δ(i,j) as the (1-α_pixel) *quantile* of (yhat - T)
    over calibration times where the event is FALSE (truth < T). Then define the
    prediction set S = { yhat >= T + δ(i,j) }.

    This controls the per-pixel false positive rate under exchangeability assumptions.

    Parameters
    ----------
    preds_cal, truths_cal : arrays (T,H,W)
    threshold : float T
    alpha_pixel : float in (0,1)
    symmetric_morph : optional dict with keys
        {"operation","radius","element","iterations"} applied symmetrically to pred/true masks
        when building *diagnostic* masks; δ calibration itself uses the raw fields.

    Returns
    -------
    delta : (H,W) nonnegative threshold bump for each pixel
    """
    yhat = np.asarray(preds_cal, dtype=np.float64)
    y = np.asarray(truths_cal, dtype=np.float64)
    if yhat.shape != y.shape or yhat.ndim != 3:
        raise ValueError("preds_cal and truths_cal must be (T,H,W) and equal shapes")

    Tval = float(threshold)

    # Compute (yhat - T) for all times, mask out event times where truth >= T
    diffs = yhat - Tval  # shape (T,H,W)
    mask_non_event = (y < Tval)  # True where we consider FP behavior
    diffs_masked = diffs.copy()
    diffs_masked[~mask_non_event] = np.nan

    # Compute (1 - α_pixel)-quantile along time ignoring NaNs, per pixel
    q = 100.0 * (1.0 - float(alpha_pixel))
    delta = np.nanpercentile(diffs_masked, q=q, axis=0)
    # If a pixel never had non-events (extremely rare), set delta=0
    delta = np.where(np.isfinite(delta), delta, 0.0)
    # Ensure δ >= 0 (only shrink predictions)
    delta = np.maximum(delta, 0.0)

    # Optional: sanity check via symmetric morphology on one reconstruction (not required for δ)
    if symmetric_morph is not None:
        # Build a quick diagnostic (won't be returned)
        from calibration.scores import fpa_loss
        pred_mask_raw = (yhat >= (Tval + delta[None, ...]))
        truth_mask_raw = (y >= Tval)
        op = symmetric_morph.get("operation", "none")
        rad = int(symmetric_morph.get("radius", 0))
        elem = symmetric_morph.get("element", "disk")
        iters = int(symmetric_morph.get("iterations", 1))
        if op != "none" and rad > 0 and iters > 0:
            pm = np.empty_like(pred_mask_raw, dtype=bool)
            gm = np.empty_like(truth_mask_raw, dtype=bool)
            for t in range(pred_mask_raw.shape[0]):
                pm[t], gm[t] = symmetric_filter(pred_mask_raw[t], truth_mask_raw[t], op, rad, elem, iters)
            # compute a single diagnostic FPA (no return)
            _ = np.mean([fpa_loss(pm[t], gm[t]) for t in range(pm.shape[0])])

    return delta


def apply_pixelwise_delta(
    preds: np.ndarray,    # (T,H,W)
    threshold: float,
    delta: np.ndarray,    # (H,W)
) -> np.ndarray:
    """
    Apply a pixelwise δ field to create prediction masks: S = { yhat >= T + δ }.
    """
    yhat = np.asarray(preds, dtype=np.float64)
    if yhat.ndim != 3:
        raise ValueError("preds must be (T,H,W)")
    d = np.asarray(delta, dtype=np.float64)
    if d.shape != yhat.shape[1:]:
        raise ValueError(f"delta shape {d.shape} must match (H,W)={yhat.shape[1:]}")
    Tval = float(threshold)
    bumped = Tval + d[None, ...]
    return (yhat >= bumped)
