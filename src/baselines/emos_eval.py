# src/baselines/emos_eval.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Sequence, Dict, Any

from calibration.scores import fpa_loss


def brier_score(prob: np.ndarray, event: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Weighted Brier score for binary events.

    Parameters
    ----------
    prob : (T,H,W) predicted probabilities in [0,1]
    event : (T,H,W) boolean
    weights : optional (H,W) weights

    Returns
    -------
    mean weighted Brier score
    """
    p = np.asarray(prob, dtype=np.float64)
    e = np.asarray(event).astype(bool)
    if p.shape != e.shape or p.ndim != 3:
        raise ValueError("prob and event must have identical 3D shapes (T,H,W)")
    w = weights
    if w is None:
        w = np.ones(p.shape[1:], dtype=np.float64)
    w2 = np.broadcast_to(w, p.shape)
    return float(np.sum(w2 * (p - e.astype(np.float64)) ** 2) / np.sum(w2))


def reliability_curve(
    prob: np.ndarray,
    event: np.ndarray,
    *,
    n_bins: int = 11,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reliability diagram statistics.

    Returns
    -------
    bin_centers : (n_bins,)
    mean_pred : (n_bins,) average predicted probability per bin
    event_freq : (n_bins,) observed event frequency per bin
    """
    p = np.asarray(prob, dtype=np.float64).ravel()
    e = np.asarray(event).astype(bool).ravel()
    if weights is None:
        w = np.ones_like(p, dtype=np.float64)
    else:
        w = np.broadcast_to(np.asarray(weights, dtype=np.float64), prob.shape).ravel()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    mean_pred = np.zeros(n_bins, dtype=np.float64)
    event_freq = np.zeros(n_bins, dtype=np.float64)
    wsum = np.zeros(n_bins, dtype=np.float64)

    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            mean_pred[b] = np.nan
            event_freq[b] = np.nan
            wsum[b] = 0.0
            continue
        w_b = w[m]
        wsum[b] = w_b.sum()
        mean_pred[b] = float(np.sum(w_b * p[m]) / wsum[b])
        event_freq[b] = float(np.sum(w_b * e[m].astype(np.float64)) / wsum[b])

    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, mean_pred, event_freq


def _fpa_for_threshold(
    prob: np.ndarray,
    event_mask: np.ndarray,
    p_thresh: float,
    spatial_weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute FPA when thresholding probabilities at p_thresh.
    """
    pred_mask = (prob >= float(p_thresh))
    # event_mask is the truth exceedance mask
    Tdim = prob.shape[0]
    losses_t = []
    for t in range(Tdim):
        losses_t.append(fpa_loss(pred_mask[t], event_mask[t], spatial_weights))
    return float(np.mean(losses_t))


def prob_to_set_calibrate_threshold(
    prob_cal: np.ndarray,      # (T,H,W)
    event_cal: np.ndarray,     # (T,H,W) boolean
    alpha: float = 0.10,
    grid: Optional[np.ndarray] = None,
    spatial_weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calibrate a probability threshold p* so that mean FPA ≤ α on calibration data.

    Parameters
    ----------
    prob_cal : predicted probabilities
    event_cal : boolean event masks
    alpha : target FPA
    grid : candidate thresholds (ascending). Default: np.linspace(0,1,101)
    spatial_weights : optional (H,W)

    Returns
    -------
    p_star : selected probability threshold
    fpa_curve : FPA at each candidate threshold
    grid : grid used
    """
    p = np.asarray(prob_cal, dtype=np.float64)
    e = np.asarray(event_cal).astype(bool)
    if p.shape != e.shape or p.ndim != 3:
        raise ValueError("prob_cal and event_cal must have identical 3D shapes (T,H,W)")

    if grid is None:
        grid = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    else:
        grid = np.asarray(grid, dtype=np.float64)
        if grid.ndim != 1 or grid.size == 0:
            raise ValueError("grid must be non-empty 1D")
        if not np.all(np.diff(grid) >= 0):
            raise ValueError("grid must be sorted ascending")

    curve = np.empty(grid.size, dtype=np.float64)
    for i, thr in enumerate(grid):
        curve[i] = _fpa_for_threshold(p, e, float(thr), spatial_weights)

    mask_ok = curve <= float(alpha) + 1e-12
    if np.any(mask_ok):
        idx = int(np.argmax(mask_ok))
    else:
        idx = grid.size - 1
    return float(grid[idx]), curve, grid


def evaluate_prob_event_baseline(
    prob: np.ndarray,            # (T,H,W)
    truths: np.ndarray,          # (T,H,W) continuous field
    threshold: float,
    *,
    spatial_weights: Optional[np.ndarray] = None,
    prob_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate a probability baseline on Brier and FPA/FNA at a given probability threshold.

    If `prob_threshold` is None, this function does not compute FPA/FNA and returns
    only probabilistic metrics. Use `prob_to_set_calibrate_threshold` to find p*.

    Returns
    -------
    dict with keys:
      - brier
      - fpa (optional)
      - fna (optional)
    """
    from calibration.scores import fna_loss

    Tval = float(threshold)
    e_mask = (np.asarray(truths, dtype=np.float64) >= Tval)
    brier = brier_score(prob, e_mask, weights=spatial_weights)
    out: Dict[str, Any] = {"brier": brier}

    if prob_threshold is not None:
        pred_mask = (np.asarray(prob, dtype=np.float64) >= float(prob_threshold))
        fpas = []
        fnas = []
        for t in range(pred_mask.shape[0]):
            fpas.append(fpa_loss(pred_mask[t], e_mask[t], spatial_weights))
            fnas.append(fna_loss(pred_mask[t], e_mask[t], spatial_weights))
        out.update(
            fpa=float(np.mean(fpas)),
            fna=float(np.mean(fnas)),
            prob_threshold=float(prob_threshold),
        )
    return out
