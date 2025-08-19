# src/eval/sets.py
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional, Sequence, Callable, Any


def _to_bool(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype != np.bool_:
        a = a.astype(bool)
    return a


def _broadcast_weights(weights: Optional[np.ndarray], shape_2d: Tuple[int, int]) -> np.ndarray:
    if weights is None:
        return np.ones(shape_2d, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    try:
        w = np.broadcast_to(w, shape_2d)
    except Exception as e:
        raise ValueError(f"weights with shape {w.shape} not broadcastable to {shape_2d}") from e
    return w


def confusion_areas(
    pred_mask: np.ndarray,  # 2D boolean
    truth_mask: np.ndarray,  # 2D boolean
    weights: Optional[np.ndarray] = None,  # 2D weights
) -> Dict[str, float]:
    """
    Weighted confusion areas (TP, FP, TN, FN) normalized to sum to 1 over domain.

    Returns
    -------
    dict with keys {"tp","fp","tn","fn"} summing to 1.0
    """
    p = _to_bool(pred_mask)
    g = _to_bool(truth_mask)
    if p.shape != g.shape:
        raise ValueError(f"Shape mismatch: pred {p.shape} vs truth {g.shape}")
    w = _broadcast_weights(weights, p.shape)
    W = float(np.sum(w))
    if W <= 0:
        return {"tp": 0.0, "fp": 0.0, "tn": 1.0, "fn": 0.0}

    tp = float(np.sum(w[p & g]) / W)
    fp = float(np.sum(w[p & (~g)]) / W)
    tn = float(np.sum(w[(~p) & (~g)]) / W)
    fn = float(np.sum(w[(~p) & g]) / W)
    # Normalize for numerical safety
    s = tp + fp + tn + fn
    if s > 0:
        tp, fp, tn, fn = (tp / s, fp / s, tn / s, fn / s)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def set_area(mask: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Weighted area fraction of a set (2D). Returns value in [0,1].
    """
    m = _to_bool(mask)
    w = _broadcast_weights(weights, m.shape)
    denom = float(np.sum(w))
    if denom <= 0:
        return 0.0
    return float(np.sum(w[m]) / denom)


def _boundary_length_2d(mask2d: np.ndarray) -> float:
    """
    Approximate boundary length by counting 4-neighbor edges where mask changes.
    """
    m = _to_bool(mask2d).astype(np.uint8)
    # horizontal edges between columns
    dh = np.not_equal(m[:, 1:], m[:, :-1]).sum()
    # vertical edges between rows
    dv = np.not_equal(m[1:, :], m[:-1, :]).sum()
    return float(dh + dv)


def boundary_length(mask: np.ndarray) -> float:
    """
    Average boundary length over time if 3D; or length for a single 2D mask.

    Returns
    -------
    float boundary length in pixel-edge units.
    """
    arr = np.asarray(mask)
    if arr.ndim == 2:
        return _boundary_length_2d(arr)
    if arr.ndim == 3:
        return float(np.mean([_boundary_length_2d(arr[t]) for t in range(arr.shape[0])]))
    raise ValueError(f"mask must be 2D or 3D, got shape {arr.shape}")


def compactness(mask: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Compactness metric (2D) averaged over time if 3D:

        C = 4π * Area / Perimeter^2

    We compute area as weighted fraction and perimeter as pixel-edge count.
    Returns NaN if perimeter is zero (no set).
    """
    arr = np.asarray(mask)
    if arr.ndim == 2:
        A = set_area(arr, weights=weights)
        P = _boundary_length_2d(arr)
        if P == 0:
            return np.nan
        return float(4.0 * np.pi * A / (P ** 2))
    if arr.ndim == 3:
        vals = []
        for t in range(arr.shape[0]):
            A = set_area(arr[t], weights=weights)
            P = _boundary_length_2d(arr[t])
            vals.append(np.nan if P == 0 else (4.0 * np.pi * A / (P ** 2)))
        return float(np.nanmean(vals))
    raise ValueError(f"mask must be 2D or 3D, got shape {arr.shape}")


def _fpa_2d(pred: np.ndarray, truth: np.ndarray, weights: Optional[np.ndarray]) -> float:
    w = _broadcast_weights(weights, pred.shape)
    denom = float(np.sum(w))
    if denom <= 0:
        return 0.0
    return float(np.sum(w[(pred.astype(bool)) & (~truth.astype(bool))]) / denom)


def _fna_2d(pred: np.ndarray, truth: np.ndarray, weights: Optional[np.ndarray]) -> float:
    w = _broadcast_weights(weights, pred.shape)
    denom = float(np.sum(w))
    if denom <= 0:
        return 0.0
    return float(np.sum(w[(~pred.astype(bool)) & (truth.astype(bool))]) / denom)


def _iou_2d(pred: np.ndarray, truth: np.ndarray) -> float:
    p = _to_bool(pred)
    g = _to_bool(truth)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return 1.0 if union == 0 else float(inter / union)


def block_bootstrap_ci(
    values: np.ndarray,
    block_ids: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: Optional[int] = 42,
) -> Tuple[float, float]:
    """
    Block bootstrap CI for the mean of per-time values.

    Parameters
    ----------
    values : (T,) per-time metric values
    block_ids : (T,) block assignment; resample blocks with replacement
    n_boot : number of bootstrap replicates
    ci : confidence level
    random_state : seed

    Returns
    -------
    (lo, hi) CI bounds
    """
    rng = np.random.default_rng(random_state)
    vals = np.asarray(values, dtype=np.float64)
    bids = np.asarray(block_ids)
    uniq, inv = np.unique(bids, return_inverse=True)
    # aggregate by block
    block_means = np.zeros(uniq.size, dtype=np.float64)
    counts = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(block_means, inv, vals)
    np.add.at(counts, inv, 1.0)
    counts = np.maximum(counts, 1.0)
    block_means /= counts

    # bootstrap over blocks
    draws = rng.integers(0, uniq.size, size=(n_boot, uniq.size))
    boot = block_means[draws].mean(axis=1)
    alpha = 1.0 - ci
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return lo, hi


def evaluate_set_masks(
    pred_masks: np.ndarray,     # (T,H,W) boolean
    truth_masks: np.ndarray,    # (T,H,W) boolean
    *,
    weights: Optional[np.ndarray] = None,  # (H,W)
    block_ids: Optional[np.ndarray] = None,  # (T,)
    compute_ci: bool = False,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Evaluate set-valued predictions against truth across time.

    Metrics reported:
      - fpa, fna, iou (averaged over time)
      - area_pred, area_truth (avg weighted area fraction)
      - boundary_length_pred (avg)
      - compactness_pred (avg)
      - optional CIs via block bootstrap

    Returns
    -------
    dict with means and (optional) CI bounds per metric.
    """
    P = _to_bool(pred_masks)
    G = _to_bool(truth_masks)
    if P.shape != G.shape or P.ndim != 3:
        raise ValueError(f"pred_masks and truth_masks must both be (T,H,W); got {P.shape} vs {G.shape}")

    Tdim = P.shape[0]
    w = _broadcast_weights(weights, P.shape[1:])

    fpa_t = np.empty(Tdim, dtype=np.float64)
    fna_t = np.empty(Tdim, dtype=np.float64)
    iou_t = np.empty(Tdim, dtype=np.float64)
    area_p_t = np.empty(Tdim, dtype=np.float64)
    area_g_t = np.empty(Tdim, dtype=np.float64)
    bl_p_t = np.empty(Tdim, dtype=np.float64)
    comp_p_t = np.empty(Tdim, dtype=np.float64)

    for t in range(Tdim):
        fpa_t[t] = _fpa_2d(P[t], G[t], w)
        fna_t[t] = _fna_2d(P[t], G[t], w)
        iou_t[t] = _iou_2d(P[t], G[t])
        area_p_t[t] = set_area(P[t], w)
        area_g_t[t] = set_area(G[t], w)
        bl_p_t[t] = _boundary_length_2d(P[t])
        # compactness per time (weights used for area)
        comp_p_t[t] = compactness(P[t], w)

    report: Dict[str, Any] = dict(
        fpa=float(np.mean(fpa_t)),
        fna=float(np.mean(fna_t)),
        iou=float(np.mean(iou_t)),
        area_pred=float(np.mean(area_p_t)),
        area_truth=float(np.mean(area_g_t)),
        boundary_length_pred=float(np.mean(bl_p_t)),
        compactness_pred=float(np.nanmean(comp_p_t)),
        T=P.shape[0],
        H=P.shape[1],
        W=P.shape[2],
    )

    if compute_ci:
        if block_ids is None or block_ids.shape[0] != Tdim:
            raise ValueError("block_ids with length T must be provided when compute_ci=True")
        for name, series in [
            ("fpa", fpa_t),
            ("fna", fna_t),
            ("iou", iou_t),
            ("area_pred", area_p_t),
            ("area_truth", area_g_t),
            ("boundary_length_pred", bl_p_t),
            ("compactness_pred", comp_p_t),
        ]:
            lo, hi = block_bootstrap_ci(series, block_ids=block_ids, n_boot=n_boot, ci=ci, random_state=random_state)
            report[f"{name}_lo"] = lo
            report[f"{name}_hi"] = hi

    return report


def evaluate_functional_intervals(
    lower: np.ndarray,  # (T,)
    upper: np.ndarray,  # (T,)
    truth_vals: np.ndarray,  # (T,)
) -> Dict[str, float]:
    """
    Evaluate functional prediction intervals (e.g., domain max, areal totals).

    Returns
    -------
    dict with:
      - coverage: fraction of times truth in [lower, upper]
      - avg_width: mean (upper - lower)
      - avg_violations: mean violation size when outside the interval
    """
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    y = np.asarray(truth_vals, dtype=np.float64).ravel()
    if not (lo.shape == hi.shape == y.shape):
        raise ValueError("lower, upper, truth_vals must have the same length")
    if np.any(hi < lo):
        raise ValueError("upper must be >= lower element-wise")

    inside = (y >= lo) & (y <= hi)
    coverage = float(np.mean(inside.astype(np.float64)))
    width = float(np.mean(hi - lo))
    # violation distance to interval
    below = np.maximum(0.0, lo - y)
    above = np.maximum(0.0, y - hi)
    violations = below + above
    avg_viol = float(np.mean(violations))
    return {"coverage": coverage, "avg_width": width, "avg_violations": avg_viol}
