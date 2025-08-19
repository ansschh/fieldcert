# src/calibration/scores.py
from __future__ import annotations

import numpy as np
from typing import Optional, Callable, Tuple


def _to_bool(mask: np.ndarray) -> np.ndarray:
    """Ensure a boolean numpy array."""
    if mask.dtype == np.bool_:
        return mask
    return mask.astype(bool)


def _safe_weights(weights: Optional[np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
    """Broadcast/validate weights array or return ones."""
    if weights is None:
        return np.ones(shape, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != shape:
        try:
            w = np.broadcast_to(w, shape)
        except Exception as e:
            raise ValueError(f"weights shape {w.shape} not broadcastable to {shape}") from e
    return w


def fpa_loss(pred_mask: np.ndarray, truth_mask: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Weighted False-Positive Area (FPA) in [0, 1].

    FPA = sum_x w(x) * 1{ x in pred \ truth } / sum_x w(x)
    """
    p = _to_bool(np.asarray(pred_mask))
    g = _to_bool(np.asarray(truth_mask))
    if p.shape != g.shape:
        raise ValueError(f"Shape mismatch pred {p.shape} vs truth {g.shape}")
    w = _safe_weights(weights, p.shape)
    denom = np.sum(w)
    if denom <= 0:
        return 0.0
    fp = (~g) & p
    return float(np.sum(w[fp]) / denom)


def fna_loss(pred_mask: np.ndarray, truth_mask: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Weighted False-Negative Area (FNA) in [0, 1].

    FNA = sum_x w(x) * 1{ x in truth \ pred } / sum_x w(x)
    """
    p = _to_bool(np.asarray(pred_mask))
    g = _to_bool(np.asarray(truth_mask))
    if p.shape != g.shape:
        raise ValueError(f"Shape mismatch pred {p.shape} vs truth {g.shape}")
    w = _safe_weights(weights, p.shape)
    denom = np.sum(w)
    if denom <= 0:
        return 0.0
    fn = g & (~p)
    return float(np.sum(w[fn]) / denom)


def combined_set_loss(
    pred_mask: np.ndarray,
    truth_mask: np.ndarray,
    weights: Optional[np.ndarray] = None,
    alpha_fp: float = 1.0,
    beta_fn: float = 0.0,
) -> float:
    """
    Combined set loss = alpha_fp * FPA + beta_fn * FNA.

    Useful when misses are more costly than false alarms (or vice versa).
    """
    fpa = fpa_loss(pred_mask, truth_mask, weights)
    if beta_fn == 0.0:
        return float(alpha_fp * fpa)
    fna = fna_loss(pred_mask, truth_mask, weights)
    return float(alpha_fp * fpa + beta_fn * fna)


def jaccard_index(pred_mask: np.ndarray, truth_mask: np.ndarray) -> float:
    """
    Intersection-over-Union (IoU) for binary masks.
    """
    p = _to_bool(np.asarray(pred_mask))
    g = _to_bool(np.asarray(truth_mask))
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def functional_abs_error(
    pred_field: np.ndarray,
    truth_field: np.ndarray,
    functional: str | Callable[[np.ndarray], float] = "max",
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Absolute error for a field-level functional (e.g., max, sum, mean).

    Parameters
    ----------
    pred_field, truth_field : arrays of identical shape
    functional : {"max","sum","mean"} or callable
    weights : optional weights used for "sum"/"mean"
    """
    yhat = np.asarray(pred_field, dtype=np.float64)
    y = np.asarray(truth_field, dtype=np.float64)
    if yhat.shape != y.shape:
        raise ValueError(f"Shape mismatch pred {yhat.shape} vs truth {y.shape}")

    if callable(functional):
        fhat = float(functional(yhat))
        fy = float(functional(y))
        return abs(fhat - fy)

    f = functional.lower()
    if f == "max":
        return float(abs(np.nanmax(yhat) - np.nanmax(y)))
    if f in ("sum", "mean"):
        w = _safe_weights(weights, yhat.shape)
        if f == "sum":
            fhat = float(np.nansum(w * yhat))
            fy = float(np.nansum(w * y))
            return abs(fhat - fy)
        # mean
        denom = float(np.nansum(w))
        if denom <= 0:
            return 0.0
        fhat = float(np.nansum(w * yhat) / denom)
        fy = float(np.nansum(w * y) / denom)
        return abs(fhat - fy)

    raise ValueError(f"Unsupported functional: {functional}")
