# src/calibration/margins.py
from __future__ import annotations

import numpy as np
from typing import Optional, Literal, Tuple


def gradient_magnitude(field: np.ndarray, axis_order: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    """
    Compute finite-difference gradient magnitude for a 2D or 3D (T,H,W) field.

    If field is 3D, derivatives are computed per time slice along the last two axes.
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")
    if arr.ndim == 2:
        gy, gx = np.gradient(arr)
        mag = np.hypot(gx, gy)
        return np.asarray(mag)
    # 3D: slice-wise
    mags = np.empty_like(arr)
    for t in range(arr.shape[0]):
        gy, gx = np.gradient(arr[t])
        mags[t] = np.hypot(gx, gy)
    return mags


def divergence_plus(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute the positive part of divergence for a vector field (u,v) on a Cartesian grid.

    Returns
    -------
    div_plus >= 0 with same shape as u/v.

    Notes: For global lat-lon grids, this is an approximation (no cos(lat) factor).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"u and v shapes differ: {u.shape} vs {v.shape}")

    if u.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D arrays; got u.ndim={u.ndim}")

    if u.ndim == 2:
        dv_dy, du_dx = np.gradient(v)[0], np.gradient(u)[1]
        div = du_dx + dv_dy
        return np.maximum(div, 0.0)

    # 3D (T,H,W)
    out = np.empty_like(u)
    for t in range(u.shape[0]):
        dv_dy, du_dx = np.gradient(v[t])[0], np.gradient(u[t])[1]
        div = du_dx + dv_dy
        out[t] = np.maximum(div, 0.0)
    return out


def robust_minmax_scale(arr: np.ndarray, q_low: float = 5.0, q_high: float = 95.0, eps: float = 1e-8) -> np.ndarray:
    """
    Scale array to [0,1] using robust percentiles (per-slice if 3D).

    For 3D arrays, scaling is done independently per time slice to adapt margins dynamically.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:
        lo, hi = np.nanpercentile(a, [q_low, q_high])
        if not np.isfinite(hi - lo) or hi <= lo:
            return np.zeros_like(a)
        return np.clip((a - lo) / (hi - lo + eps), 0.0, 1.0)
    elif a.ndim == 3:
        out = np.empty_like(a)
        for t in range(a.shape[0]):
            lo, hi = np.nanpercentile(a[t], [q_low, q_high])
            if not np.isfinite(hi - lo) or hi <= lo:
                out[t] = 0.0
            else:
                out[t] = np.clip((a[t] - lo) / (hi - lo + eps), 0.0, 1.0)
        return out
    else:
        raise ValueError(f"Expected 2D or 3D, got {a.shape}")


def build_margin_field(
    pred_field: np.ndarray,
    method: Literal["grad_mag", "constant", "divergence_plus", "custom"] = "grad_mag",
    *,
    constant_value: float = 1.0,
    wind_u: Optional[np.ndarray] = None,
    wind_v: Optional[np.ndarray] = None,
    c0: float = 0.0,
    c1: float = 1.0,
    c2: float = 0.0,
    normalize: bool = True,
    q_low: float = 5.0,
    q_high: float = 95.0,
    custom_field: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Construct a nonnegative margin field m(x) to be used in threshold-bump S(λ) = { yhat >= T + λ m }.

    method:
      - "grad_mag": m = c0 + c1*|∇yhat| + c2*div_plus (if u/v provided)
      - "divergence_plus": m = c0 + c2*div_plus(u,v)
      - "constant": m = constant_value
      - "custom": m = custom_field (validated nonnegative)

    normalize:
      If True, apply robust_minmax_scale to m to keep values in [0,1], preserving zeros.
    """
    yhat = np.asarray(pred_field, dtype=np.float64)

    if method == "constant":
        m = np.full_like(yhat, fill_value=max(constant_value, 0.0), dtype=np.float64)

    elif method == "grad_mag":
        gm = gradient_magnitude(yhat)
        m = c0 + c1 * gm
        if wind_u is not None and wind_v is not None and c2 != 0.0:
            dplus = divergence_plus(wind_u, wind_v)
            m = m + c2 * dplus

    elif method == "divergence_plus":
        if wind_u is None or wind_v is None:
            raise ValueError("divergence_plus method requires wind_u and wind_v")
        m = c0 + c2 * divergence_plus(wind_u, wind_v)

    elif method == "custom":
        if custom_field is None:
            raise ValueError("custom_field must be provided when method='custom'")
        m = np.asarray(custom_field, dtype=np.float64)
        if m.shape != yhat.shape:
            try:
                m = np.broadcast_to(m, yhat.shape)
            except Exception as e:
                raise ValueError(f"custom_field shape {m.shape} not broadcastable to {yhat.shape}") from e
        # ensure nonnegative
        m = np.maximum(m, 0.0)
    else:
        raise ValueError(f"Unknown margin method: {method}")

    if normalize:
        # Preserve sparsity: only scale where m>0 to avoid creating spurious nonzeros.
        mask_pos = m > 0
        if np.any(mask_pos):
            scaled = robust_minmax_scale(m)
            m[mask_pos] = scaled[mask_pos]
        # clamp negatives
        m = np.maximum(m, 0.0)

    return m
