# src/eval/prob.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


def brier_score(
    prob: np.ndarray,      # (T,H,W)
    event: np.ndarray,     # (T,H,W) boolean
    weights: Optional[np.ndarray] = None,  # (H,W)
) -> float:
    """
    Weighted Brier score for binary events.

    Returns
    -------
    scalar mean over time/space, weighted by `weights` over (H,W).
    """
    p = np.asarray(prob, dtype=np.float64)
    e = np.asarray(event).astype(bool)
    if p.shape != e.shape or p.ndim != 3:
        raise ValueError(f"Shapes must match and be 3D: prob{p.shape} vs event{e.shape}")
    if weights is None:
        w2 = np.ones_like(p, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w2 = np.broadcast_to(w, p.shape)
    num = np.sum(w2 * (p - e.astype(np.float64)) ** 2)
    den = float(np.sum(w2))
    return float(num / den) if den > 0 else 0.0


def brier_skill_score(
    prob: np.ndarray,      # (T,H,W)
    event: np.ndarray,     # (T,H,W) boolean
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Brier Skill Score relative to climatology (event frequency).

    BSS = 1 - BS / BS_ref, where BS_ref is the Brier score of constant p = event frequency.
    """
    p = np.asarray(prob, dtype=np.float64)
    e = np.asarray(event).astype(bool)
    if p.shape != e.shape or p.ndim != 3:
        raise ValueError("prob and event must be same shape (T,H,W)")
    if weights is None:
        w2 = np.ones_like(p, dtype=np.float64)
    else:
        w2 = np.broadcast_to(np.asarray(weights, dtype=np.float64), p.shape)

    # Climatological event frequency
    freq = float(np.sum(w2 * e) / np.sum(w2)) if np.sum(w2) > 0 else np.mean(e)
    bs = brier_score(p, e, weights=weights)
    bs_ref = brier_score(np.full_like(p, fill_value=freq, dtype=np.float64), e, weights=weights)
    if bs_ref <= 1e-12:
        return 0.0
    return float(1.0 - bs / bs_ref)


def reliability_curve(
    prob: np.ndarray,      # (T,H,W)
    event: np.ndarray,     # (T,H,W) boolean
    *,
    n_bins: int = 11,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reliability diagram statistics.

    Returns
    -------
    (bin_centers, mean_pred_per_bin, event_freq_per_bin)
    """
    p = np.asarray(prob, dtype=np.float64).ravel()
    e = np.asarray(event).astype(bool).ravel()
    if weights is None:
        w = np.ones_like(p, dtype=np.float64)
    else:
        # broadcast weights (H,W) to (T,H,W) then ravel
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
        wb = w[m]
        wsum[b] = wb.sum()
        mean_pred[b] = float(np.sum(wb * p[m]) / wsum[b])
        event_freq[b] = float(np.sum(wb * e[m].astype(np.float64)) / wsum[b])

    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, mean_pred, event_freq


def crps_ensemble(
    truth: np.ndarray,            # (T,H,W)
    ensemble: np.ndarray,         # (T,M,H,W)
    weights: Optional[np.ndarray] = None,   # (H,W)
    chunk_size: int = 20000,
) -> float:
    """
    Continuous Ranked Probability Score (CRPS) for an ensemble forecast.

    Uses the unbiased ensemble approximation:
        CRPS = (1/M) * mean |X_m - y|  -  (1/M^2) * sum_{k=1}^M (2k - M - 1) X_(k)
    where X_(k) are ensemble members sorted ascending (per sample).

    Implemented in memory-safe chunks over flattened samples.

    Parameters
    ----------
    truth : (T,H,W) array of observations
    ensemble : (T,M,H,W) array of ensemble members
    weights : optional (H,W) spatial weights
    chunk_size : number of samples per chunk to avoid large memory use

    Returns
    -------
    scalar weighted mean CRPS over all samples.
    """
    y = np.asarray(truth, dtype=np.float64)
    ens = np.asarray(ensemble, dtype=np.float64)
    if ens.ndim != 4:
        raise ValueError("ensemble must be 4D (T,M,H,W)")
    if y.shape != (ens.shape[0], ens.shape[2], ens.shape[3]):
        raise ValueError(f"truth shape {y.shape} must match ensemble (T,H,W) {(ens.shape[0], ens.shape[2], ens.shape[3])}")

    Tdim, M, H, W = ens.shape
    N = Tdim * H * W

    if weights is None:
        w2 = np.ones((H, W), dtype=np.float64)
    else:
        w2 = np.asarray(weights, dtype=np.float64)
        w2 = np.broadcast_to(w2, (H, W))

    # Flatten truth and weights
    y_flat = y.reshape(N)
    w_flat = np.broadcast_to(w2, (Tdim, H, W)).reshape(N)

    # Prepare coefficient vector for the dispersion term
    # coeff_k = (2k - M - 1) / M^2  for k=1..M
    k = np.arange(1, M + 1, dtype=np.float64)
    coeff = (2.0 * k - M - 1.0) / (M * M)  # shape (M,)

    # Iterate in chunks
    crps_sum = 0.0
    weight_sum = 0.0
    start = 0
    while start < N:
        end = min(start + chunk_size, N)
        idx = slice(start, end)

        # Extract chunk of ensemble: reshape to (chunk_n, M)
        # Build view by stacking along samples
        ens_chunk = ens.reshape(Tdim, M, H, W).transpose(0, 2, 3, 1).reshape(N, M)[idx]  # (n, M)
        y_chunk = y_flat[idx]  # (n,)
        w_chunk = w_flat[idx]  # (n,)

        # First term: mean absolute error to observation
        term1 = np.mean(np.abs(ens_chunk - y_chunk[:, None]), axis=1)  # (n,)

        # Second term: ensemble self-dispersion via sorted values and coeff
        xs = np.sort(ens_chunk, axis=1)  # (n,M)
        term2 = xs @ coeff  # (n,)

        crps = term1 - term2  # (n,)
        # Weighted accumulation
        crps_sum += float(np.sum(w_chunk * crps))
        weight_sum += float(np.sum(w_chunk))

        start = end

    return float(crps_sum / weight_sum) if weight_sum > 0 else 0.0
