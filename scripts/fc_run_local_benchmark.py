# scripts/fc_run_local_benchmark.py
from __future__ import annotations
import os, sys, argparse, json, heapq
import numpy as np
import xarray as xr
from typing import Dict, Any, Tuple, Optional, List

# make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calibration import CRCCalibrator, CRCSettings  # type: ignore
from calibration.morphology import symmetric_filter, morphological_filter  # type: ignore
from baselines import calibrate_global_bump_by_fpa, apply_global_bump  # type: ignore
from baselines import calibrate_morph_radius_by_fpa, apply_morph_radius  # type: ignore
from baselines.emos_eval import prob_to_set_calibrate_threshold, evaluate_prob_event_baseline  # type: ignore
from eval import compute_area_weights  # type: ignore

# ------------------ Utilities ------------------

def open_local(path: str) -> xr.DataArray:
    """Open a local Zarr written by wb2_download_local.py."""
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    da = xr.open_zarr(path, consolidated=True)
    # result can be Dataset if saved as DA; normalize to DataArray
    if isinstance(da, xr.Dataset):
        if len(da.data_vars) != 1:
            raise ValueError(f"Expected one variable in {path}, found {list(da.data_vars)}")
        name = list(da.data_vars)[0]
        da = da[name]
    # Ensure coord names/order
    if "time" not in da.coords and "forecast_time" in da.coords:
        da = da.rename({"forecast_time": "time"})
    if "latitude" in da.dims and "lat" not in da.dims:
        da = da.rename({"latitude": "lat"})
    if "longitude" in da.dims and "lon" not in da.dims:
        da = da.rename({"longitude": "lon"})
    return da.transpose("time", ...).chunk({"time": da.chunks["time"][0] if "time" in da.chunks else 64})

def grad_mag_norm(frames: np.ndarray) -> np.ndarray:
    """Per-frame gradient magnitude, normalized to [0,1] per frame."""
    T, H, W = frames.shape
    out = np.empty_like(frames, dtype=np.float64)
    for i in range(T):
        f = frames[i]
        gy, gx = np.gradient(f)
        mag = np.hypot(gx, gy)
        m = float(np.max(mag))
        out[i] = (mag / m) if m > 0 else 0.0
    return out

def weighted_fpa(pred: np.ndarray, truth: np.ndarray, w: np.ndarray) -> float:
    W = float(np.sum(w))
    return 0.0 if W == 0 else float(np.sum(w[(pred & (~truth))]) / W)

def weighted_fna(pred: np.ndarray, truth: np.ndarray, w: np.ndarray) -> float:
    W = float(np.sum(w))
    return 0.0 if W == 0 else float(np.sum(w[((~pred) & truth)]) / W)

def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return 1.0 if union == 0 else float(inter / union)

def iterate_chunks(da_fc: xr.DataArray, da_tr: xr.DataArray, chunk_t: int):
    """Yield aligned numpy blocks of (yhat, y) with at most chunk_t time steps."""
    assert da_fc.sizes["time"] == da_tr.sizes["time"], "forecast/truth times must be same length"
    T = da_fc.sizes["time"]
    for s in range(0, T, chunk_t):
        e = min(s + chunk_t, T)
        yhat = da_fc.isel(time=slice(s, e)).data  # dask array
        y = da_tr.isel(time=slice(s, e)).data
        # compute to NumPy here (chunk-sized)
        yhat = np.asarray(yhat.compute())
        y = np.asarray(y.compute())
        yield yhat, y

def calibrate_crc_streaming(da_fc_cal: xr.DataArray, da_tr_cal: xr.DataArray, Tval: float,
                            lam_grid: np.ndarray, w_hw: np.ndarray,
                            morph_op: str, morph_r: int, morph_elem: str, morph_iters: int,
                            chunk_t: int) -> float:
    """Streaming CRC calibration: returns lambda* achieving target alpha (mean FPA)."""
    num = np.zeros_like(lam_grid, dtype=np.float64)  # sum of FPA_t numerators over time
    count_t = 0
    Wsum = float(np.sum(w_hw))
    for yhat_blk, y_blk in iterate_chunks(da_fc_cal, da_tr_cal, chunk_t):
        margins = grad_mag_norm(yhat_blk)
        truth_mask = (y_blk >= Tval)
        for j, lam in enumerate(lam_grid):
            pred_mask = (yhat_blk >= (Tval + lam * margins))
            # optional symmetric morphology during calibration
            if morph_op != "none" and morph_r > 0 and morph_iters > 0:
                pm = np.empty_like(pred_mask, dtype=bool)
                gm = np.empty_like(truth_mask, dtype=bool)
                for t in range(pred_mask.shape[0]):
                    pm[t], gm[t] = symmetric_filter(pred_mask[t], truth_mask[t],
                                                    morph_op, int(morph_r), morph_elem, int(morph_iters))
                pred_mask = pm; truth_mask_blk = gm
            else:
                truth_mask_blk = truth_mask
            # sum weighted FP area across frames in this block
            for t in range(pred_mask.shape[0]):
                num[j] += np.sum(w_hw[(pred_mask[t] & (~truth_mask_blk[t]))])
        count_t += yhat_blk.shape[0]
    # mean FPA across time = (num / Wsum) / count_t
    fpa_mean = (num / Wsum) / max(1, count_t)
    mask_ok = fpa_mean <= (crc_alpha + 1e-12)  # crc_alpha is set in main before call
    if np.any(mask_ok):
        return float(lam_grid[np.argmax(mask_ok)])
    return float(lam_grid[-1])

def calibrate_prob_threshold_streaming(da_fc_cal: xr.DataArray, da_tr_cal: xr.DataArray, Tval: float,
                                      iso_model, p_grid: np.ndarray, w_hw: np.ndarray,
                                      chunk_t: int) -> float:
    """Calibrate probability threshold p* by streaming FPA over a grid."""
    num = np.zeros_like(p_grid, dtype=np.float64)
    count_t = 0
    Wsum = float(np.sum(w_hw))
    for yhat_blk, y_blk in iterate_chunks(da_fc_cal, da_tr_cal, chunk_t):
        prob = iso_model.predict(yhat_blk.ravel()).reshape(yhat_blk.shape)
        truth_mask = (y_blk >= Tval)
        for j, thr in enumerate(p_grid):
            pred_mask = (prob >= float(thr))
            for t in range(pred_mask.shape[0]):
                num[j] += np.sum(w_hw[(pred_mask[t] & (~truth_mask[t]))])
        count_t += yhat_blk.shape[0]
    fpa = (num / Wsum) / max(1, count_t)
    ok = fpa <= (crc_alpha + 1e-12)
    return float(p_grid[np.argmax(ok)]) if np.any(ok) else float(p_grid[-1])

def sample_for_isotonic(da_fc_cal: xr.DataArray, da_tr_cal: xr.DataArray, Tval: float,
                        max_samples: int, rng: np.random.Generator, chunk_t: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs: list[np.ndarray] = []; Ys: list[np.ndarray] = []
    remain = max_samples
    for yhat_blk, y_blk in iterate_chunks(da_fc_cal, da_tr_cal, chunk_t):
        x = yhat_blk.ravel()
        y = (y_blk >= Tval).astype(np.int8).ravel()
        n = x.shape[0]
        take = min(remain, n)
        if take <= 0: break
        idx = rng.choice(n, size=take, replace=False)
        Xs.append(x[idx]); Ys.append(y[idx])
        remain -= take
    if len(Xs) == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int8)
    return np.concatenate(Xs), np.concatenate(Ys)

def evaluate_streaming(da_fc_test: xr.DataArray, da_tr_test: xr.DataArray, Tval: float, w_hw: np.ndarray,
                       method: str, params: Dict[str, Any], morph_cfg: Dict[str, Any], chunk_t: int) -> Dict[str, float]:
    """Compute FPA/FNA/IoU for a given UQ method streaming over time."""
    fpa_list: list[float] = []; fna_list: list[float] = []; iou_list: list[float] = []
    Wsum = float(np.sum(w_hw))
    for yhat_blk, y_blk in iterate_chunks(da_fc_test, da_tr_test, chunk_t):
        truth_mask = (y_blk >= Tval)
        if method == "crc":
            lam = float(params["lambda_star"])
            margins = grad_mag_norm(yhat_blk)
            pred_mask = (yhat_blk >= (Tval + lam * margins))
            if morph_cfg["operation"] != "none" and morph_cfg["radius"] > 0 and morph_cfg["iterations"] > 0:
                pm = np.empty_like(pred_mask, dtype=bool); gm = np.empty_like(truth_mask, dtype=bool)
                for t in range(pred_mask.shape[0]):
                    pm[t], gm[t] = symmetric_filter(pred_mask[t], truth_mask[t],
                                                    morph_cfg["operation"], int(morph_cfg["radius"]),
                                                    morph_cfg["element"], int(morph_cfg["iterations"]))
                pred_mask, truth_mask = pm, gm
        elif method == "global":
            tau = float(params["tau_star"])
            pred_mask = (yhat_blk >= (Tval + tau))
        elif method == "morph":
            r = int(params["r_star"])
            base = (yhat_blk >= Tval)
            pred_mask = np.empty_like(base, dtype=bool)
            for t in range(base.shape[0]):
                pred_mask[t] = morphological_filter(base[t], operation="erode",
                                                    radius=r, element="disk", iterations=1)
        elif method == "prob":
            iso = params["iso"]; p_star = float(params["p_star"])
            prob = iso.predict(yhat_blk.ravel()).reshape(yhat_blk.shape)
            pred_mask = (prob >= p_star)
        else:
            raise ValueError(method)

        for t in range(pred_mask.shape[0]):
            p = pred_mask[t]; g = truth_mask[t]
            fpa_list.append(0.0 if Wsum == 0 else float(np.sum(w_hw[(p & (~g))]) / Wsum))
            fna_list.append(0.0 if Wsum == 0 else float(np.sum(w_hw[((~p) & g)]) / Wsum))
            iou_list.append(iou(p, g))
    return {
        "fpa": float(np.mean(fpa_list)),
        "fna": float(np.mean(fna_list)),
        "iou": float(np.mean(iou_list)),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chunked, local benchmark: FieldCert + baselines (memory-safe).")
    ap.add_argument("--forecast_zarr", required=True, help="Path to local forecast zarr")
    ap.add_argument("--truth_zarr", required=True, help="Path to local truth zarr")
    ap.add_argument("--variable", required=True)
    ap.add_argument("--lead_hours", type=int, required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--time_chunk", type=int, default=64)
    ap.add_argument("--morph_op", default="none", choices=["none","open","close","erode","dilate"])
    ap.add_argument("--morph_radius", type=int, default=0)
    ap.add_argument("--morph_iters", type=int, default=1)
    ap.add_argument("--iso_max_samples", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global crc_alpha
    crc_alpha = float(args.alpha)

    # Open local Zarrs
    da_fc = open_local(args.forecast_zarr)
    da_tr = open_local(args.truth_zarr)
    if da_fc.sizes["time"] != da_tr.sizes["time"]:
        raise RuntimeError(f"time length mismatch: forecast={da_fc.sizes['time']} truth={da_tr.sizes['time']}")

    # Area weights (lat/lon consistent across WB-2)
    lat = np.asarray(da_fc["lat"].values if "lat" in da_fc.coords else da_fc["y"].values)
    lon = np.asarray(da_fc["lon"].values if "lon" in da_fc.coords else da_fc["x"].values)
    Wts = compute_area_weights(lat, lon, normalize=True)

    # Split cal/test by time (70/30 split of local Zarr)
    T = da_fc.sizes["time"]
    n_cal = max(1, int(0.7 * T))
    da_fc_cal = da_fc.isel(time=slice(0, n_cal))
    da_tr_cal = da_tr.isel(time=slice(0, n_cal))
    da_fc_test = da_fc.isel(time=slice(n_cal, None))
    da_tr_test = da_tr.isel(time=slice(n_cal, None))

    # ---------- Calibrate: CRC ----------
    lam_grid = np.arange(0.0, 2.0001, 0.05, dtype=np.float64)
    morph_cfg = {"operation": args.morph_op, "radius": int(args.morph_radius),
                 "element": "disk", "iterations": int(args.morph_iters)}
    lam_star = calibrate_crc_streaming(da_fc_cal, da_tr_cal, float(args.threshold),
                                       lam_grid, Wts,
                                       morph_cfg["operation"], morph_cfg["radius"], morph_cfg["element"],
                                       morph_cfg["iterations"], int(args.time_chunk))

    # ---------- Calibrate: Global bump ----------
    # For global bump & morph CP calibration we can load small arrays (time streamed below),
    # but to stay consistent, we do a coarse scan by streaming (reuse CRC machinery idea).
    # We will still call the baseline calibrators on small in-memory slices if T is small.
    # Here we leverage our chunked loop to emulate calibration:
    tau_grid = np.arange(0.0, 5.0001, 0.1, dtype=np.float64)
    tau_num = np.zeros_like(tau_grid); count_t = 0; Wsum = float(np.sum(Wts))
    for yhat_blk, y_blk in iterate_chunks(da_fc_cal, da_tr_cal, int(args.time_chunk)):
        truth_mask = (y_blk >= float(args.threshold))
        for j, tau in enumerate(tau_grid):
            pred_mask = (yhat_blk >= (float(args.threshold) + tau))
            for t in range(pred_mask.shape[0]):
                tau_num[j] += np.sum(Wts[(pred_mask[t] & (~truth_mask[t]))])
        count_t += yhat_blk.shape[0]
    tau_fpa = (tau_num / Wsum) / max(1, count_t)
    ok = tau_fpa <= (crc_alpha + 1e-12)
    tau_star = float(tau_grid[np.argmax(ok)]) if np.any(ok) else float(tau_grid[-1])

    # ---------- Calibrate: Morph CP (radius) ----------
    r_candidates = np.arange(0, 16, dtype=int)
    r_num = np.zeros_like(r_candidates, dtype=np.float64); count_t = 0
    for yhat_blk, y_blk in iterate_chunks(da_fc_cal, da_tr_cal, int(args.time_chunk)):
        base = (yhat_blk >= float(args.threshold)); truth_mask = (y_blk >= float(args.threshold))
        for j, r in enumerate(r_candidates):
            pred_mask = np.empty_like(base, dtype=bool)
            for t in range(base.shape[0]):
                pred_mask[t] = morphological_filter(base[t], operation="erode",
                                                    radius=int(r), element="disk", iterations=1)
            for t in range(base.shape[0]):
                r_num[j] += np.sum(Wts[(pred_mask[t] & (~truth_mask[t]))])
        count_t += yhat_blk.shape[0]
    r_fpa = (r_num / Wsum) / max(1, count_t)
    ok = r_fpa <= (crc_alpha + 1e-12)
    r_star = int(r_candidates[np.argmax(ok)]) if np.any(ok) else int(r_candidates[-1])

    # ---------- Calibrate: Isotonic probability ----------
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(int(args.seed))
    Xs, Ys = sample_for_isotonic(da_fc_cal, da_tr_cal, float(args.threshold), int(args.iso_max_samples), rng, int(args.time_chunk))
    if Xs.size == 0:
        raise RuntimeError("No samples available to fit isotonic regression.")
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(Xs, Ys.astype(np.float64))
    p_grid = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    p_star = calibrate_prob_threshold_streaming(da_fc_cal, da_tr_cal, float(args.threshold), iso, p_grid, Wts, int(args.time_chunk))

    # ---------- Evaluate on TEST (streaming) ----------
    m_crc  = evaluate_streaming(da_fc_test, da_tr_test, float(args.threshold), Wts,
                                method="crc", params={"lambda_star": lam_star}, morph_cfg=morph_cfg, chunk_t=int(args.time_chunk))
    m_gb   = evaluate_streaming(da_fc_test, da_tr_test, float(args.threshold), Wts,
                                method="global", params={"tau_star": tau_star}, morph_cfg=morph_cfg, chunk_t=int(args.time_chunk))
    m_mcp  = evaluate_streaming(da_fc_test, da_tr_test, float(args.threshold), Wts,
                                method="morph", params={"r_star": r_star}, morph_cfg=morph_cfg, chunk_t=int(args.time_chunk))
    m_prob = evaluate_streaming(da_fc_test, da_tr_test, float(args.threshold), Wts,
                                method="prob", params={"iso": iso, "p_star": p_star}, morph_cfg=morph_cfg, chunk_t=int(args.time_chunk))

    out = {
        "variable": args.variable,
        "lead_hours": int(args.lead_hours),
        "threshold": float(args.threshold),
        "alpha": float(args.alpha),
        "time_len": int(T),
        "n_cal": int(n_cal),
        "lambda_star": float(lam_star),
        "tau_star": float(tau_star),
        "r_star": int(r_star),
        "p_star": float(p_star),
        "metrics": {
            "fieldcert_crc": m_crc,
            "global_bump": m_gb,
            "morph_cp": m_mcp,
            "prob_isotonic": m_prob,
        },
        "morph_cfg": morph_cfg,
        "paths": {"forecast_zarr": args.forecast_zarr, "truth_zarr": args.truth_zarr},
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] wrote {args.out_json}")
