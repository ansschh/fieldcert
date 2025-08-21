# scripts/fc_run_local_benchmark_enhanced.py
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
from eval import compute_area_weights  # type: ignore

# ------------------ Utilities ------------------

def open_local(path: str) -> xr.DataArray:
    """Open a local Zarr written by wb2_download_local.py.
    Standardize to dims (time, lat, lon) and drop any singleton extras.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    da = xr.open_zarr(path, consolidated=True, decode_timedelta=False)
    if isinstance(da, xr.Dataset):
        if len(da.data_vars) != 1:
            raise ValueError(f"Expected one variable in {path}, found {list(da.data_vars)}")
        da = da[list(da.data_vars)[0]]
    # rename standard coords
    if "time" not in da.coords and "forecast_time" in da.coords:
        da = da.rename({"forecast_time": "time"})
    if "latitude" in da.dims and "lat" not in da.dims:
        da = da.rename({"latitude": "lat"})
    if "longitude" in da.dims and "lon" not in da.dims:
        da = da.rename({"longitude": "lon"})
    if "y" in da.dims and "lat" not in da.dims:
        da = da.rename({"y": "lat"})
    if "x" in da.dims and "lon" not in da.dims:
        da = da.rename({"x": "lon"})
    # average over any ensemble/member dimension if present
    for ens_dim in ("number", "member", "ens", "ensemble", "realization"):
        if ens_dim in da.dims and int(da.sizes[ens_dim]) > 1:
            da = da.mean(ens_dim)
    # drop any leftover singleton dimensions (e.g., lead=1, height=1)
    da = da.squeeze(drop=True)
    # enforce (time, lat, lon) if available
    if all(k in da.dims for k in ("time", "lat", "lon")):
        da = da.transpose("time", "lat", "lon")
    else:
        da = da.transpose("time", ...)
    # chunk by time, default chunk size 64 if missing chunks
    return da.chunk({"time": da.chunks["time"][0] if hasattr(da, "chunks") and "time" in da.chunks else 64})

def select_nearest_lead_da(da: xr.DataArray, lead_hours: int) -> xr.DataArray:
    """If a lead dimension exists (prediction_timedelta/lead/step), select the nearest lead and squeeze."""
    for c in ("prediction_timedelta", "lead", "step"):
        # Only select if this is truly a dimension; some datasets carry these as aux coords.
        if c in da.dims:
            vals = da[c].values
            if np.issubdtype(vals.dtype, np.timedelta64):
                hours = (vals / np.timedelta64(1, "h")).astype(np.float64)
            else:
                try:
                    hours = vals.astype(np.float64)
                except Exception:
                    continue
            idx = int(np.argmin(np.abs(hours - float(lead_hours))))
            return da.isel({c: idx}).squeeze(drop=True)
    return da

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

def iterate_chunks(da_fc: xr.DataArray, da_tr: xr.DataArray, chunk_t: int):
    """Yield aligned numpy blocks of (yhat, y)."""
    assert da_fc.sizes["time"] == da_tr.sizes["time"], "forecast/truth times must be same length"
    T = da_fc.sizes["time"]
    for s in range(0, T, chunk_t):
        e = min(s + chunk_t, T)
        yhat = np.asarray(da_fc.isel(time=slice(s, e)).data.compute())
        y = np.asarray(da_tr.isel(time=slice(s, e)).data.compute())
        yield s, e, yhat, y

def set_area_fraction(mask: np.ndarray, w: np.ndarray) -> float:
    denom = float(np.sum(w))
    return 0.0 if denom == 0 else float(np.sum(w[mask]) / denom)

def iou(p: np.ndarray, g: np.ndarray) -> float:
    inter = np.logical_and(p, g).sum()
    uni = np.logical_or(p, g).sum()
    return 1.0 if uni == 0 else float(inter / uni)

# ------------------ Example selection ------------------

def find_topk_by_truth_area(da_tr_test: xr.DataArray, Tval: float, w_hw: np.ndarray, chunk_t: int, k: int) -> List[int]:
    """Return test-time indices (0-based) of top-K times by areal truth exceedance."""
    if k <= 0:
        return []
    heap: List[Tuple[float, int]] = []  # (area, time_idx)
    offset = 0
    for s, e, _, y_blk in iterate_chunks(da_tr_test, da_tr_test, chunk_t):
        truth_blk = (y_blk >= Tval)
        for t in range(truth_blk.shape[0]):
            area = set_area_fraction(truth_blk[t], w_hw)
            time_idx = offset + t
            if len(heap) < k:
                heapq.heappush(heap, (area, time_idx))
            else:
                if area > heap[0][0]:
                    heapq.heapreplace(heap, (area, time_idx))
        offset += (e - s)
    # return in descending area
    return [idx for _, idx in sorted(heap, key=lambda z: -z[0])]

# ------------------ Evaluate (with optional JSONL logging + examples) ------------------

def evaluate_streaming_all(
    da_fc_test: xr.DataArray,
    da_tr_test: xr.DataArray,
    Tval: float,
    w_hw: np.ndarray,
    params: Dict[str, Any],
    morph_cfg: Dict[str, Any],
    chunk_t: int,
    log_jsonl: Optional[str] = None,
    examples_out: Optional[str] = None,
    example_indices: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate CRC / Global / Morph / Prob (isotonic) streaming.
    params: dict with keys
        lambda_star, tau_star, r_star, iso (callable .predict that maps flat array -> prob), p_star
    If log_jsonl is provided, writes one JSON line per time with per-method metrics.
    If examples_out + example_indices provided, dumps NPZs for those times with fields and masks.
    """
    os.makedirs(os.path.dirname(log_jsonl), exist_ok=True) if log_jsonl else None
    if examples_out:
        os.makedirs(examples_out, exist_ok=True)
    f_json = open(log_jsonl, "w") if log_jsonl else None

    sums = {m: {"fpa": 0.0, "fna": 0.0, "iou": 0.0, "n": 0} for m in ["fieldcert_crc","global_bump","morph_cp","prob_isotonic"]}
    Wsum = float(np.sum(w_hw))
    offset = 0

    for s, e, yhat_blk, y_blk in iterate_chunks(da_fc_test, da_tr_test, chunk_t):
        Tblk = yhat_blk.shape[0]
        truth_mask = (y_blk >= Tval)

        # predictions for each method
        margins = grad_mag_norm(yhat_blk)
        pred_crc = (yhat_blk >= (Tval + float(params["lambda_star"]) * margins))
        pred_gb = (yhat_blk >= (Tval + float(params["tau_star"])))
        base = (yhat_blk >= Tval)
        pred_morph = np.empty_like(base, dtype=bool)
        for t in range(base.shape[0]):
            pred_morph[t] = morphological_filter(base[t], operation="erode",
                                                 radius=int(params["r_star"]), element="disk", iterations=1)
        iso = params["iso"]; p_star = float(params["p_star"])
        prob = iso.predict(yhat_blk.ravel()).reshape(yhat_blk.shape)
        pred_prob = (prob >= p_star)

        # optional symmetric morph for CRC only (keeps truth aligned to filtering)
        if morph_cfg["operation"] != "none" and morph_cfg["radius"] > 0 and morph_cfg["iterations"] > 0:
            pm = np.empty_like(pred_crc, dtype=bool); gm = np.empty_like(truth_mask, dtype=bool)
            for t in range(pred_crc.shape[0]):
                pm[t], gm[t] = symmetric_filter(pred_crc[t], truth_mask[t],
                                                morph_cfg["operation"], int(morph_cfg["radius"]),
                                                morph_cfg["element"], int(morph_cfg["iterations"]))
            pred_crc = pm; truth_mask_crc = gm
        else:
            truth_mask_crc = truth_mask

        # per-time metrics (+ optional logging and example dumps)
        for t in range(Tblk):
            g = truth_mask[t]; g_crc = truth_mask_crc[t]
            
            # Compute metrics for all methods
            metrics_t = {}
            for name, p in [
                ("fieldcert_crc", pred_crc[t]),
                ("global_bump",   pred_gb[t]),
                ("morph_cp",      pred_morph[t]),
                ("prob_isotonic", pred_prob[t]),
            ]:
                truth_ref = g_crc if name == "fieldcert_crc" else g
                fpa = 0.0 if Wsum == 0 else float(np.sum(w_hw[(p & (~truth_ref))]) / Wsum)
                fna = 0.0 if Wsum == 0 else float(np.sum(w_hw[((~p) & truth_ref)]) / Wsum)
                j   = iou(p, truth_ref)
                sums[name]["fpa"] += fpa
                sums[name]["fna"] += fna
                sums[name]["iou"] += j
                sums[name]["n"] += 1
                metrics_t[name] = {"fpa": fpa, "fna": fna, "iou": j}
            
            # JSONL log?
            if f_json is not None:
                rec = {
                    "time_index": offset + t,
                    "threshold": float(Tval),
                    "metrics": metrics_t
                }
                f_json.write(json.dumps(rec) + "\n")

            # example dump?
            if examples_out and example_indices and (offset + t) in example_indices:
                out_npz = os.path.join(examples_out, f"example_t{offset+t:06d}.npz")
                np.savez_compressed(
                    out_npz,
                    yhat=yhat_blk[t],
                    truth=y_blk[t],
                    truth_mask=g,
                    pred_crc=pred_crc[t],
                    pred_global=pred_gb[t],
                    pred_morph=pred_morph[t],
                    prob=prob[t],
                    pred_prob=pred_prob[t],
                )

        offset += Tblk

    # aggregate means
    out = {}
    for name, agg in sums.items():
        n = max(1, agg["n"])
        out[name] = {
            "fpa": float(agg["fpa"] / n),
            "fna": float(agg["fna"] / n),
            "iou": float(agg["iou"] / n),
        }
    if f_json is not None:
        f_json.close()
    return out

# ------------------ Main ------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chunked, local benchmark: FieldCert + baselines (with JSON logging).")
    ap.add_argument("--forecast_zarr", required=True)
    ap.add_argument("--truth_zarr", required=True)
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
    # NEW: logging & examples
    ap.add_argument("--log_time_jsonl", default=None,
                    help="If set, write per-time metrics JSONL here (one line per timestamp).")
    ap.add_argument("--examples_topk", type=int, default=0,
                    help="If >0, dump NPZs for top-K truth-area times in the TEST split.")
    ap.add_argument("--examples_dir", default=None,
                    help="Directory to store example NPZs (required if --examples_topk>0)")
    args = ap.parse_args()

    # Open local Zarrs
    da_fc = open_local(args.forecast_zarr)
    # If a lead dim slipped through in the saved Zarr, select nearest and squeeze
    da_fc = select_nearest_lead_da(da_fc, int(args.lead_hours))
    da_tr = open_local(args.truth_zarr)
    if da_fc.sizes["time"] != da_tr.sizes["time"]:
        raise RuntimeError(f"time length mismatch: forecast={da_fc.sizes['time']} truth={da_tr.sizes['time']}")

    # Area weights (lat/lon consistent across WB-2)
    if "lat" in da_fc.coords:
        lat_vals = np.asarray(da_fc["lat"].values)
    elif "latitude" in da_fc.coords:
        lat_vals = np.asarray(da_fc["latitude"].values)
    else:
        lat_vals = np.asarray(da_fc["y"].values)

    if "lon" in da_fc.coords:
        lon_vals = np.asarray(da_fc["lon"].values)
    elif "longitude" in da_fc.coords:
        lon_vals = np.asarray(da_fc["longitude"].values)
    else:
        lon_vals = np.asarray(da_fc["x"].values)

    # Build a minimal dataset with standard names expected by compute_area_weights
    dummy_ds = xr.Dataset(coords={
        "latitude": ("latitude", lat_vals),
        "longitude": ("longitude", lon_vals),
    })
    Wts = compute_area_weights(dummy_ds, lat_name="latitude", normalize=True)
    # Ensure orientation matches spatial dims (H, W)
    if "lat" in da_fc.dims and "lon" in da_fc.dims:
        H, W = int(da_fc.sizes["lat"]), int(da_fc.sizes["lon"])
    elif "y" in da_fc.dims and "x" in da_fc.dims:
        H, W = int(da_fc.sizes["y"]), int(da_fc.sizes["x"])
    elif "latitude" in da_fc.dims and "longitude" in da_fc.dims:
        H, W = int(da_fc.sizes["latitude"]), int(da_fc.sizes["longitude"])
    else:
        # fallback to using a time slice but prefer dims when possible
        shp = da_fc.isel(time=0).shape
        if len(shp) < 2:
            raise ValueError(f"Cannot infer spatial dims for area weights from shape {shp}")
        H, W = int(shp[-2]), int(shp[-1])
    if Wts.shape != (H, W):
        if Wts.T.shape == (H, W):
            Wts = Wts.T
        else:
            raise ValueError(f"Area weights shape {Wts.shape} does not match spatial dims {(H, W)}; da dims={da_fc.dims}")

    # 70/30 split
    T = da_fc.sizes["time"]
    n_cal = max(1, int(0.7 * T))
    da_fc_cal = da_fc.isel(time=slice(0, n_cal))
    da_tr_cal = da_tr.isel(time=slice(0, n_cal))
    da_fc_test = da_fc.isel(time=slice(n_cal, None))
    da_tr_test = da_tr.isel(time=slice(n_cal, None))

    # ---------- Calibrate ----------
    lam_grid = np.arange(0.0, 2.0001, 0.05, dtype=np.float64)
    morph_cfg = {"operation": args.morph_op, "radius": int(args.morph_radius),
                 "element": "disk", "iterations": int(args.morph_iters)}

    # CRC calibration (stream FPA over lambda grid)
    tau_grid = np.arange(0.0, 5.0001, 0.1, dtype=np.float64)
    r_candidates = np.arange(0, 16, dtype=int)
    alpha = float(args.alpha)
    Tval = float(args.threshold)
    rng = np.random.default_rng(int(args.seed))
    Wsum = float(np.sum(Wts))

    # CRC lambda*: streaming scan
    num = np.zeros_like(lam_grid); count = 0
    for s, e, yhat_blk, y_blk in iterate_chunks(da_fc_cal, da_tr_cal, int(args.time_chunk)):
        margins = grad_mag_norm(yhat_blk)
        truth = (y_blk >= Tval)
        for j, lam in enumerate(lam_grid):
            pred = (yhat_blk >= (Tval + lam * margins))
            for t in range(pred.shape[0]):
                num[j] += np.sum(Wts[(pred[t] & (~truth[t]))])
        count += (e - s)
    fpa_mean = (num / Wsum) / max(1, count)
    ok = fpa_mean <= (alpha + 1e-12)
    lambda_star = float(lam_grid[np.argmax(ok)]) if np.any(ok) else float(lam_grid[-1])

    # Global bump tau*
    tau_num = np.zeros_like(tau_grid); count = 0
    for s, e, yhat_blk, y_blk in iterate_chunks(da_fc_cal, da_tr_cal, int(args.time_chunk)):
        truth = (y_blk >= Tval)
        for j, tau in enumerate(tau_grid):
            pred = (yhat_blk >= (Tval + tau))
            for t in range(pred.shape[0]):
                tau_num[j] += np.sum(Wts[(pred[t] & (~truth[t]))])
        count += (e - s)
    tau_fpa = (tau_num / Wsum) / max(1, count)
    ok = tau_fpa <= (alpha + 1e-12)
    tau_star = float(tau_grid[np.argmax(ok)]) if np.any(ok) else float(tau_grid[-1])

    # Morph radius r*
    r_num = np.zeros_like(r_candidates, dtype=np.float64); count = 0
    for s, e, yhat_blk, y_blk in iterate_chunks(da_fc_cal, da_tr_cal, int(args.time_chunk)):
        base = (yhat_blk >= Tval); truth = (y_blk >= Tval)
        for j, r in enumerate(r_candidates):
            pred = np.empty_like(base, dtype=bool)
            for t in range(base.shape[0]):
                pred[t] = morphological_filter(base[t], operation="erode", radius=int(r), element="disk", iterations=1)
            for t in range(pred.shape[0]):
                r_num[j] += np.sum(Wts[(pred[t] & (~truth[t]))])
        count += (e - s)
    r_fpa = (r_num / Wsum) / max(1, count)
    ok = r_fpa <= (alpha + 1e-12)
    r_star = int(r_candidates[np.argmax(ok)]) if np.any(ok) else int(r_candidates[-1])

    # Isotonic probability fit (sampled)
    from sklearn.isotonic import IsotonicRegression
    # sample up to iso_max_samples uniformly over time×space
    Xs: list[np.ndarray] = []; Ys: list[np.ndarray] = []
    remain = int(args.iso_max_samples)
    for _, _, yhat_blk, y_blk in iterate_chunks(da_fc_cal, da_tr_cal, int(args.time_chunk)):
        x = yhat_blk.ravel(); ybin = (y_blk >= Tval).astype(np.int8).ravel()
        n = x.shape[0]; take = min(remain, n)
        if take > 0:
            idx = rng.choice(n, size=take, replace=False)
            Xs.append(x[idx]); Ys.append(ybin[idx]); remain -= take
        if remain <= 0:
            break
    if len(Xs) == 0:
        raise RuntimeError("No samples available to fit isotonic regression.")
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(np.concatenate(Xs), np.concatenate(Ys).astype(np.float64))
    # p* calibration (grid)
    p_grid = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    num = np.zeros_like(p_grid); count = 0
    for _, _, yhat_blk, y_blk in iterate_chunks(da_fc_cal, da_tr_cal, int(args.time_chunk)):
        prob = iso.predict(yhat_blk.ravel()).reshape(yhat_blk.shape)
        truth = (y_blk >= Tval)
        for j, thr in enumerate(p_grid):
            pred = (prob >= thr)
            for t in range(pred.shape[0]):
                num[j] += np.sum(Wts[(pred[t] & (~truth[t]))])
        count += prob.shape[0]
    fpa = (num / Wsum) / max(1, count)
    ok = fpa <= (alpha + 1e-12)
    p_star = float(p_grid[np.argmax(ok)]) if np.any(ok) else float(p_grid[-1])

    # ---------- Examples: pick top-K test times by truth area ----------
    example_indices = []
    if args.examples_topk > 0:
        if not args.examples_dir:
            raise ValueError("--examples_dir must be set when --examples_topk>0")
        example_indices = find_topk_by_truth_area(da_tr_test, Tval, Wts, int(args.time_chunk), int(args.examples_topk))

    # ---------- Evaluate on TEST (with JSONL + examples) ----------
    metrics = evaluate_streaming_all(
        da_fc_test, da_tr_test, Tval, Wts,
        params={"lambda_star": lambda_star, "tau_star": tau_star, "r_star": r_star, "iso": iso, "p_star": p_star},
        morph_cfg=morph_cfg,
        chunk_t=int(args.time_chunk),
        log_jsonl=args.log_time_jsonl,
        examples_out=args.examples_dir,
        example_indices=example_indices,
    )

    out = {
        "variable": args.variable,
        "lead_hours": int(args.lead_hours),
        "threshold": float(args.threshold),
        "alpha": float(args.alpha),
        "time_len": int(T),
        "n_cal": int(n_cal),
        "lambda_star": float(lambda_star),
        "tau_star": float(tau_star),
        "r_star": int(r_star),
        "p_star": float(p_star),
        "metrics": metrics,
        "morph_cfg": morph_cfg,
        "paths": {"forecast_zarr": args.forecast_zarr, "truth_zarr": args.truth_zarr},
        "logging": {"per_time_jsonl": args.log_time_jsonl, "examples_dir": args.examples_dir, "examples_indices": example_indices},
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] wrote {args.out_json}")
    if args.log_time_jsonl:
        print(f"[OK] per-time metrics JSONL: {args.log_time_jsonl}")
    if args.examples_topk > 0:
        print(f"[OK] example NPZs: {args.examples_dir} (top-K indices: {example_indices})")
