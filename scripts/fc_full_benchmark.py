# scripts/fc_full_benchmark.py
from __future__ import annotations

import os, sys, json, argparse, csv
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# Make src/ importable regardless of working dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import xarray as xr  # noqa: E402

# Our modules
from calibration import (
    CRCCalibrator, CRCSettings, build_margin_field,
)
from calibration.morphology import threshold_bump_mask, symmetric_filter  # noqa: E402
from calibration.regimes import assign_blocks  # noqa: E402
from baselines import (
    calibrate_global_bump_by_fpa, apply_global_bump,
    calibrate_morph_radius_by_fpa, apply_morph_radius,
    calibrate_pixelwise_delta, apply_pixelwise_delta,
)
from eval import (
    area_weights_from_latlon,  # noqa: E402
    brier_score, reliability_curve,  # for prob baseline report
)

# ---------- Provider registry (WB-2 public bucket paths) ----------

WB2_ERA5 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
WB2_ENS_MEAN = "gs://weatherbench2/datasets/ens/2018-2022-240x121_equiangular_with_poles_conservative_mean.zarr"
# Full ensemble path (not used by default here to keep it tractable)
WB2_IFS_ENS = "gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr"

# GraphCast (WB-2 provides winter windows for these years)
def _graphcast_path(year: int) -> str:
    # Valid only for 2018 and 2020 in WB-2
    if year == 2018:
        return "gs://weatherbench2/datasets/graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr"
    if year == 2020:
        return "gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr"
    raise ValueError("GraphCast WB-2 paths are provided for years 2018 and 2020 only.")

# NeuralGCM deterministic (2020)
WB2_NEURALGCM_2020 = "gs://weatherbench2/datasets/neuralgcm_deterministic/2020-240x121_equiangular_with_poles_conservative.zarr"


# ---------- Utilities ----------

def _open_zarr(path: str, chunks: Any = "auto") -> xr.Dataset:
    storage_options = {"token": "anon"} if path.startswith("gs://") else None
    return xr.open_zarr(path, storage_options=storage_options, chunks=chunks)


def _select_lead(da: xr.DataArray, lead_hours: int) -> xr.DataArray:
    for c in ("prediction_timedelta", "lead", "step"):
        if c in da.coords:
            return da.sel({c: np.timedelta64(int(lead_hours), "h")})
    raise KeyError(f"Lead coordinate not found in {list(da.coords)}")


def _std_coords(da: xr.DataArray) -> xr.DataArray:
    """Ensure dims like (time, lat, lon) and consistent names."""
    if "time" not in da.coords:
        if "forecast_time" in da.coords:
            da = da.rename({"forecast_time": "time"})
        else:
            raise KeyError("No 'time' coordinate present.")
    # latitude/longitude to lat/lon
    if "latitude" in da.dims and "lat" not in da.dims:
        da = da.rename({"latitude": "lat"})
    if "longitude" in da.dims and "lon" not in da.dims:
        da = da.rename({"longitude": "lon"})
    # transpose so time first
    dims = list(da.dims)
    order = ["time"]
    order += [d for d in ("lat", "y") if d in dims]
    order += [d for d in ("lon", "x") if d in dims]
    return da.transpose(*order)


def _year_bounds_mask(times: np.ndarray, y0: int, y1: int) -> np.ndarray:
    years = (times.astype("datetime64[Y]").astype(int) + 1970)
    return (years >= y0) & (years <= y1)


def _align_truth_by_valid_time(
    ds_obs: xr.Dataset,
    var_obs: str,
    valid_times: np.ndarray,
    tolerance_hours: int = 3,
) -> xr.DataArray:
    da_obs = ds_obs[var_obs]
    if "time" not in da_obs.coords and "forecast_time" in da_obs.coords:
        da_obs = da_obs.rename({"forecast_time": "time"})
    if "latitude" in da_obs.dims and "lat" not in da_obs.dims:
        da_obs = da_obs.rename({"latitude": "lat"})
    if "longitude" in da_obs.dims and "lon" not in da_obs.dims:
        da_obs = da_obs.rename({"longitude": "lon"})
    da_obs = da_obs.transpose("time", ...)
    tol = np.timedelta64(int(tolerance_hours), "h")
    valid_da = xr.DataArray(valid_times, dims=("time",))
    return da_obs.sel(time=valid_da, method="nearest", tolerance=tol)


def _deterministic_prob_isotonic(
    yhat_cal: np.ndarray, y_cal: np.ndarray, threshold: float,
    max_samples: int = 2_000_000, random_state: int = 42,
) -> Tuple[Any, Dict[str, float]]:
    """
    Fit isotonic regression p(event|yhat) on a subsample.
    Returns the fitted model and basic diagnostics (%positives etc.).
    """
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(random_state)
    T, H, W = yhat_cal.shape
    x = yhat_cal.ravel()
    y = (y_cal >= float(threshold)).astype(np.int8).ravel()
    N = x.shape[0]
    if N > max_samples:
        # stratified-ish: sample half positives, half negatives if possible
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        n_pos = min(len(idx_pos), max_samples // 2)
        n_neg = max_samples - n_pos
        sel_pos = rng.choice(idx_pos, size=n_pos, replace=False) if len(idx_pos) > 0 else np.array([], dtype=int)
        sel_neg = rng.choice(idx_neg, size=n_neg, replace=False)
        idx = np.concatenate([sel_pos, sel_neg])
        x, y = x[idx], y[idx]
    # Fit isotonic (increasing with yhat)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(x, y.astype(np.float64))
    diag = {"frac_pos": float(np.mean(y))}
    return iso, diag


def _apply_isotonic(iso: Any, yhat: np.ndarray) -> np.ndarray:
    """Predict p(event) for each gridcell/time given deterministic yhat via isotonic mapping."""
    x = yhat.ravel()
    p = iso.predict(x).astype(np.float64)
    return p.reshape(yhat.shape)


def _evaluate_sets_once(
    pred_mask: np.ndarray,
    truth_mask: np.ndarray,
    weights_hw: Optional[np.ndarray],
) -> Dict[str, float]:
    """Fast inline evaluation of FPA/FNA/IoU (avoids importing whole eval.sets here)."""
    w = weights_hw
    T = pred_mask.shape[0]
    fpa_t, fna_t, iou_t = [], [], []
    if w is None:
        w = np.ones(pred_mask.shape[1:], dtype=np.float64)
    for t in range(T):
        p = pred_mask[t].astype(bool)
        g = truth_mask[t].astype(bool)
        W = float(np.sum(w))
        if W <= 0:
            fpa = fna = 0.0
        else:
            fpa = float(np.sum(w[(p & (~g))]) / W)
            fna = float(np.sum(w[((~p) & g)]) / W)
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        iou = 1.0 if union == 0 else float(inter / union)
        fpa_t.append(fpa); fna_t.append(fna); iou_t.append(iou)
    return {
        "fpa": float(np.mean(fpa_t)),
        "fna": float(np.mean(fna_t)),
        "iou": float(np.mean(iou_t)),
    }


# ---------- Forecast readers per provider ----------

def _read_provider_timeseries(
    provider: str,
    variable: str,
    lead_hours: int,
    years_cal: Tuple[int, int],
    years_test: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (yhat_cal, y_cal, times_cal, yhat_test, y_test, times_test).
    All arrays are (T,H,W). Raises if the variable is missing.
    """
    # Open ERA5 once (truth)
    ds_obs = _open_zarr(WB2_ERA5, chunks="auto")
    if variable not in ds_obs:
        raise KeyError(f"Variable '{variable}' not present in ERA5 dataset.")
    # Area weights will be extracted after forecast load

    if provider == "ifs_mean":
        ds_fc = _open_zarr(WB2_ENS_MEAN, chunks="auto")
        if variable not in ds_fc:
            raise KeyError(f"Variable '{variable}' missing in ENS-mean dataset.")
        da = ds_fc[variable]
        da = _select_lead(da, lead_hours)
        da = _std_coords(da)
        times = da["time"].values.astype("datetime64[ns]")
        valid_times = times + np.timedelta64(int(lead_hours), "h")
        # split by year bounds on valid times (so truth lines up)
        cal_mask = _year_bounds_mask(valid_times, years_cal[0], years_cal[1])
        test_mask = _year_bounds_mask(valid_times, years_test[0], years_test[1])
        if not np.any(cal_mask): raise RuntimeError("No cal times for ENS-mean with requested years.")
        if not np.any(test_mask): raise RuntimeError("No test times for ENS-mean with requested years.")
        da_cal = da.sel(time=da["time"].values[cal_mask])
        da_test = da.sel(time=da["time"].values[test_mask])
        vt_cal = valid_times[cal_mask]; vt_test = valid_times[test_mask]
        obs_cal = _align_truth_by_valid_time(ds_obs, variable, vt_cal, tolerance_hours=3)
        obs_test = _align_truth_by_valid_time(ds_obs, variable, vt_test, tolerance_hours=3)
        yhat_cal = da_cal.values; y_cal = obs_cal.values
        yhat_test = da_test.values; y_test = obs_test.values
        return yhat_cal, y_cal, vt_cal, yhat_test, y_test, vt_test

    elif provider == "graphcast":
        # Two windows exist: 2018 and 2020
        def read_year(year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            path = _graphcast_path(year)
            ds_gc = _open_zarr(path, chunks="auto")
            if variable not in ds_gc:
                raise KeyError(f"Variable '{variable}' missing in GraphCast {year}.")
            da = ds_gc[variable]
            da = _select_lead(da, lead_hours)
            da = _std_coords(da)
            times = da["time"].values.astype("datetime64[ns]")
            valid_times = times + np.timedelta64(int(lead_hours), "h")
            obs = _align_truth_by_valid_time(ds_obs, variable, valid_times, tolerance_hours=3)
            return da.values, obs.values, valid_times

        # cal: 2018, test: 2020 (WB-2 provided subsets)
        yhat_c, y_c, t_c = read_year(2018)
        yhat_t, y_t, t_t = read_year(2020)
        return yhat_c, y_c, t_c, yhat_t, y_t, t_t

    elif provider == "neuralgcm":
        # Only 2020 provided. Split by time proportion.
        ds_ng = _open_zarr(WB2_NEURALGCM_2020, chunks="auto")
        if variable not in ds_ng:
            raise KeyError(f"Variable '{variable}' missing in NeuralGCM 2020.")
        da = ds_ng[variable]
        da = _select_lead(da, lead_hours)
        da = _std_coords(da)
        times = da["time"].values.astype("datetime64[ns]")
        valid_times = times + np.timedelta64(int(lead_hours), "h")
        obs = _align_truth_by_valid_time(ds_obs, variable, valid_times, tolerance_hours=3)
        yhat = da.values; y = obs.values
        # 70/30 split
        n = yhat.shape[0]
        n_cal = max(1, int(0.7 * n))
        return yhat[:n_cal], y[:n_cal], valid_times[:n_cal], yhat[n_cal:], y[n_cal:], valid_times[n_cal:]

    elif provider == "persistence":
        # Forecast = ERA5 at time (valid_time - lead)
        # Build target valid times by choosing an initialization stream; use ENS mean init times for consistency.
        ds_fc = _open_zarr(WB2_ENS_MEAN, chunks="auto")
        if variable not in ds_fc:
            raise KeyError(f"Variable '{variable}' missing in ENS-mean dataset (used for init times).")
        da_init = _std_coords(_select_lead(ds_fc[variable], lead_hours))
        init_times = da_init["time"].values.astype("datetime64[ns]")
        valid_times = init_times + np.timedelta64(int(lead_hours), "h")
        # Split by valid-time years (match ENS split)
        cal_mask = _year_bounds_mask(valid_times, years_cal[0], years_cal[1])
        test_mask = _year_bounds_mask(valid_times, years_test[0], years_test[1])
        if not np.any(cal_mask): raise RuntimeError("No cal times for persistence with requested years.")
        if not np.any(test_mask): raise RuntimeError("No test times for persistence with requested years.")

        # Build persistence yhat by sampling ERA5 at t = valid_time - lead
        ds_e = _open_zarr(WB2_ERA5, chunks="auto")
        if variable not in ds_e:
            raise KeyError(f"Variable '{variable}' not in ERA5 for persistence.")
        da_e = ds_e[variable]
        if "time" not in da_e.coords and "forecast_time" in da_e.coords:
            da_e = da_e.rename({"forecast_time": "time"})
        # Align truth at valid time, and yhat at valid_time - lead
        vt_cal = valid_times[cal_mask]; vt_test = valid_times[test_mask]
        obs_cal = _align_truth_by_valid_time(ds_e, variable, vt_cal, tolerance_hours=3)
        obs_test = _align_truth_by_valid_time(ds_e, variable, vt_test, tolerance_hours=3)
        # Build yhat by shifting
        t0_cal = vt_cal - np.timedelta64(int(lead_hours), "h")
        t0_test = vt_test - np.timedelta64(int(lead_hours), "h")
        yhat_cal = _align_truth_by_valid_time(ds_e, variable, t0_cal, tolerance_hours=3).values
        yhat_test = _align_truth_by_valid_time(ds_e, variable, t0_test, tolerance_hours=3).values
        return yhat_cal, obs_cal.values, vt_cal, yhat_test, obs_test.values, vt_test

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------- Main benchmark routine ----------

def run_one_combination(
    out_dir: str,
    provider: str,
    variable: str,
    lead_hours: int,
    threshold: float,
    alpha: float,
    years_cal: Tuple[int, int],
    years_test: Tuple[int, int],
    morph_cfg: Dict[str, Any],
    prob_max_samples: int = 2_000_000,
) -> Dict[str, Any]:
    """
    Runs all UQ methods for a (provider, variable, lead) with given split/alpha.
    Saves a JSONL file incrementally and returns the summary dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"results_{provider}_{variable}_{lead_hours}h.jsonl")
    print(f"\n=== Provider={provider} var={variable} lead={lead_hours}h T={threshold} α={alpha} ===")

    # Load timeseries
    yhat_cal, y_cal, t_cal, yhat_test, y_test, t_test = _read_provider_timeseries(
        provider, variable, lead_hours, years_cal, years_test
    )
    if yhat_cal.size == 0 or yhat_test.size == 0:
        raise RuntimeError("Empty arrays after selection; check dataset paths/years.")

    # Area weights (lat/lon from forecast dataset via one slice)
    H, W = yhat_cal.shape[1], yhat_cal.shape[2]
    # Build weights from ERA5 lat/lon grid (consistent across WB-2 at 240x121)
    # Fetch lat/lon from ERA5 directly:
    ds_obs = _open_zarr(WB2_ERA5, chunks="auto")
    da_obs = ds_obs[variable]
    if "latitude" in da_obs.dims and "lat" not in da_obs.dims:
        da_obs = da_obs.rename({"latitude": "lat"})
    if "longitude" in da_obs.dims and "lon" not in da_obs.dims:
        da_obs = da_obs.rename({"longitude": "lon"})
    lat = np.asarray(da_obs["lat"].values); lon = np.asarray(da_obs["lon"].values)
    Wts = area_weights_from_latlon(lat, lon, normalize=True)

    # Truth exceedance masks
    Tval = float(threshold)
    truth_cal_mask = (y_cal >= Tval)
    truth_test_mask = (y_test >= Tval)

    # ------------------- UQ #1: FieldCert / CRC -------------------
    lam_grid = np.arange(0.0, 2.0001, 0.05, dtype=np.float64)
    crc_settings = CRCSettings(
        alpha=float(alpha),
        lambda_grid=lam_grid,
        slack_B=1.0,
        loss_type="fpa",
        morph_operation=morph_cfg["operation"],
        morph_radius=int(morph_cfg["radius"]),
        morph_element=morph_cfg["element"],
        morph_iterations=int(morph_cfg["iterations"]),
    )
    crc = CRCCalibrator(settings=crc_settings)
    block_ids_cal = assign_blocks(t_cal, block="week")
    margins_cal = build_margin_field(yhat_cal, method="grad_mag", normalize=True)

    crc_res = crc.fit_for_regime(
        preds=yhat_cal,
        truths=y_cal,
        threshold=Tval,
        margins=margins_cal,
        block_ids=block_ids_cal,
        spatial_weights=Wts,
    )
    lam_star = float(crc_res.lambda_star)
    # Apply on test
    margins_test = build_margin_field(yhat_test, method="grad_mag", normalize=True)
    pred_crc = threshold_bump_mask(yhat_test, Tval, margins_test, lam_star)
    if morph_cfg["operation"] != "none" and morph_cfg["radius"] > 0 and morph_cfg["iterations"] > 0:
        pm = np.empty_like(pred_crc, dtype=bool); gm = np.empty_like(truth_test_mask, dtype=bool)
        for i in range(pred_crc.shape[0]):
            pm[i], gm[i] = symmetric_filter(
                pred_crc[i], truth_test_mask[i],
                morph_cfg["operation"], int(morph_cfg["radius"]), morph_cfg["element"], int(morph_cfg["iterations"])
            )
        pred_crc, truth_test_eval = pm, gm
    else:
        truth_test_eval = truth_test_mask
    m_crc = _evaluate_sets_once(pred_crc, truth_test_eval, Wts)

    # ------------------- UQ #2: Global bump -----------------------
    tau_star, _, _ = calibrate_global_bump_by_fpa(
        yhat_cal, y_cal, Tval, alpha=float(alpha), spatial_weights=Wts
    )
    pred_global = apply_global_bump(yhat_test, Tval, tau_star)
    m_global = _evaluate_sets_once(pred_global, truth_test_mask, Wts)

    # ------------------- UQ #3: Morphological CP (erosion) --------
    r_star, _, _ = calibrate_morph_radius_by_fpa(
        yhat_cal, y_cal, Tval, alpha=float(alpha), spatial_weights=Wts
    )
    pred_morph = apply_morph_radius(yhat_test, Tval, int(r_star))
    m_morph = _evaluate_sets_once(pred_morph, truth_test_mask, Wts)

    # ------------------- UQ #4: Pixelwise split-CP ----------------
    delta = calibrate_pixelwise_delta(yhat_cal, y_cal, Tval, alpha_pixel=float(alpha))
    pred_pixelcp = apply_pixelwise_delta(yhat_test, Tval, delta)
    m_pixelcp = _evaluate_sets_once(pred_pixelcp, truth_test_mask, Wts)

    # ------------------- UQ #5: Prob. baseline (isotonic) ---------
    iso, diag = _deterministic_prob_isotonic(yhat_cal, y_cal, Tval, max_samples=prob_max_samples)
    prob_cal = _apply_isotonic(iso, yhat_cal)
    prob_test = _apply_isotonic(iso, yhat_test)
    # Calibrate probability threshold on cal to meet FPA target
    from baselines.emos_eval import prob_to_set_calibrate_threshold, evaluate_prob_event_baseline
    p_star, _, _ = prob_to_set_calibrate_threshold(prob_cal, truth_cal_mask, alpha=float(alpha), spatial_weights=Wts)
    met_prob = evaluate_prob_event_baseline(prob_test, y_test, Tval, spatial_weights=Wts, prob_threshold=float(p_star))

    # ------------------- Persist / dump results --------------------
    all_metrics = {
        "provider": provider,
        "variable": variable,
        "lead_hours": int(lead_hours),
        "threshold": float(Tval),
        "alpha": float(alpha),
        "cal_years": list(years_cal),
        "test_years": list(years_test),
        "n_cal": int(yhat_cal.shape[0]),
        "n_test": int(yhat_test.shape[0]),
        "fieldcert": {"lambda_star": lam_star, **m_crc},
        "global_bump": {"tau_star": float(tau_star), **m_global},
        "morph_cp": {"r_star": int(r_star), **m_morph},
        "pixel_cp": {"delta_stats": {"mean": float(np.mean(delta)), "p95": float(np.percentile(delta, 95.0))}, **m_pixelcp},
        "prob_isotonic": {"p_star": float(p_star), "brier": float(met_prob.get("brier", np.nan)),
                          "fpa": float(met_prob.get("fpa", np.nan)),
                          "fna": float(met_prob.get("fna", np.nan))},
        "prob_diag": diag,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(all_metrics) + "\n")

    print(f"[OK] {provider}/{variable}/{lead_hours}h :: "
          f"FPA (α={alpha}) → CRC={m_crc['fpa']:.3f}, Global={m_global['fpa']:.3f}, "
          f"Morph={m_morph['fpa']:.3f}, PixelCP={m_pixelcp['fpa']:.3f}, ProbIso={all_metrics['prob_isotonic']['fpa']:.3f}")
    return all_metrics


def _default_thresholds() -> Dict[str, List[float]]:
    return {
        "10m_wind_speed": [15.0, 20.0, 25.0],
        "total_precipitation_24hr": [10.0, 20.0, 40.0],
    }


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _summary_csv(all_rows: List[Dict[str, Any]], out_csv: str) -> None:
    """
    Flatten the nested dicts and write a tidy CSV with one row per (provider, var, lead, thr, method).
    """
    rows: List[Dict[str, Any]] = []
    for d in all_rows:
        common = {
            "provider": d["provider"],
            "variable": d["variable"],
            "lead_hours": d["lead_hours"],
            "threshold": d["threshold"],
            "alpha": d["alpha"],
            "cal_years": "-".join(map(str, d["cal_years"])),
            "test_years": "-".join(map(str, d["test_years"])),
            "n_cal": d["n_cal"],
            "n_test": d["n_test"],
        }
        for method in ["fieldcert", "global_bump", "morph_cp", "pixel_cp", "prob_isotonic"]:
            m = d[method]
            row = dict(common)
            row["method"] = method
            # core metrics
            row["fpa"] = m.get("fpa")
            row["fna"] = m.get("fna")
            row["iou"] = m.get("iou")
            # extras
            for extra in ("lambda_star", "tau_star", "r_star", "p_star", "brier"):
                if extra in m:
                    row[extra] = m[extra]
            rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] Wrote summary CSV: {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Full FieldCert benchmark across providers/UQ methods.")
    ap.add_argument("--providers", default="ifs_mean,graphcast,neuralgcm,persistence",
                    help="Comma-separated providers: ifs_mean, graphcast, neuralgcm, persistence")
    ap.add_argument("--variables", default="10m_wind_speed,total_precipitation_24hr",
                    help="Comma-separated variables to evaluate")
    ap.add_argument("--leads", default="24,48,72", help="Comma-separated integer leads in hours")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--outdir", default="/workspace/results/full_benchmark")
    # Year splits (defaults reasonable per provider)
    ap.add_argument("--ens_years_cal", default="2019-2021")
    ap.add_argument("--ens_years_test", default="2022-2022")
    ap.add_argument("--pers_years_cal", default="2019-2021")
    ap.add_argument("--pers_years_test", default="2022-2022")
    # GraphCast: fixed to (2018 cal, 2020 test); NeuralGCM: internal 70/30 in 2020
    # Morphology knobs applied to CRC only (symmetrically on pred/true)
    ap.add_argument("--morph_op", default="none", choices=["none","open","close","erode","dilate"])
    ap.add_argument("--morph_radius", type=int, default=0)
    ap.add_argument("--morph_iters", type=int, default=1)
    ap.add_argument("--prob_max_samples", type=int, default=2_000_000,
                    help="Max samples for isotonic prob baseline fit")
    args = ap.parse_args()

    def parse_years(s: str) -> Tuple[int, int]:
        a, b = s.split("-"); return int(a), int(b)

    providers = _parse_list(args.providers)
    variables = _parse_list(args.variables)
    leads = [int(x) for x in _parse_list(args.leads)]
    alpha = float(args.alpha)
    outdir = args.outdir
    morph_cfg = {
        "operation": args.morph_op,
        "radius": int(args.morph_radius),
        "element": "disk",
        "iterations": int(args.morph_iters),
    }
    thresholds_map = _default_thresholds()

    ens_cal, ens_test = parse_years(args.ens_years_cal), parse_years(args.ens_years_test)
    pers_cal, pers_test = parse_years(args.pers_years_cal), parse_years(args.pers_years_test)

    all_rows: List[Dict[str, Any]] = []
    for provider in providers:
        for var in variables:
            if var not in thresholds_map:
                print(f"[WARN] No default thresholds for variable '{var}'. Skipping.")
                continue
            for lead in leads:
                for thr in thresholds_map[var]:
                    # pick years split per provider
                    if provider == "ifs_mean":
                        ycal, ytest = ens_cal, ens_test
                    elif provider == "persistence":
                        ycal, ytest = pers_cal, pers_test
                    elif provider == "graphcast":
                        ycal, ytest = (2018, 2018), (2020, 2020)
                    elif provider == "neuralgcm":
                        # internal split by proportion in loader
                        ycal, ytest = (2020, 2020), (2020, 2020)
                    else:
                        raise ValueError(f"Unknown provider '{provider}'")

                    try:
                        res = run_one_combination(
                            out_dir=outdir,
                            provider=provider,
                            variable=var,
                            lead_hours=lead,
                            threshold=thr,
                            alpha=alpha,
                            years_cal=ycal,
                            years_test=ytest,
                            morph_cfg=morph_cfg,
                            prob_max_samples=int(args.prob_max_samples),
                        )
                        all_rows.append(res)
                    except Exception as e:
                        print(f"[ERROR] {provider}/{var}/{lead}h/T{thr}: {e}")

    # Write summary CSV
    _summary_csv(all_rows, os.path.join(outdir, "summary.csv"))

    # Simple “who wins” printout at the end
    print("\n=== Sanity leaderboard (lower FPA better) ===")
    for d in all_rows:
        fpas = {
            "fieldcert": d["fieldcert"]["fpa"],
            "global_bump": d["global_bump"]["fpa"],
            "morph_cp": d["morph_cp"]["fpa"],
            "pixel_cp": d["pixel_cp"]["fpa"],
            "prob_isotonic": d["prob_isotonic"]["fpa"],
        }
        best = min(fpas, key=fpas.get)
        print(f"{d['provider']}/{d['variable']}/{d['lead_hours']}h/T{d['threshold']}:"
              f" best={best} ({fpas[best]:.3f}), FieldCert={fpas['fieldcert']:.3f}")

if __name__ == "__main__":
    main()
