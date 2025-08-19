# scripts/fc_run_emos.py
from __future__ import annotations
import argparse, json, os
import numpy as np
import xarray as xr
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from baselines import EMOSLogistic, prob_to_set_calibrate_threshold, evaluate_prob_event_baseline
from eval.wb2 import open_wb2_dataset, compute_area_weights

def parse_years(s: str) -> tuple[int, int]:
    if "-" in s:
        a, b = s.split("-")
        return int(a), int(b)
    y = int(s)
    return y, y

def main():
    ap = argparse.ArgumentParser(description="Fit & evaluate EMOS logistic baseline from IFS ENS.")
    ap.add_argument("--variable", default="10m_wind_speed")
    ap.add_argument("--lead_hours", type=int, default=24)
    ap.add_argument("--years_cal", default="2019", help="Calibration years")
    ap.add_argument("--years_test", default="2020", help="Test years")
    ap.add_argument("--members", type=int, default=10, help="Number of ensemble members to use (<=50)")
    ap.add_argument("--ens_path", default="gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr")
    ap.add_argument("--era5_path", default="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
    ap.add_argument("--threshold", type=float, default=20.0)
    ap.add_argument("--alpha", type=float, default=0.10, help="Target FPA for prob->set")
    ap.add_argument("--out_json", default="/workspace/results/emos_metrics.json")
    args = ap.parse_args()

    var, lead_h = args.variable, int(args.lead_hours)
    y0c, y1c = parse_years(args.years_cal)
    y0t, y1t = parse_years(args.years_test)
    M = int(args.members)

    print(f"[INFO] EMOS setup: variable={var}, lead={lead_h}h, members={M}")
    print(f"[INFO] Calibration years: {y0c}-{y1c}, Test years: {y0t}-{y1t}")

    # Open ensemble dataset (very large -> stream small subsets)
    print("[INFO] Opening IFS ensemble dataset...")
    ds_ens = open_wb2_dataset(args.ifs_ens_path, chunks={"time": 50, "number": M})
    
    if var not in ds_ens:
        raise ValueError(f"Variable '{var}' not found in ensemble dataset. Available: {list(ds_ens.data_vars)}")
    
    da_ens = ds_ens[var]  # dims: (time, number, prediction_timedelta, lat, lon) or similar
    print(f"[INFO] Ensemble data shape: {da_ens.shape}, dims: {da_ens.dims}")
    
    # Select lead time
    if "prediction_timedelta" not in da_ens.coords:
        raise RuntimeError("prediction_timedelta coordinate not found in IFS ENS dataset")
    
    lead_timedelta = np.timedelta64(lead_h, "h")
    da_ens_lead = da_ens.sel(prediction_timedelta=lead_timedelta, method="nearest")
    print(f"[INFO] After lead selection: {da_ens_lead.shape}, dims: {da_ens_lead.dims}")
    
    # Get time coordinate and filter by years
    times = da_ens_lead["time"].values
    years = (times.astype("datetime64[Y]").astype(int) + 1970)
    cal_mask = (years >= y0c) & (years <= y1c)
    test_mask = (years >= y0t) & (years <= y1t)
    
    if not np.any(cal_mask):
        raise RuntimeError("No calibration times found for requested years.")
    if not np.any(test_mask):
        raise RuntimeError("No test times found for requested years.")
    
    print(f"[INFO] Time steps - calibration: {np.sum(cal_mask)}, test: {np.sum(test_mask)}")
    
    # Select ensemble members
    num_coord = "number" if "number" in da_ens_lead.dims else "ensemble"
    members = da_ens_lead.coords[num_coord].values
    if len(members) < M:
        print(f"[WARNING] Only {len(members)} members available, using all of them")
        M = len(members)
        sel_members = members
    else:
        sel_members = members[:M]
    
    da_ens_lead = da_ens_lead.sel({num_coord: sel_members})
    print(f"[INFO] Using {M} ensemble members: {sel_members}")

    # Load ensemble data for calibration and test periods
    print("[INFO] Loading ensemble data...")
    times_cal = times[cal_mask]
    times_test = times[test_mask]
    
    ens_cal = da_ens_lead.sel(time=times_cal).transpose("time", num_coord, ...).values
    ens_test = da_ens_lead.sel(time=times_test).transpose("time", num_coord, ...).values
    
    print(f"[INFO] Ensemble shapes - cal: {ens_cal.shape}, test: {ens_test.shape}")

    # Open ERA5 and align by nearest valid time
    print("[INFO] Opening ERA5 observations...")
    ds_obs = open_wb2_dataset(args.era5_path, chunks={"time": 100})
    
    if var not in ds_obs:
        raise ValueError(f"Variable '{var}' not found in observation dataset. Available: {list(ds_obs.data_vars)}")
    
    da_obs = ds_obs[var]
    
    # Standardize observation coordinate names
    obs_coord_mapping = {}
    if "forecast_time" in da_obs.coords and "time" not in da_obs.coords:
        obs_coord_mapping["forecast_time"] = "time"
    if "latitude" in da_obs.coords and "lat" not in da_obs.coords:
        obs_coord_mapping["latitude"] = "lat"
    if "longitude" in da_obs.coords and "lon" not in da_obs.coords:
        obs_coord_mapping["longitude"] = "lon"
    
    if obs_coord_mapping:
        da_obs = da_obs.rename(obs_coord_mapping)

    # Ensure time is first dimension for observations
    obs_dims = list(da_obs.dims)
    if "time" in obs_dims:
        obs_dims.remove("time")
        obs_dims.insert(0, "time")
        da_obs = da_obs.transpose(*obs_dims)

    # Compute valid times (forecast initialization + lead time)
    valid_cal = times_cal + np.timedelta64(lead_h, "h")
    valid_test = times_test + np.timedelta64(lead_h, "h")

    print("[INFO] Aligning observations with forecast valid times...")
    tolerance = np.timedelta64(3, "h")
    y_cal = da_obs.sel(time=xr.DataArray(valid_cal, dims=("time",)), 
                       method="nearest", tolerance=tolerance).values
    y_test = da_obs.sel(time=xr.DataArray(valid_test, dims=("time",)), 
                        method="nearest", tolerance=tolerance).values

    print(f"[INFO] Observation shapes - cal: {y_cal.shape}, test: {y_test.shape}")

    # Drop rows with NaNs in truth
    mask_c = np.all(np.isfinite(y_cal.reshape(y_cal.shape[0], -1)), axis=1)
    mask_t = np.all(np.isfinite(y_test.reshape(y_test.shape[0], -1)), axis=1)
    
    ens_cal = ens_cal[mask_c]
    ens_test = ens_test[mask_t]
    y_cal = y_cal[mask_c]
    y_test = y_test[mask_t]

    print(f"[INFO] After filtering NaNs - cal: {ens_cal.shape}, test: {ens_test.shape}")

    # Get coordinates for area weighting
    lat_name = "lat" if "lat" in da_ens_lead.coords else "latitude"
    lon_name = "lon" if "lon" in da_ens_lead.coords else "longitude"
    lat = da_ens_lead.coords[lat_name].values
    lon = da_ens_lead.coords[lon_name].values
    
    # Create dummy dataset for area weights computation
    dummy_ds = xr.Dataset(coords={lat_name: lat, lon_name: lon})
    Wts = compute_area_weights(dummy_ds, lat_name=lat_name, normalize=True)

    print(f"[INFO] Area weights shape: {Wts.shape}")

    # Fit EMOS logistic on calibration data
    print("[INFO] Fitting EMOS logistic model...")
    model = EMOSLogistic()
    model.fit(ens_cal, y_cal, threshold=float(args.threshold), 
              max_samples=2_000_000, sample_strategy="stratified")
    
    # Get calibration probabilities
    prob_cal = model.predict_proba(ens_cal)
    event_cal = (y_cal >= float(args.threshold))

    print(f"[INFO] Calibration event frequency: {np.mean(event_cal):.4f}")

    # Calibrate probability threshold to meet target FPA
    print("[INFO] Calibrating probability threshold...")
    p_star, curve, grid = prob_to_set_calibrate_threshold(
        prob_cal, event_cal, alpha=float(args.alpha), spatial_weights=Wts
    )
    
    print(f"[INFO] Calibrated probability threshold: {p_star:.4f}")

    # Evaluate on test data
    print("[INFO] Evaluating on test data...")
    prob_test = model.predict_proba(ens_test)
    metrics = evaluate_prob_event_baseline(
        prob_test, y_test, float(args.threshold), 
        spatial_weights=Wts, prob_threshold=float(p_star)
    )

    print(f"[INFO] Test results:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")

    # Save results
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    results = {
        "variable": var,
        "lead_hours": lead_h,
        "years_cal": [y0c, y1c],
        "years_test": [y0t, y1t],
        "members": M,
        "threshold": float(args.threshold),
        "alpha": float(args.alpha),
        "prob_threshold_star": float(p_star),
        "calibration_event_freq": float(np.mean(event_cal)),
        "test_event_freq": float(np.mean(y_test >= args.threshold)),
        "metrics": metrics,
        "data_shapes": {
            "calibration": list(ens_cal.shape),
            "test": list(ens_test.shape)
        }
    }
    
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[OK] EMOS metrics saved to {args.out_json}")

if __name__ == "__main__":
    main()
