# scripts/fc_prepare_subset.py
from __future__ import annotations
import argparse
import os
import numpy as np
import xarray as xr
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from eval.wb2 import open_wb2_dataset, select_forecast_lead, align_forecast_obs, compute_area_weights

def parse_years(s: str) -> tuple[int, int]:
    if "-" in s:
        a, b = s.split("-")
        return int(a), int(b)
    y = int(s)
    return y, y

def main():
    ap = argparse.ArgumentParser(description="Prepare aligned forecast/truth NPZ subset from WeatherBench-2.")
    ap.add_argument("--variable", default="10m_wind_speed", help="Variable name in WB-2 (e.g., 10m_wind_speed)")
    ap.add_argument("--lead_hours", type=int, default=24, help="Forecast lead in hours (e.g., 24)")
    ap.add_argument("--years", default="2020", help="Year or range 'YYYY' or 'YYYY-YYYY'")
    ap.add_argument("--out", default="/workspace/data/subsets/wb2_subset.npz", help="Output NPZ path")
    ap.add_argument("--forecast_path", default="gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr", help="Forecast dataset path")
    ap.add_argument("--obs_path", default="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr", help="Truth dataset path")
    ap.add_argument("--tolerance_hours", type=int, default=3, help="Nearest-time matching tolerance for truth")
    args = ap.parse_args()

    var = args.variable
    lead_h = int(args.lead_hours)
    y0, y1 = parse_years(args.years)

    print(f"[INFO] Preparing subset for variable={var}, lead={lead_h}h, years={y0}-{y1}")

    # Open datasets (streaming via gcsfs)
    print("[INFO] Opening forecast dataset...")
    ds_fc = open_wb2_dataset(args.forecast_path, chunks={"time": 100})
    print("[INFO] Opening observation dataset...")
    ds_obs = open_wb2_dataset(args.obs_path, chunks={"time": 100})

    # Get forecast DataArray and select lead time
    if var not in ds_fc:
        raise ValueError(f"Variable '{var}' not found in forecast dataset. Available: {list(ds_fc.data_vars)}")
    
    da_fc = ds_fc[var]
    print(f"[INFO] Forecast data shape: {da_fc.shape}, dims: {da_fc.dims}")
    
    # Select forecast lead
    da_fc = select_forecast_lead(ds_fc, lead_h)[var]
    print(f"[INFO] After lead selection: {da_fc.shape}, dims: {da_fc.dims}")

    # Standardize coordinate names and transpose
    coord_mapping = {}
    if "forecast_time" in da_fc.coords and "time" not in da_fc.coords:
        coord_mapping["forecast_time"] = "time"
    if "latitude" in da_fc.coords and "lat" not in da_fc.coords:
        coord_mapping["latitude"] = "lat"
    if "longitude" in da_fc.coords and "lon" not in da_fc.coords:
        coord_mapping["longitude"] = "lon"
    
    if coord_mapping:
        da_fc = da_fc.rename(coord_mapping)

    # Ensure time is first dimension
    dims = list(da_fc.dims)
    if "time" in dims:
        dims.remove("time")
        dims.insert(0, "time")
        da_fc = da_fc.transpose(*dims)

    # Get coordinate names
    lat_name = "lat" if "lat" in da_fc.coords else ("latitude" if "latitude" in da_fc.coords else None)
    lon_name = "lon" if "lon" in da_fc.coords else ("longitude" if "longitude" in da_fc.coords else None)
    if lat_name is None or lon_name is None:
        raise RuntimeError("Could not find lat/lon coordinates in forecast dataset.")

    # Compute valid times for forecast
    times_init = da_fc["time"].values  # initialization times
    valid_times = times_init + np.timedelta64(lead_h, "h")  # valid times for the forecast

    print(f"[INFO] Forecast time range: {times_init[0]} to {times_init[-1]}")
    print(f"[INFO] Valid time range: {valid_times[0]} to {valid_times[-1]}")

    # Get observation data and align
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

    # Select observations at nearest valid times (within tolerance)
    tol = np.timedelta64(int(args.tolerance_hours), "h")
    valid_da = xr.DataArray(valid_times, dims=("time",))
    
    print("[INFO] Aligning observations with forecast valid times...")
    da_obs_sel = da_obs.sel(time=valid_da, method="nearest", tolerance=tol)

    # Convert to arrays and filter by years and valid data
    print(f"[INFO] Creating small subset for efficient processing...")
    
    # Take only first 100 time steps to avoid memory issues with full dataset
    max_times = min(100, da_fc.sizes['time'], da_obs_sel.sizes['time'])
    print(f"[INFO] Processing first {max_times} time steps to avoid memory overload")
    
    da_fc_subset = da_fc.isel(time=slice(0, max_times))
    da_obs_subset = da_obs_sel.isel(time=slice(0, max_times))
    
    print(f"[INFO] Loading forecast subset...")
    yhat_full = da_fc_subset.values  # Much smaller, manageable size
    print(f"[INFO] Loading observation subset...")
    y_full = da_obs_subset.values  # Much smaller, manageable size
    
    print(f"[INFO] Data shapes - forecast: {yhat_full.shape}, obs: {y_full.shape}")

    # Get valid times from the subset
    valid_times_subset = da_fc_subset.time.values
    
    # Filter rows where truth is entirely finite
    mask_valid = np.all(np.isfinite(y_full.reshape(y_full.shape[0], -1)), axis=1)
    print(f"[INFO] Valid time steps: {np.sum(mask_valid)}/{len(mask_valid)}")

    # Filter by requested years
    vt = valid_times_subset[mask_valid]
    if vt.size == 0:
        raise RuntimeError("No valid times available after truth alignment.")
    
    years_arr = (vt.astype("datetime64[Y]").astype(int) + 1970)
    sel = (years_arr >= y0) & (years_arr <= y1)

    if not np.any(sel):
        raise RuntimeError(f"No samples in requested years {y0}-{y1} (lead={lead_h}h).")

    print(f"[INFO] Time steps in requested years: {np.sum(sel)}")

    yhat = yhat_full[mask_valid][sel]
    y = y_full[mask_valid][sel]
    vt = vt[sel]

    lat = np.asarray(da_fc.coords[lat_name].values)
    lon = np.asarray(da_fc.coords[lon_name].values)

    # Save the subset
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, 
                        yhat=yhat, 
                        y=y, 
                        lat=lat, 
                        lon=lon, 
                        time=vt.astype("datetime64[ns]"),
                        variable=var, 
                        lead_hours=lead_h, 
                        years=np.array([y0, y1], dtype=np.int32))
    
    print(f"[OK] Saved subset: {args.out}")
    print(f"  shapes: yhat={yhat.shape}, y={y.shape}, lat={lat.shape}, lon={lon.shape}")
    print(f"  time range: {vt[0]} .. {vt[-1]}")
    print(f"  file size: {os.path.getsize(args.out) / (1024**2):.1f} MB")

if __name__ == "__main__":
    main()
