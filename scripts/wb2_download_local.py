# scripts/wb2_download_local.py
from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr

# WB-2 public buckets (240x121 equiangular) - VERIFIED WORKING PATHS
WB2_ERA5 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
WB2_ENS_MEAN = "gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr"
WB2_NEURALGCM_2020 = "gs://weatherbench2/datasets/neuralgcm_deterministic/2020-240x121_equiangular_with_poles_conservative.zarr"

def graphcast_path(year: int) -> str:
    if year == 2018:
        return "gs://weatherbench2/datasets/graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr"
    if year == 2020:
        return "gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr"
    raise ValueError("GraphCast WB-2 subsets are provided for years {2018, 2020}.")

def open_zarr(path: str, chunks="auto") -> xr.Dataset:
    storage_options = {"token": "anon"} if path.startswith("gs://") else None
    return xr.open_zarr(path, storage_options=storage_options, chunks=chunks, decode_timedelta=False)

def select_lead(da: xr.DataArray, lead_hours: int) -> xr.DataArray:
    for c in ("prediction_timedelta", "lead", "step"):
        if (c in da.coords) or (c in da.dims):
            # Robustly select nearest lead by index, then drop/squeeze the lead dim
            coord_vals = da.coords[c].values
            if np.issubdtype(coord_vals.dtype, np.timedelta64):
                hours = (coord_vals / np.timedelta64(1, "h")).astype(np.float64)
            else:
                hours = coord_vals.astype(np.float64)
            idx = int(np.argmin(np.abs(hours - float(lead_hours))))
            out = da.isel({c: idx}).squeeze(drop=True)
            return out
    raise KeyError(f"No lead coord among ('prediction_timedelta','lead','step'); got {list(da.coords)}")

def std_coords(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.coords and "forecast_time" in da.coords:
        da = da.rename({"forecast_time": "time"})
    if "latitude" in da.dims and "lat" not in da.dims:
        da = da.rename({"latitude": "lat"})
    if "longitude" in da.dims and "lon" not in da.dims:
        da = da.rename({"longitude": "lon"})
    # Drop any leftover singleton dims (e.g., height=1)
    da = da.squeeze(drop=True)
    # Standardize ordering to prioritize (time, lat, lon) but allow extras via '...'
    if all(k in da.dims for k in ("time", "lat", "lon")):
        return da.transpose("time", "lat", "lon", ...)
    return da.transpose("time", ...)

def parse_years(s: str) -> tuple[int,int]:
    if "-" in s: a,b = s.split("-"); return int(a), int(b)
    y = int(s); return y, y

def year_mask(times: np.ndarray, y0: int, y1: int) -> np.ndarray:
    yrs = (times.astype("datetime64[Y]").astype(int) + 1970)
    return (yrs >= y0) & (yrs <= y1)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Materialize local Zarr for forecast+truth (memory-safe).")
    ap.add_argument("--provider", required=True, choices=["ifs_mean", "graphcast2018", "graphcast2020", "neuralgcm2020"],
                    help="Forecast provider subset to download.")
    ap.add_argument("--variable", required=True, help="Variable name (must exist in provider & ERA5).")
    ap.add_argument("--lead_hours", type=int, required=True, help="Lead time (hours).")
    ap.add_argument("--years", default="2020", help="Year or range 'YYYY-YYYY' for ifs_mean; ignored for fixed subsets.")
    ap.add_argument("--out_root", default="/workspace/data/local_wb2",
                    help="Root directory; two zarrs will be written under forecast/ and truth/")
    ap.add_argument("--time_chunk", type=int, default=64, help="Time chunk size for local Zarr")
    args = ap.parse_args()

    provider = args.provider
    variable = args.variable
    lead_h = int(args.lead_hours)
    out_root = args.out_root
    tchunk = int(args.time_chunk)

    # Forecast dataset / selection
    if provider == "ifs_mean":
        ds_fc = open_zarr(WB2_ENS_MEAN, chunks="auto")
        if variable not in ds_fc: raise KeyError(f"{variable} not in ENS-mean.")
        # Start from raw variable, reduce any ensemble dim first
        da = ds_fc[variable]
        for ens_dim in ("number", "member", "ens", "ensemble", "realization"):
            if ens_dim in da.dims:
                da = da.mean(ens_dim)
        # Select the desired lead and standardize coords
        da = std_coords(select_lead(da, lead_h))
        # Drop any other non-spatial singleton dims
        da = da.squeeze(drop=True)
        # Ensure final ordering is (time, lat, lon)
        if all(k in da.dims for k in ("time", "lat", "lon")):
            da = da.transpose("time", "lat", "lon", ...)
        times = da["time"].values.astype("datetime64[ns]")
        valid_times = times + np.timedelta64(lead_h, "h")
        y0,y1 = parse_years(args.years)
        mask = year_mask(valid_times, y0, y1)
        if not np.any(mask): raise RuntimeError("No times in requested year window for this lead.")
        da = da.sel(time=da["time"].values[mask])
        # Ensure truth selection uses the same filtered valid times
        valid_times = valid_times[mask]
        tag = f"ifs_mean__{variable}__L{lead_h}h__{y0}-{y1}"

    elif provider == "graphcast2018":
        ds = open_zarr(graphcast_path(2018), chunks="auto")
        if variable not in ds: raise KeyError(f"{variable} not in GraphCast 2018.")
        da = std_coords(select_lead(ds[variable], lead_h))
        tag = f"graphcast2018__{variable}__L{lead_h}h"
        valid_times = da["time"].values.astype("datetime64[ns]") + np.timedelta64(lead_h, "h")

    elif provider == "graphcast2020":
        ds = open_zarr(graphcast_path(2020), chunks="auto")
        if variable not in ds: raise KeyError(f"{variable} not in GraphCast 2020.")
        da = std_coords(select_lead(ds[variable], lead_h))
        tag = f"graphcast2020__{variable}__L{lead_h}h"
        valid_times = da["time"].values.astype("datetime64[ns]") + np.timedelta64(lead_h, "h")

    elif provider == "neuralgcm2020":
        ds = open_zarr(WB2_NEURALGCM_2020, chunks="auto")
        if variable not in ds: raise KeyError(f"{variable} not in NeuralGCM 2020.")
        da = std_coords(select_lead(ds[variable], lead_h))
        tag = f"neuralgcm2020__{variable}__L{lead_h}h"
        valid_times = da["time"].values.astype("datetime64[ns]") + np.timedelta64(lead_h, "h")

    else:
        raise ValueError("Unknown provider")

    # Chunk time for the local store
    da = da.chunk({"time": tchunk})

    # Write forecast Zarr
    f_out = os.path.join(out_root, "forecast", f"{tag}.zarr")
    ensure_dir(os.path.dirname(f_out))
    print(f"[forecast] writing: {f_out}")
    # Use compatible encoding to avoid zarr version conflicts
    encoding = {da.name: {"compressor": None}} if da.name else {}
    # Explicitly write Zarr v2 to remain compatible across environments
    da.to_zarr(f_out, mode="w", consolidated=True, encoding=encoding, zarr_version=2)

    # Build ERA5 truth at the exact valid times (nearest within 3h)
    ds_obs = open_zarr(WB2_ERA5, chunks="auto")
    if variable not in ds_obs: raise KeyError(f"{variable} not in ERA5.")
    da_obs = ds_obs[variable]
    if "time" not in da_obs.coords and "forecast_time" in da_obs.coords:
        da_obs = da_obs.rename({"forecast_time": "time"})
    if "latitude" in da_obs.dims and "lat" not in da_obs.dims:
        da_obs = da_obs.rename({"latitude": "lat"})
    if "longitude" in da_obs.dims and "lon" not in da_obs.dims:
        da_obs = da_obs.rename({"longitude": "lon"})
    da_obs = da_obs.transpose("time", ...)

    vt = valid_times
    tol = np.timedelta64(3, "h")
    print(f"[truth] selecting {vt.size} valid times (nearest within {tol}) from ERA5...")
    # Lazy selection; then write chunked
    vt_da = xr.DataArray(vt, dims=("time",))
    sel = da_obs.sel(time=vt_da, method="nearest", tolerance=tol).chunk({"time": tchunk})
    t_out = os.path.join(out_root, "truth", f"{tag}.zarr")
    ensure_dir(os.path.dirname(t_out))
    print(f"[truth] writing: {t_out}")
    # Use compatible encoding to avoid zarr version conflicts
    encoding = {sel.name: {"compressor": None}} if sel.name else {}
    # Explicitly write Zarr v2 to remain compatible across environments
    sel.to_zarr(t_out, mode="w", consolidated=True, encoding=encoding, zarr_version=2)

    print("[OK] Done.")
    print(f"Forecast Zarr: {f_out}")
    print(f"Truth Zarr:    {t_out}")

if __name__ == "__main__":
    main()
