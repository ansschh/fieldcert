# src/eval/wb2.py
from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Dict, List, Tuple, Optional, Union, Sequence, Any


def open_wb2_dataset(
    path: str,
    *,
    engine: str = "zarr",
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """
    Open a WeatherBench-2 dataset from local path or GCS.

    Parameters
    ----------
    path : str
        Path to dataset, local or gs:// URL
    engine : str, default 'zarr'
        Engine to use for opening the dataset
    chunks : dict, optional
        Chunk sizes for dask array

    Returns
    -------
    xr.Dataset
        WeatherBench-2 dataset
    """
    if path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        store = gcsfs.mapping.GCSMap(path, gcs=fs, check=False)
        return xr.open_zarr(store, chunks=chunks)
    else:
        return xr.open_zarr(path, chunks=chunks)


def select_forecast_lead(
    ds: xr.Dataset,
    lead_hours: Union[int, List[int]],
    *,
    lead_dim: str = "prediction_timedelta",
    valid_time_var: str = "valid_time",
) -> xr.Dataset:
    """
    Select specific forecast lead time(s) from a forecast dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Forecast dataset with lead_dim dimension
    lead_hours : int or list of int
        Lead time(s) in hours to select
    lead_dim : str, default 'step'
        Name of the lead time dimension
    valid_time_var : str, default 'valid_time'
        Name of the valid time variable

    Returns
    -------
    xr.Dataset
        Dataset with selected lead time(s)
    """
    if isinstance(lead_hours, int):
        lead_hours = [lead_hours]
    
    # Find the indices corresponding to the requested lead hours
    lead_values = ds[lead_dim].values
    
    # Convert lead_values to hours if they are timedelta64
    if np.issubdtype(lead_values.dtype, np.timedelta64):
        lead_hours_array = lead_values / np.timedelta64(1, 'h')
    else:
        lead_hours_array = lead_values
    
    lead_indices = []
    for lh in lead_hours:
        idx = np.argmin(np.abs(lead_hours_array - lh))
        if abs(lead_hours_array[idx] - lh) > 1:  # Allow 1 hour tolerance
            raise ValueError(f"Lead time {lh}h not found in dataset (closest: {lead_hours_array[idx]}h)")
        lead_indices.append(idx)
    
    # Select the lead times
    result = ds.isel({lead_dim: lead_indices})
    
    # If we selected a single lead time and the dimension was dropped, restore it
    if len(lead_hours) == 1 and lead_dim not in result.dims:
        result = result.expand_dims(lead_dim)
    
    return result


def align_forecast_obs(
    forecast: xr.Dataset,
    obs: xr.Dataset,
    *,
    fcst_valid_time: str = "valid_time",
    obs_time: str = "time",
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Align forecast and observation datasets by valid time.

    Parameters
    ----------
    forecast : xr.Dataset
        Forecast dataset with valid time variable
    obs : xr.Dataset
        Observation dataset with time dimension
    fcst_valid_time : str, default 'valid_time'
        Name of the valid time variable in forecast dataset
    obs_time : str, default 'time'
        Name of the time dimension in observation dataset

    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
        Aligned (forecast, observation) datasets
    """
    # Get valid times from forecast
    if fcst_valid_time in forecast:
        valid_times = forecast[fcst_valid_time].values
    else:
        raise ValueError(f"Valid time variable '{fcst_valid_time}' not found in forecast dataset")
    
    # Select matching times in observations
    obs_aligned = obs.sel({obs_time: valid_times}, method="nearest", tolerance="1h")
    
    # Check if we got all times
    if len(obs_aligned[obs_time]) != len(valid_times):
        missing = len(valid_times) - len(obs_aligned[obs_time])
        raise ValueError(f"Could not find matching observations for {missing} forecast times")
    
    return forecast, obs_aligned


def compute_area_weights(
    ds: xr.Dataset,
    *,
    lat_name: str = "latitude",
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute area weights based on latitude.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with latitude dimension/coordinate
    lat_name : str, default 'latitude'
        Name of the latitude dimension/coordinate
    normalize : bool, default True
        Whether to normalize weights to sum to 1

    Returns
    -------
    np.ndarray
        Area weights with same shape as the spatial grid
    """
    if lat_name not in ds:
        raise ValueError(f"Latitude variable '{lat_name}' not found in dataset")
    
    lat = ds[lat_name].values
    lat_rad = np.deg2rad(lat)
    
    # Compute weights proportional to cos(lat)
    weights = np.cos(lat_rad)
    
    # Broadcast to full grid if needed
    if lat.ndim == 1:
        # Assume standard lat/lon grid
        lon_dim = [d for d in ds.dims if d != lat_name and len(ds[d]) > 1][0]
        weights = np.broadcast_to(weights[:, np.newaxis], (len(lat), len(ds[lon_dim])))
    
    # Normalize if requested
    if normalize and np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    
    return weights


def build_bbox_mask(
    ds: xr.Dataset,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    *,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> np.ndarray:
    """
    Build a boolean mask for a bounding box region.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with latitude and longitude coordinates
    lat_range : Tuple[float, float]
        (min_lat, max_lat) range in degrees
    lon_range : Tuple[float, float]
        (min_lon, max_lon) range in degrees
    lat_name : str, default 'latitude'
        Name of the latitude coordinate
    lon_name : str, default 'longitude'
        Name of the longitude coordinate

    Returns
    -------
    np.ndarray
        Boolean mask with True inside the bounding box
    """
    if lat_name not in ds or lon_name not in ds:
        raise ValueError(f"Coordinates {lat_name} or {lon_name} not found in dataset")
    
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    
    # Handle wrapped longitudes
    min_lon, max_lon = lon_range
    if min_lon > max_lon:  # Crossing the dateline
        lon_mask = (lon >= min_lon) | (lon <= max_lon)
    else:
        lon_mask = (lon >= min_lon) & (lon <= max_lon)
    
    min_lat, max_lat = lat_range
    lat_mask = (lat >= min_lat) & (lat <= max_lat)
    
    # Create the full mask
    if lat.ndim == 1 and lon.ndim == 1:
        # Standard lat/lon grid
        mask = np.logical_and.outer(lat_mask, lon_mask)
    else:
        # Non-standard grid
        mask = (lat >= min_lat) & (lat <= max_lat) & lon_mask
    
    return mask
