#!/usr/bin/env python3
"""
Check WeatherBench-2 dataset paths and accessibility.
This script helps diagnose connectivity and path issues.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import gcsfs
import xarray as xr
from eval.wb2 import open_wb2_dataset

def check_gcs_access():
    """Test basic GCS access."""
    print("🔍 Testing GCS access...")
    try:
        fs = gcsfs.GCSFileSystem(token='anon')
        # Try to list the weatherbench2 bucket
        files = fs.ls('weatherbench2', detail=False)[:5]  # Just first 5 items
        print(f"   ✅ GCS access OK. Found {len(files)} items in weatherbench2 bucket")
        for f in files:
            print(f"      - {f}")
        return True
    except Exception as e:
        print(f"   ❌ GCS access failed: {e}")
        return False

def check_dataset_path(path, name):
    """Check if a specific dataset path exists and is accessible."""
    print(f"\n🔍 Testing {name}:")
    print(f"   Path: {path}")
    
    try:
        fs = gcsfs.GCSFileSystem(token='anon')
        # Check if path exists
        if fs.exists(path):
            print("   ✅ Path exists")
            
            # Try to list contents
            try:
                contents = fs.ls(path, detail=False)[:3]  # First 3 items
                print(f"   ✅ Can list contents ({len(contents)} items)")
                for item in contents:
                    print(f"      - {os.path.basename(item)}")
            except Exception as e:
                print(f"   ⚠️  Path exists but can't list contents: {e}")
            
            # Try to open with xarray
            try:
                print("   🔍 Testing xarray access...")
                ds = open_wb2_dataset(path, chunks={'time': 1})
                print(f"   ✅ xarray access OK. Variables: {list(ds.data_vars)[:3]}...")
                ds.close()
                return True
            except Exception as e:
                print(f"   ❌ xarray access failed: {e}")
                return False
        else:
            print("   ❌ Path does not exist")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking path: {e}")
        return False

def suggest_alternatives():
    """Suggest alternative dataset paths."""
    print("\n💡 Checking alternative WeatherBench-2 paths...")
    
    # Alternative paths to try
    alternatives = [
        ("ERA5 (alternative)", "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"),
        ("IFS ENS mean (alternative)", "gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative_mean.zarr"),
        ("ERA5 (high-res)", "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-1440x721_equiangular_with_poles_conservative.zarr"),
        ("IFS HRES", "gs://weatherbench2/datasets/ifs_hres/2018-2022-240x121_equiangular_with_poles_conservative.zarr"),
    ]
    
    working_paths = []
    for name, path in alternatives:
        if check_dataset_path(path, name):
            working_paths.append((name, path))
    
    return working_paths

def main():
    print("🌍 WeatherBench-2 Dataset Connectivity Check")
    print("=" * 50)
    
    # Test basic GCS access
    if not check_gcs_access():
        print("\n❌ Cannot access GCS. Check internet connection.")
        return
    
    # Test the default paths from our scripts
    default_paths = [
        ("IFS ENS mean (default)", "gs://weatherbench2/datasets/ens/2018-2022-240x121_equiangular_with_poles_conservative_mean.zarr"),
        ("ERA5 (default)", "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"),
        ("IFS ENS full", "gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr"),
    ]
    
    working_paths = []
    for name, path in default_paths:
        if check_dataset_path(path, name):
            working_paths.append((name, path))
    
    # If no default paths work, try alternatives
    if not working_paths:
        print("\n⚠️  Default paths failed. Checking alternatives...")
        working_paths = suggest_alternatives()
    
    # Summary
    print("\n" + "=" * 50)
    if working_paths:
        print("✅ WORKING DATASET PATHS:")
        for name, path in working_paths:
            print(f"   {name}: {path}")
        
        print(f"\n💡 Recommendation: Update your script to use one of the working paths above.")
        print("   You can modify the default paths in fc_prepare_subset.py")
    else:
        print("❌ NO WORKING PATHS FOUND")
        print("   This could be due to:")
        print("   1. WeatherBench-2 dataset paths have changed")
        print("   2. Network connectivity issues")
        print("   3. GCS authentication problems")
        print("   4. Temporary service outage")
        
        print(f"\n💡 Try:")
        print("   1. Check WeatherBench-2 documentation for updated paths")
        print("   2. Verify internet connectivity")
        print("   3. Try again later (temporary outage)")

if __name__ == "__main__":
    main()
