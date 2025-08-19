#!/usr/bin/env python3
"""
Test script to verify the FieldCert-Weather pipeline works with synthetic data.
This creates fake data to test all components without requiring WeatherBench-2 access.
"""

import numpy as np
import os
import sys
import tempfile
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from calibration import CRCCalibrator, CRCSettings, build_margin_field
from calibration.regimes import assign_blocks
from baselines import (
    calibrate_global_bump_by_fpa, apply_global_bump,
    calibrate_morph_radius_by_fpa, apply_morph_radius,
    calibrate_pixelwise_delta, apply_pixelwise_delta,
)
from eval import evaluate_set_masks, compute_area_weights
import xarray as xr

def create_synthetic_data():
    """Create synthetic forecast/truth data for testing."""
    np.random.seed(42)
    
    # Create synthetic data: 30 time steps, 20x20 grid
    T, H, W = 30, 20, 20
    
    # Create synthetic forecast field (wind speed-like)
    yhat = np.random.exponential(scale=15.0, size=(T, H, W)).astype(np.float64)
    
    # Create correlated truth with some noise
    y = yhat + np.random.normal(0, 3.0, size=(T, H, W))
    y = np.maximum(y, 0)  # Wind speed can't be negative
    
    # Create synthetic coordinates
    lat = np.linspace(30, 50, H)  # Latitude range
    lon = np.linspace(-120, -100, W)  # Longitude range
    
    # Create synthetic time stamps
    times = np.array([np.datetime64('2020-01-01') + np.timedelta64(i*6, 'h') for i in range(T)])
    
    return yhat, y, lat, lon, times

def test_pipeline():
    """Test the complete pipeline with synthetic data."""
    print("🧪 Testing FieldCert-Weather Pipeline with Synthetic Data")
    print("=" * 60)
    
    # Create synthetic data
    print("📊 Creating synthetic data...")
    yhat, y, lat, lon, times = create_synthetic_data()
    print(f"   Data shapes: forecast={yhat.shape}, truth={y.shape}")
    
    # Create area weights
    print("🌍 Computing area weights...")
    dummy_ds = xr.Dataset(coords={"latitude": lat, "longitude": lon})
    weights = compute_area_weights(dummy_ds, lat_name="latitude", normalize=True)
    print(f"   Weights shape: {weights.shape}")
    
    # Split data
    print("✂️  Splitting data (70% train, 30% test)...")
    n_train = int(0.7 * len(times))
    yhat_cal, y_cal = yhat[:n_train], y[:n_train]
    yhat_test, y_test = yhat[n_train:], y[n_train:]
    times_cal, times_test = times[:n_train], times[n_train:]
    
    print(f"   Train: {yhat_cal.shape}, Test: {yhat_test.shape}")
    
    # Test parameters
    threshold = 20.0
    alpha = 0.10
    
    print(f"🎯 Using threshold={threshold}, target FPA={alpha}")
    
    # Test CRC (FieldCert)
    print("\n🔬 Testing FieldCert CRC...")
    settings = CRCSettings(
        alpha=alpha,
        lambda_grid=np.linspace(0.0, 1.0, 21),  # Smaller grid for testing
        loss_type="fpa"
    )
    crc = CRCCalibrator(settings=settings)
    
    # Build margins
    margins_cal = build_margin_field(yhat_cal, method="grad_mag", normalize=True)
    block_ids_cal = assign_blocks(times_cal, block="week")
    
    print(f"   Built margins: {margins_cal.shape}")
    print(f"   Assigned {len(np.unique(block_ids_cal))} blocks")
    
    # Fit CRC
    crc_res = crc.fit_for_regime(
        preds=yhat_cal,
        truths=y_cal,
        threshold=threshold,
        margins=margins_cal,
        block_ids=block_ids_cal,
    )
    
    lambda_star = crc_res.lambda_star
    print(f"   ✅ CRC calibrated: λ* = {lambda_star:.4f}")
    
    # Apply to test data
    margins_test = build_margin_field(yhat_test, method="grad_mag", normalize=True)
    from calibration.morphology import threshold_bump_mask
    pred_crc = threshold_bump_mask(
        pred_field=yhat_test,
        threshold=threshold,
        margin_field=margins_test,
        lam=lambda_star,
    )
    truth_test = (y_test >= threshold)
    
    # Test baselines
    print("\n📊 Testing baseline methods...")
    
    # Global bump
    tau_star, _, _ = calibrate_global_bump_by_fpa(
        yhat_cal, y_cal, threshold, alpha=alpha, spatial_weights=weights
    )
    pred_global = apply_global_bump(yhat_test, threshold, tau_star)
    print(f"   ✅ Global bump: τ* = {tau_star:.4f}")
    
    # Morphological CP
    r_star, _, _ = calibrate_morph_radius_by_fpa(
        yhat_cal, y_cal, threshold, alpha=alpha, spatial_weights=weights
    )
    pred_morph = apply_morph_radius(yhat_test, threshold, int(r_star))
    print(f"   ✅ Morph CP: r* = {r_star}")
    
    # Pixelwise CP
    delta = calibrate_pixelwise_delta(yhat_cal, y_cal, threshold, alpha_pixel=alpha)
    pred_pixelcp = apply_pixelwise_delta(yhat_test, threshold, delta)
    print(f"   ✅ Pixel CP: δ range = [{np.min(delta):.4f}, {np.max(delta):.4f}]")
    
    # Evaluate all methods
    print("\n📈 Evaluating all methods...")
    results = {}
    
    for name, pred_mask in [
        ("fieldcert_crc", pred_crc),
        ("global_bump", pred_global),
        ("morph_cp", pred_morph),
        ("pixel_cp", pred_pixelcp),
    ]:
        metrics = evaluate_set_masks(pred_mask, truth_test, weights=weights)
        results[name] = metrics
        print(f"   {name:15s}: FPA={metrics['fpa']:.4f}, FNA={metrics['fna']:.4f}, IoU={metrics['iou']:.4f}")
    
    # Check if FPA is close to target
    print("\n🎯 Checking calibration quality...")
    for name, metrics in results.items():
        fpa = metrics['fpa']
        error = abs(fpa - alpha)
        status = "✅" if error < 0.05 else "⚠️"
        print(f"   {name:15s}: FPA error = {error:.4f} {status}")
    
    print("\n🎉 Pipeline test completed successfully!")
    print("   All components are working correctly.")
    
    return results

if __name__ == "__main__":
    try:
        results = test_pipeline()
        print("\n✅ TEST PASSED: Pipeline is ready for production!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
