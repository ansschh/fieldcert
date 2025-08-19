# scripts/fc_run_crc_baselines.py
from __future__ import annotations
import argparse, json, os
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from calibration import (
    CRCCalibrator, CRCSettings,
    build_margin_field,
)
from calibration.morphology import threshold_bump_mask, symmetric_filter
from baselines import (
    calibrate_global_bump_by_fpa, apply_global_bump,
    calibrate_morph_radius_by_fpa, apply_morph_radius,
    calibrate_pixelwise_delta, apply_pixelwise_delta,
)
from eval import evaluate_set_masks, compute_area_weights
from calibration.regimes import assign_blocks

def main():
    ap = argparse.ArgumentParser(description="Run FieldCert CRC and baselines on a prepared subset NPZ.")
    ap.add_argument("--subset", required=True, help="Path to NPZ from fc_prepare_subset.py")
    ap.add_argument("--threshold", type=float, default=20.0, help="Event threshold (unit depends on variable)")
    ap.add_argument("--alpha", type=float, default=0.10, help="Target FPA")
    ap.add_argument("--train_frac", type=float, default=0.7, help="Fraction of time for calibration/training")
    ap.add_argument("--out_json", default="/workspace/results/metrics.json", help="Output metrics JSON")
    # CRC knobs
    ap.add_argument("--crc_lambda_max", type=float, default=2.0)
    ap.add_argument("--crc_lambda_step", type=float, default=0.05)
    ap.add_argument("--morph_op", default="none", choices=["none","open","close","erode","dilate"])
    ap.add_argument("--morph_radius", type=int, default=0)
    ap.add_argument("--morph_iters", type=int, default=1)
    args = ap.parse_args()

    print(f"[INFO] Loading subset from {args.subset}")
    data = np.load(args.subset, allow_pickle=True)
    yhat = data["yhat"]  # (T,H,W)
    y = data["y"]        # (T,H,W)
    lat = data["lat"]
    lon = data["lon"]
    times = data["time"].astype("datetime64[ns]")

    print(f"[INFO] Data shapes: yhat={yhat.shape}, y={y.shape}")
    print(f"[INFO] Variable: {data['variable']}, Lead: {data['lead_hours']}h, Years: {data['years']}")

    Tdim = y.shape[0]
    H, W = y.shape[1], y.shape[2]
    
    # Compute area weights from lat/lon
    # Create a simple dataset-like structure for compute_area_weights
    import xarray as xr
    dummy_ds = xr.Dataset(coords={"latitude": lat, "longitude": lon})
    Wts = compute_area_weights(dummy_ds, lat_name="latitude", normalize=True)  # (H,W)

    print(f"[INFO] Computed area weights: {Wts.shape}")

    # Split into calibration/test by time
    n_train = int(np.floor(args.train_frac * Tdim))
    idx_cal = np.arange(0, n_train)
    idx_test = np.arange(n_train, Tdim)

    yhat_cal, y_cal = yhat[idx_cal], y[idx_cal]
    yhat_test, y_test = yhat[idx_test], y[idx_test]
    times_cal, times_test = times[idx_cal], times[idx_test]

    print(f"[INFO] Split: calibration={len(idx_cal)}, test={len(idx_test)}")

    # --- CRC (FieldCert)
    print("[INFO] Running FieldCert CRC calibration...")
    lam_grid = np.arange(0.0, float(args.crc_lambda_max) + 1e-12, float(args.crc_lambda_step))
    settings = CRCSettings(
        alpha=float(args.alpha),
        lambda_grid=lam_grid,
        slack_B=1.0,
        loss_type="fpa",
        morph_operation=args.morph_op,
        morph_radius=int(args.morph_radius),
        morph_element="disk",
        morph_iterations=int(args.morph_iters),
    )
    crc = CRCCalibrator(settings=settings)

    # Physics-aware margin: gradient magnitude of forecast
    # Handle ensemble data by taking ensemble mean first
    if yhat_cal.ndim == 5:  # (T, ensemble, lead, H, W)
        yhat_cal_mean = yhat_cal.mean(axis=1)  # Average over ensemble
        if yhat_cal_mean.shape[1] == 1:  # Remove singleton lead dimension
            yhat_cal_mean = yhat_cal_mean.squeeze(axis=1)  # (T, H, W)
    else:
        yhat_cal_mean = yhat_cal
    
    print(f"[INFO] Forecast shape for margins: {yhat_cal_mean.shape}")
    margins_cal = build_margin_field(yhat_cal_mean, method="grad_mag", normalize=True)
    print(f"[INFO] Built margin fields: {margins_cal.shape}")

    # block ids: weekly (exchangeability units)
    block_ids_cal = assign_blocks(times_cal, block="week")
    print(f"[INFO] Assigned {len(np.unique(block_ids_cal))} calibration blocks")

    crc_res = crc.fit_for_regime(
        preds=yhat_cal,
        truths=y_cal,
        threshold=args.threshold,
        margins=margins_cal,
        block_ids=block_ids_cal,
        spatial_weights=Wts,
    )
    lam_star = crc_res.lambda_star
    print(f"[INFO] CRC calibrated lambda* = {lam_star:.4f}")

    # Apply lambda* to test set
    margins_test = build_margin_field(yhat_test, method="grad_mag", normalize=True)
    pred_crc = threshold_bump_mask(
        field=yhat_test, 
        threshold=float(args.threshold), 
        lambda_bump=float(lam_star),
        truth_mask=None,  # No truth mask needed for prediction
        morph_operation=args.morph_op,
        morph_radius=int(args.morph_radius),
        morph_element="disk",
        morph_iterations=int(args.morph_iters),
    )
    
    truth_test = (y_test >= float(args.threshold))
    
    # Apply symmetric morphology if specified
    if args.morph_op != "none" and args.morph_radius > 0 and args.morph_iters > 0:
        print(f"[INFO] Applying symmetric morphology: {args.morph_op}, radius={args.morph_radius}")
        pm = np.empty_like(pred_crc, dtype=bool)
        gm = np.empty_like(truth_test, dtype=bool)
        for t in range(pred_crc.shape[0]):
            pm[t], gm[t] = symmetric_filter(
                pred_crc[t], truth_test[t],
                args.morph_op, int(args.morph_radius), "disk", int(args.morph_iters)
            )
        pred_crc, truth_test = pm, gm

    # --- Baselines (calibrate on cal; apply on test)
    print("[INFO] Running baseline methods...")
    
    # 1) Global bump
    print("[INFO] Calibrating global bump baseline...")
    tau_star, _, _ = calibrate_global_bump_by_fpa(
        yhat_cal, y_cal, float(args.threshold), 
        alpha=float(args.alpha), spatial_weights=Wts
    )
    pred_global = apply_global_bump(yhat_test, float(args.threshold), tau_star)
    print(f"[INFO] Global bump tau* = {tau_star:.4f}")

    # 2) Morphological CP: erosion radius
    print("[INFO] Calibrating morphological CP baseline...")
    r_star, _, _ = calibrate_morph_radius_by_fpa(
        yhat_cal, y_cal, float(args.threshold), 
        alpha=float(args.alpha), spatial_weights=Wts
    )
    pred_morph = apply_morph_radius(yhat_test, float(args.threshold), int(r_star))
    print(f"[INFO] Morphological CP radius* = {r_star}")

    # 3) Pixelwise split CP
    print("[INFO] Calibrating pixelwise CP baseline...")
    delta = calibrate_pixelwise_delta(
        yhat_cal, y_cal, float(args.threshold), 
        alpha_pixel=float(args.alpha)
    )
    pred_pixelcp = apply_pixelwise_delta(yhat_test, float(args.threshold), delta)
    print(f"[INFO] Pixelwise CP delta range: [{np.min(delta):.4f}, {np.max(delta):.4f}]")

    # Evaluation for all methods
    print("[INFO] Evaluating all methods...")
    metrics = {}
    for name, P in [
        ("fieldcert_crc", pred_crc),
        ("global_bump", pred_global),
        ("morph_cp", pred_morph),
        ("pixel_cp", pred_pixelcp),
    ]:
        print(f"[INFO] Evaluating {name}...")
        rep = evaluate_set_masks(P, truth_test, weights=Wts, compute_ci=False)
        metrics[name] = rep
        print(f"  FPA: {rep['fpa']:.4f}, FNA: {rep['fna']:.4f}, IoU: {rep['iou']:.4f}")

    # Save results
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    results = {
        "subset": os.path.basename(args.subset),
        "variable": str(data["variable"]),
        "lead_hours": int(data["lead_hours"]),
        "years": [int(x) for x in data["years"]],
        "threshold": float(args.threshold),
        "alpha": float(args.alpha),
        "train_frac": float(args.train_frac),
        "crc_lambda_star": float(lam_star),
        "tau_star": float(tau_star),
        "r_star": int(r_star),
        "metrics": metrics,
        "data_shapes": {
            "calibration": [len(idx_cal), H, W],
            "test": [len(idx_test), H, W]
        }
    }
    
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[OK] Metrics saved to {args.out_json}")
    print(f"[OK] Results summary:")
    for method, res in metrics.items():
        print(f"  {method:15s}: FPA={res['fpa']:.4f}, FNA={res['fna']:.4f}, IoU={res['iou']:.4f}")

if __name__ == "__main__":
    main()
