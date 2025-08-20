#!/usr/bin/env python3
"""
FieldCert-Weather Comprehensive Evaluation Script

This script runs a thorough evaluation of FieldCert against multiple UQ methods
across multiple base simulation models on the full WeatherBench-2 dataset.

Usage:
    python fc_comprehensive_eval.py --output results/comprehensive/
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from calibration.crc import CRCCalibrator
from calibration.margins import build_margin_field
from calibration.morphology import threshold_bump_mask
from calibration.regimes import assign_blocks_by_time
from baselines.global_bump import calibrate_global_bump_by_fpa, apply_global_bump
from baselines.morph_cp import calibrate_morph_radius_by_fpa, apply_morph_radius
from baselines.pixelwise_cp import calibrate_pixelwise_delta, apply_pixelwise_delta
from eval.wb2 import load_wb2_forecast, load_wb2_observations, select_lead_time
from eval.metrics import compute_area_weights, evaluate_set_valued_forecast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a base simulation model."""
    name: str
    forecast_path: str
    ensemble_size: int
    description: str
    model_type: str = "ensemble"

@dataclass
class UQMethodConfig:
    """Configuration for a UQ method."""
    name: str
    method_type: str
    description: str
    hyperparams: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    """Configuration for the comprehensive experiment."""
    variable: str = "10m_wind_speed"
    lead_hours: int = 24
    years: List[int] = field(default_factory=lambda: [2020, 2021])
    threshold: float = 20.0
    alpha: float = 0.10
    train_frac: float = 0.7
    max_samples: Optional[int] = None  # None = full dataset
    output_dir: str = "results/comprehensive"
    models: List[ModelConfig] = field(default_factory=list)
    uq_methods: List[UQMethodConfig] = field(default_factory=list)

class ComprehensiveEvaluator:
    """Main class for running comprehensive FieldCert evaluation."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_output_dir()
        
    def setup_output_dir(self):
        """Create output directory structure."""
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "metrics").mkdir(exist_ok=True)
        (self.output_path / "plots").mkdir(exist_ok=True)
        
    def load_data(self, model_config: ModelConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load forecast and observation data for a given model."""
        logger.info(f"Loading data for model: {model_config.name}")
        
        # Load forecast data
        ds_fc = load_wb2_forecast(model_config.forecast_path, variable=self.config.variable)
        ds_fc = select_lead_time(ds_fc, self.config.lead_hours)
        
        # Load observations
        obs_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
        ds_obs = load_wb2_observations(obs_path, variable=self.config.variable)
        
        # Filter by years and align times
        year_mask = ds_fc.time.dt.year.isin(self.config.years)
        ds_fc = ds_fc.isel(time=year_mask)
        
        valid_times = ds_fc.time + ds_fc.prediction_timedelta
        ds_obs_aligned = ds_obs.sel(time=valid_times, method='nearest')
        
        # Convert to numpy arrays
        yhat = ds_fc.values
        y = ds_obs_aligned.values
        lat = ds_fc.latitude.values
        lon = ds_fc.longitude.values
        
        # Limit samples if specified
        if self.config.max_samples is not None:
            n_samples = min(self.config.max_samples, yhat.shape[0])
            yhat = yhat[:n_samples]
            y = y[:n_samples]
            
        logger.info(f"Data shapes - forecast: {yhat.shape}, obs: {y.shape}")
        return yhat, y, lat, lon
        
    def prepare_ensemble_data(self, yhat: np.ndarray) -> np.ndarray:
        """Prepare ensemble data for UQ methods."""
        if yhat.ndim == 5:  # (T, ensemble, lead, H, W)
            yhat_mean = yhat.mean(axis=1)  # (T, lead, H, W)
            if yhat_mean.shape[1] == 1:
                yhat_mean = yhat_mean.squeeze(axis=1)  # (T, H, W)
            return yhat_mean
        elif yhat.ndim == 4:  # (T, lead, H, W)
            if yhat.shape[1] == 1:
                return yhat.squeeze(axis=1)
            else:
                raise ValueError("Multiple lead times not supported")
        elif yhat.ndim == 3:  # (T, H, W)
            return yhat
        else:
            raise ValueError(f"Unexpected forecast shape: {yhat.shape}")
            
    def apply_uq_method(
        self, 
        method_config: UQMethodConfig,
        yhat_cal: np.ndarray,
        y_cal: np.ndarray,
        yhat_test: np.ndarray,
        spatial_weights: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply a UQ method and return predictions and metadata."""
        
        method_name = method_config.method_type
        logger.info(f"Applying UQ method: {method_name}")
        
        if method_name == "fieldcert":
            return self._apply_fieldcert(yhat_cal, y_cal, yhat_test, spatial_weights)
        elif method_name == "global_bump":
            return self._apply_global_bump(yhat_cal, y_cal, yhat_test, spatial_weights)
        elif method_name == "morph_cp":
            return self._apply_morph_cp(yhat_cal, y_cal, yhat_test, spatial_weights)
        elif method_name == "pixelwise_cp":
            return self._apply_pixelwise_cp(yhat_cal, y_cal, yhat_test, spatial_weights)
        else:
            raise ValueError(f"Unknown UQ method: {method_name}")
            
    def _apply_fieldcert(
        self, yhat_cal, y_cal, yhat_test, spatial_weights
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply FieldCert CRC method."""
        
        # Build physics-aware margins
        margins_cal = build_margin_field(yhat_cal, method="grad_mag", normalize=True)
        margins_test = build_margin_field(yhat_test, method="grad_mag", normalize=True)
        
        # Assign calibration blocks
        block_ids = assign_blocks_by_time(len(yhat_cal), n_blocks=5)
        
        # CRC calibration
        crc = CRCCalibrator()
        crc_result = crc.fit_for_regime(
            preds=yhat_cal,
            truths=y_cal,
            threshold=self.config.threshold,
            margins=margins_cal,
            block_ids=block_ids,
            spatial_weights=spatial_weights
        )
        
        # Apply to test data
        pred_masks = threshold_bump_mask(
            pred_field=yhat_test,
            threshold=self.config.threshold,
            margin_field=margins_test,
            lam=crc_result.lambda_star
        )
        
        metadata = {
            "lambda_star": crc_result.lambda_star,
            "method": "fieldcert_crc",
            "n_blocks": 5
        }
        
        return pred_masks, metadata
        
    def _apply_global_bump(
        self, yhat_cal, y_cal, yhat_test, spatial_weights
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply global bump baseline."""
        
        tau_star, _, _ = calibrate_global_bump_by_fpa(
            yhat_cal, y_cal, self.config.threshold,
            alpha=self.config.alpha, spatial_weights=spatial_weights
        )
        
        pred_masks = apply_global_bump(yhat_test, self.config.threshold, tau_star)
        
        metadata = {
            "tau_star": tau_star,
            "method": "global_bump"
        }
        
        return pred_masks, metadata
        
    def _apply_morph_cp(
        self, yhat_cal, y_cal, yhat_test, spatial_weights
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply morphological conformal prediction."""
        
        r_star, _, _ = calibrate_morph_radius_by_fpa(
            yhat_cal, y_cal, self.config.threshold,
            alpha=self.config.alpha, spatial_weights=spatial_weights
        )
        
        pred_masks = apply_morph_radius(yhat_test, self.config.threshold, int(r_star))
        
        metadata = {
            "radius_star": r_star,
            "method": "morph_cp"
        }
        
        return pred_masks, metadata
        
    def _apply_pixelwise_cp(
        self, yhat_cal, y_cal, yhat_test, spatial_weights
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply pixelwise split conformal prediction."""
        
        delta = calibrate_pixelwise_delta(
            yhat_cal, y_cal, self.config.threshold,
            alpha_pixel=self.config.alpha
        )
        
        pred_masks = apply_pixelwise_delta(yhat_test, self.config.threshold, delta)
        
        metadata = {
            "delta_range": [float(np.min(delta)), float(np.max(delta))],
            "method": "pixelwise_cp"
        }
        
        return pred_masks, metadata
        
    def evaluate_predictions(
        self, 
        pred_masks: np.ndarray, 
        truth_masks: np.ndarray, 
        spatial_weights: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate set-valued predictions."""
        
        metrics = {}
        n_test = len(pred_masks)
        
        for i in range(n_test):
            result = evaluate_set_valued_forecast(
                pred_masks[i], truth_masks[i], spatial_weights
            )
            for key, value in result.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
                
        # Compute summary statistics
        summary_metrics = {}
        for key, values in metrics.items():
            summary_metrics[f"{key}_mean"] = float(np.mean(values))
            summary_metrics[f"{key}_std"] = float(np.std(values))
            summary_metrics[f"{key}_median"] = float(np.median(values))
            
        return summary_metrics
        
    def run_single_experiment(
        self, model_config: ModelConfig, uq_config: UQMethodConfig
    ) -> Dict[str, Any]:
        """Run a single (model, UQ method) experiment."""
        
        logger.info(f"Running experiment: {model_config.name} + {uq_config.name}")
        
        # Load data
        yhat, y, lat, lon = self.load_data(model_config)
        
        # Prepare ensemble data
        yhat_mean = self.prepare_ensemble_data(yhat)
        
        # Compute spatial weights
        import xarray as xr
        dummy_ds = xr.Dataset(coords={"latitude": lat, "longitude": lon})
        spatial_weights = compute_area_weights(dummy_ds, lat_name="latitude", normalize=True)
        
        # Ensure weights match data shape
        H, W = yhat_mean.shape[-2:]
        if spatial_weights.shape != (H, W):
            spatial_weights = spatial_weights.T
            
        # Split data
        n_train = int(np.floor(self.config.train_frac * len(yhat_mean)))
        yhat_cal = yhat_mean[:n_train]
        yhat_test = yhat_mean[n_train:]
        y_cal = y[:n_train]
        y_test = y[n_train:]
        
        # Apply UQ method
        start_time = time.time()
        pred_masks, metadata = self.apply_uq_method(
            uq_config, yhat_cal, y_cal, yhat_test, spatial_weights
        )
        runtime = time.time() - start_time
        
        # Evaluate predictions
        truth_masks = (y_test >= self.config.threshold)
        metrics = self.evaluate_predictions(pred_masks, truth_masks, spatial_weights)
        
        # Compile results
        result = {
            "model": model_config.name,
            "uq_method": uq_config.name,
            "model_description": model_config.description,
            "uq_description": uq_config.description,
            "runtime_seconds": runtime,
            "n_calibration": len(yhat_cal),
            "n_test": len(yhat_test),
            "data_shape": yhat_mean.shape,
            "metadata": metadata,
            "metrics": metrics
        }
        
        return result
        
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all (model, UQ method) combinations."""
        
        logger.info("Starting comprehensive evaluation...")
        logger.info(f"Models: {[m.name for m in self.config.models]}")
        logger.info(f"UQ methods: {[u.name for u in self.config.uq_methods]}")
        
        all_results = []
        
        # Run all combinations
        total_experiments = len(self.config.models) * len(self.config.uq_methods)
        pbar = tqdm(total=total_experiments, desc="Running experiments")
        
        for model_config in self.config.models:
            for uq_config in self.config.uq_methods:
                try:
                    result = self.run_single_experiment(model_config, uq_config)
                    all_results.append(result)
                    logger.info(f"Completed: {model_config.name} + {uq_config.name}")
                except Exception as e:
                    logger.error(f"Failed: {model_config.name} + {uq_config.name}: {e}")
                    
                pbar.update(1)
                
        pbar.close()
        
        # Save results
        results_summary = {
            "config": self.config.__dict__,
            "results": all_results,
            "n_experiments": len(all_results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to JSON
        results_file = self.output_path / "comprehensive_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
            
        logger.info(f"Results saved to: {results_file}")
        
        return results_summary
        
    def generate_analysis(self, results_summary: Dict[str, Any]):
        """Generate comprehensive analysis and plots."""
        
        logger.info("Generating analysis and plots...")
        
        # Convert results to DataFrame
        df = pd.DataFrame(results_summary["results"])
        
        # Extract key metrics
        for metric in ["fpa_mean", "fna_mean", "iou_mean"]:
            df[metric] = df["metrics"].apply(lambda x: x.get(metric, np.nan))
            
        # Generate summary table
        self._generate_summary_table(df)
        
        # Generate plots
        self._generate_performance_plots(df)
        self._generate_superiority_analysis(df)
        
    def _generate_summary_table(self, df: pd.DataFrame):
        """Generate summary performance table."""
        
        # Create pivot table for each metric
        for metric in ["fpa_mean", "fna_mean", "iou_mean"]:
            pivot = df.pivot(index="model", columns="uq_method", values=metric)
            
            # Save to CSV
            csv_file = self.output_path / f"summary_{metric}.csv"
            pivot.to_csv(csv_file)
            
            # Create LaTeX table
            latex_file = self.output_path / f"summary_{metric}.tex"
            with open(latex_file, 'w') as f:
                f.write(pivot.to_latex(float_format="%.4f"))
                
        logger.info("Summary tables saved to CSV and LaTeX formats")
        
    def _generate_performance_plots(self, df: pd.DataFrame):
        """Generate performance comparison plots."""
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # FPA vs FNA scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model in df["model"].unique():
            model_data = df[df["model"] == model]
            ax.scatter(
                model_data["fpa_mean"], 
                model_data["fna_mean"],
                label=model,
                s=100,
                alpha=0.7
            )
            
            # Annotate points with UQ method names
            for _, row in model_data.iterrows():
                ax.annotate(
                    row["uq_method"],
                    (row["fpa_mean"], row["fna_mean"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8
                )
                
        ax.set_xlabel("False Positive Area (FPA)")
        ax.set_ylabel("False Negative Area (FNA)")
        ax.set_title("UQ Method Performance Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / "plots" / "fpa_vs_fna.png", dpi=300)
        plt.close()
        
        # IoU comparison bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pivot_iou = df.pivot(index="model", columns="uq_method", values="iou_mean")
        pivot_iou.plot(kind="bar", ax=ax, width=0.8)
        
        ax.set_xlabel("Base Model")
        ax.set_ylabel("Intersection over Union (IoU)")
        ax.set_title("IoU Performance by Model and UQ Method")
        ax.legend(title="UQ Method", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_path / "plots" / "iou_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Performance plots saved")
        
    def _generate_superiority_analysis(self, df: pd.DataFrame):
        """Generate FieldCert superiority analysis."""
        
        # Check if FieldCert is consistently better
        fieldcert_results = df[df["uq_method"].str.contains("fieldcert", case=False)]
        other_results = df[~df["uq_method"].str.contains("fieldcert", case=False)]
        
        if len(fieldcert_results) == 0:
            logger.warning("No FieldCert results found for superiority analysis")
            return
            
        # For each model, compare FieldCert vs other methods
        superiority_analysis = []
        
        for model in df["model"].unique():
            fc_model = fieldcert_results[fieldcert_results["model"] == model]
            other_model = other_results[other_results["model"] == model]
            
            if len(fc_model) == 0 or len(other_model) == 0:
                continue
                
            fc_iou = fc_model["iou_mean"].iloc[0]
            fc_fpa = fc_model["fpa_mean"].iloc[0]
            
            for _, other_row in other_model.iterrows():
                superiority_analysis.append({
                    "model": model,
                    "fieldcert_iou": fc_iou,
                    "other_method": other_row["uq_method"],
                    "other_iou": other_row["iou_mean"],
                    "iou_improvement": fc_iou - other_row["iou_mean"],
                    "fieldcert_fpa": fc_fpa,
                    "other_fpa": other_row["fpa_mean"],
                    "fpa_improvement": other_row["fpa_mean"] - fc_fpa
                })
                
        # Save superiority analysis
        if superiority_analysis:
            sup_df = pd.DataFrame(superiority_analysis)
            sup_df.to_csv(self.output_path / "fieldcert_superiority_analysis.csv", index=False)
            
            # Print summary statistics
            logger.info("FieldCert Superiority Analysis:")
            logger.info(f"  IoU improvements: mean={sup_df['iou_improvement'].mean():.4f}, "
                       f"positive={100*np.mean(sup_df['iou_improvement'] > 0):.1f}%")
            logger.info(f"  FPA improvements: mean={sup_df['fpa_improvement'].mean():.4f}, "
                       f"positive={100*np.mean(sup_df['fpa_improvement'] > 0):.1f}%")
                       
        logger.info("Superiority analysis saved")

def create_default_config() -> ExperimentConfig:
    """Create default experiment configuration."""
    
    # Define base models
    models = [
        ModelConfig(
            name="IFS_ENS",
            forecast_path="gs://weatherbench2/datasets/ifs_ens/2018-2022_0012_0p25_0p25.zarr",
            ensemble_size=50,
            description="ECMWF IFS Ensemble",
            model_type="ensemble"
        ),
        # Add more models for comprehensive evaluation
        # ModelConfig(
        #     name="GraphCast",
        #     forecast_path="path/to/graphcast/predictions.zarr",
        #     ensemble_size=1,
        #     description="GraphCast Neural Weather Model",
        #     model_type="neural_operator"
        # ),
    ]
    
    # Define UQ methods
    uq_methods = [
        UQMethodConfig(
            name="FieldCert_CRC",
            method_type="fieldcert",
            description="FieldCert with Conformal Risk Control"
        ),
        UQMethodConfig(
            name="Global_Bump",
            method_type="global_bump", 
            description="Global Threshold Bump Baseline"
        ),
        UQMethodConfig(
            name="Morphological_CP",
            method_type="morph_cp",
            description="Morphological Conformal Prediction"
        ),
        UQMethodConfig(
            name="Pixelwise_CP",
            method_type="pixelwise_cp",
            description="Pixelwise Split Conformal Prediction"
        ),
    ]
    
    return ExperimentConfig(
        variable="10m_wind_speed",
        lead_hours=24,
        years=[2020, 2021],  # Full 2 years
        threshold=20.0,
        alpha=0.10,
        train_frac=0.7,
        max_samples=None,  # Use full dataset
        output_dir="results/comprehensive",
        models=models,
        uq_methods=uq_methods
    )

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive FieldCert-Weather Evaluation"
    )
    parser.add_argument(
        "--output", 
        default="results/comprehensive",
        help="Output directory for results"
    )
    parser.add_argument(
        "--variable",
        default="10m_wind_speed",
        help="Weather variable to evaluate"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="Threshold for extreme events"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2020, 2021],
        help="Years to include in evaluation"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples (None for full dataset)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_default_config()
    config.output_dir = args.output
    config.variable = args.variable
    config.threshold = args.threshold
    config.years = args.years
    config.max_samples = args.max_samples
    
    logger.info("Starting Comprehensive FieldCert-Weather Evaluation")
    logger.info(f"Configuration: {config}")
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(config)
    results_summary = evaluator.run_all_experiments()
    
    # Generate analysis
    evaluator.generate_analysis(results_summary)
    
    logger.info("Comprehensive evaluation completed successfully!")
    logger.info(f"Results saved to: {evaluator.output_path}")
    
    # Print final summary
    df = pd.DataFrame(results_summary["results"])
    logger.info("\n=== FINAL RESULTS SUMMARY ===")
    for model in df["model"].unique():
        logger.info(f"\nModel: {model}")
        model_results = df[df["model"] == model]
        for _, row in model_results.iterrows():
            metrics = row["metrics"]
            logger.info(f"  {row['uq_method']}: FPA={metrics.get('fpa_mean', 0):.4f}, "
                       f"FNA={metrics.get('fna_mean', 0):.4f}, IoU={metrics.get('iou_mean', 0):.4f}")

if __name__ == "__main__":
    main()
