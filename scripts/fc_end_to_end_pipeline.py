#!/usr/bin/env python3
"""
FieldCert End-to-End Pipeline

This script runs the complete FieldCert evaluation pipeline from start to finish:
1. Download local Zarr datasets (targeted subsets)
2. Run enhanced benchmarking with logging and case studies
3. Generate publication-ready figures with JSON sidecars

Includes a small test configuration to validate the workflow without errors.
"""

from __future__ import annotations
import os
import sys
import argparse
import subprocess
import json
import time
from typing import List, Dict, Any
from pathlib import Path

# Configuration for test vs full runs
TEST_CONFIG = {
    "provider": "ifs_mean",
    "variable": "10m_wind_speed", 
    "lead_hours": [24],
    "years": "2020-2020",  # Single year for test
    "threshold": 20.0,
    "alpha": 0.10,
    "time_chunk": 32,  # Smaller chunks for test
    "examples_topk": 2,  # Fewer examples for test
    "max_cases": 2,
}

FULL_CONFIG = {
    "providers": ["ifs_mean", "graphcast2020"],
    "variables": ["10m_wind_speed", "total_precipitation_24hr"],
    "lead_hours": [24, 48, 72],
    "years": "2019-2022",
    "thresholds": [15.0, 20.0, 25.0],  # Multiple thresholds
    "alphas": [0.05, 0.10, 0.20],
    "time_chunk": 64,
    "examples_topk": 5,
    "max_cases": 4,
}

class FieldCertPipeline:
    def __init__(self, config: Dict[str, Any], base_dir: str = "/workspace", test_mode: bool = True):
        self.config = config
        self.base_dir = Path(base_dir)
        self.test_mode = test_mode
        self.data_dir = self.base_dir / "data" / "local_wb2"
        self.results_dir = self.base_dir / "results" / ("test_pipeline" if test_mode else "full_pipeline")
        self.figs_dir = self.results_dir / "figs"
        
        # Ensure directories exist
        for d in [self.data_dir, self.results_dir, self.figs_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """Simple logging with timestamps."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status."""
        self.log(f"Starting: {description}")
        self.log(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.log(f"✅ Completed: {description}")
            if result.stdout.strip():
                self.log(f"Output: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Failed: {description}", "ERROR")
            self.log(f"Error: {e.stderr}", "ERROR")
            return False
    
    def step1_download_data(self) -> bool:
        """Step 1: Download local Zarr datasets."""
        self.log("=" * 60)
        self.log("STEP 1: DOWNLOADING DATASETS")
        self.log("=" * 60)
        
        if self.test_mode:
            # Test mode: single provider/variable/lead
            provider = self.config["provider"]
            variable = self.config["variable"]
            leads = self.config["lead_hours"]
            years = self.config["years"]
            
            for lead in leads:
                cmd = [
                    "python", "scripts/wb2_download_local.py",
                    "--provider", provider,
                    "--variable", variable,
                    "--lead_hours", str(lead),
                    "--years", years,
                    "--out_root", str(self.data_dir),
                    "--time_chunk", str(self.config["time_chunk"])
                ]
                
                if not self.run_command(cmd, f"Download {provider}/{variable}/{lead}h"):
                    return False
        else:
            # Full mode: multiple providers/variables/leads
            providers = self.config["providers"]
            variables = self.config["variables"]
            leads = self.config["lead_hours"]
            years = self.config["years"]
            
            for provider in providers:
                for variable in variables:
                    for lead in leads:
                        cmd = [
                            "python", "scripts/wb2_download_local.py",
                            "--provider", provider,
                            "--variable", variable,
                            "--lead_hours", str(lead),
                            "--years", years,
                            "--out_root", str(self.data_dir),
                            "--time_chunk", str(self.config["time_chunk"])
                        ]
                        
                        if not self.run_command(cmd, f"Download {provider}/{variable}/{lead}h"):
                            return False
        
        self.log("✅ All datasets downloaded successfully")
        return True
    
    def step2_run_benchmarks(self) -> bool:
        """Step 2: Run enhanced benchmarking with logging."""
        self.log("=" * 60)
        self.log("STEP 2: RUNNING BENCHMARKS")
        self.log("=" * 60)
        
        if self.test_mode:
            # Test mode: single configuration
            provider = self.config["provider"]
            variable = self.config["variable"]
            leads = self.config["lead_hours"]
            years = self.config["years"]
            threshold = self.config["threshold"]
            alpha = self.config["alpha"]
            
            for lead in leads:
                # Build paths
                tag = f"{provider}__{variable}__L{lead}h__{years}"
                forecast_zarr = self.data_dir / "forecast" / f"{tag}.zarr"
                truth_zarr = self.data_dir / "truth" / f"{tag}.zarr"
                
                # Output paths
                base_name = f"{provider}_{variable}_L{lead}h_T{threshold}"
                summary_json = self.results_dir / f"{base_name}.summary.json"
                times_jsonl = self.results_dir / f"{base_name}.times.jsonl"
                examples_dir = self.results_dir / "examples"
                
                cmd = [
                    "python", "scripts/fc_run_local_benchmark_enhanced.py",
                    "--forecast_zarr", str(forecast_zarr),
                    "--truth_zarr", str(truth_zarr),
                    "--variable", variable,
                    "--lead_hours", str(lead),
                    "--threshold", str(threshold),
                    "--alpha", str(alpha),
                    "--time_chunk", str(self.config["time_chunk"]),
                    "--log_time_jsonl", str(times_jsonl),
                    "--examples_topk", str(self.config["examples_topk"]),
                    "--examples_dir", str(examples_dir),
                    "--out_json", str(summary_json)
                ]
                
                if not self.run_command(cmd, f"Benchmark {provider}/{variable}/{lead}h"):
                    return False
        else:
            # Full mode: multiple configurations
            providers = self.config["providers"]
            variables = self.config["variables"]
            leads = self.config["lead_hours"]
            years = self.config["years"]
            thresholds = self.config["thresholds"]
            alphas = self.config["alphas"]
            
            for provider in providers:
                for variable in variables:
                    for lead in leads:
                        for threshold in thresholds:
                            for alpha in alphas:
                                # Build paths
                                tag = f"{provider}__{variable}__L{lead}h__{years}"
                                forecast_zarr = self.data_dir / "forecast" / f"{tag}.zarr"
                                truth_zarr = self.data_dir / "truth" / f"{tag}.zarr"
                                
                                # Output paths
                                base_name = f"{provider}_{variable}_L{lead}h_T{threshold}_A{alpha}"
                                summary_json = self.results_dir / f"{base_name}.summary.json"
                                times_jsonl = self.results_dir / f"{base_name}.times.jsonl"
                                examples_dir = self.results_dir / "examples"
                                
                                cmd = [
                                    "python", "scripts/fc_run_local_benchmark_enhanced.py",
                                    "--forecast_zarr", str(forecast_zarr),
                                    "--truth_zarr", str(truth_zarr),
                                    "--variable", variable,
                                    "--lead_hours", str(lead),
                                    "--threshold", str(threshold),
                                    "--alpha", str(alpha),
                                    "--time_chunk", str(self.config["time_chunk"]),
                                    "--log_time_jsonl", str(times_jsonl),
                                    "--examples_topk", str(self.config["examples_topk"]),
                                    "--examples_dir", str(examples_dir),
                                    "--out_json", str(summary_json)
                                ]
                                
                                if not self.run_command(cmd, f"Benchmark {provider}/{variable}/{lead}h/T{threshold}/A{alpha}"):
                                    return False
        
        self.log("✅ All benchmarks completed successfully")
        return True
    
    def step3_create_summary_csv(self) -> bool:
        """Step 3: Create summary CSV from individual JSON results."""
        self.log("=" * 60)
        self.log("STEP 3: CREATING SUMMARY CSV")
        self.log("=" * 60)
        
        # Find all summary JSON files
        json_files = list(self.results_dir.glob("*.summary.json"))
        if not json_files:
            self.log("❌ No summary JSON files found", "ERROR")
            return False
        
        # Create summary CSV structure
        import pandas as pd
        
        rows = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract metadata
                common = {
                    "variable": data["variable"],
                    "lead_hours": data["lead_hours"],
                    "threshold": data["threshold"],
                    "alpha": data["alpha"],
                    "n_cal": data["n_cal"],
                    "n_test": data.get("time_len", 0) - data["n_cal"],
                }
                
                # Extract metrics for each method
                for method, metrics in data["metrics"].items():
                    row = dict(common)
                    row["method"] = method
                    row["fpa"] = metrics["fpa"]
                    row["fna"] = metrics["fna"]
                    row["iou"] = metrics["iou"]
                    
                    # Add method-specific parameters
                    if method == "fieldcert_crc":
                        row["lambda_star"] = data.get("lambda_star")
                    elif method == "global_bump":
                        row["tau_star"] = data.get("tau_star")
                    elif method == "morph_cp":
                        row["r_star"] = data.get("r_star")
                    elif method == "prob_isotonic":
                        row["p_star"] = data.get("p_star")
                    
                    rows.append(row)
                    
            except Exception as e:
                self.log(f"❌ Error processing {json_file}: {e}", "ERROR")
                continue
        
        if not rows:
            self.log("❌ No valid data extracted from JSON files", "ERROR")
            return False
        
        # Create DataFrame and save CSV
        df = pd.DataFrame(rows)
        summary_csv = self.results_dir / "summary.csv"
        df.to_csv(summary_csv, index=False)
        
        self.log(f"✅ Created summary CSV with {len(rows)} rows: {summary_csv}")
        return True
    
    def step4_generate_figures(self) -> bool:
        """Step 4: Generate publication-ready figures."""
        self.log("=" * 60)
        self.log("STEP 4: GENERATING FIGURES")
        self.log("=" * 60)
        
        # Check if summary CSV exists
        summary_csv = self.results_dir / "summary.csv"
        if not summary_csv.exists():
            self.log("❌ Summary CSV not found", "ERROR")
            return False
        
        # Check for examples directory
        examples_dir = self.results_dir / "examples"
        examples_arg = str(examples_dir) if examples_dir.exists() else None
        
        if self.test_mode:
            # Test mode: simple configuration
            providers = [self.config["provider"]]
            variables = [self.config["variable"]]
            leads = self.config["lead_hours"]
            alphas = [self.config["alpha"]]
        else:
            # Full mode: comprehensive configuration
            providers = self.config["providers"]
            variables = self.config["variables"]
            leads = self.config["lead_hours"]
            alphas = self.config["alphas"]
        
        cmd = [
            "python", "scripts/figs_generate.py",
            "--summaries_glob", str(summary_csv),
            "--providers", ",".join(providers),
            "--variables", ",".join(variables),
            "--leads", ",".join(map(str, leads)),
            "--alphas", ",".join(map(str, alphas)),
            "--outdir", str(self.figs_dir),
            "--max_cases", str(self.config["max_cases"])
        ]
        
        if examples_arg:
            cmd.extend(["--cases_dir", examples_arg])
        
        if not self.run_command(cmd, "Generate figures"):
            return False
        
        self.log("✅ All figures generated successfully")
        return True
    
    def run_pipeline(self) -> bool:
        """Run the complete end-to-end pipeline."""
        start_time = time.time()
        
        self.log("🚀 Starting FieldCert End-to-End Pipeline")
        self.log(f"Mode: {'TEST' if self.test_mode else 'FULL'}")
        self.log(f"Base directory: {self.base_dir}")
        self.log(f"Results directory: {self.results_dir}")
        
        # Run all steps
        steps = [
            ("Download Data", self.step1_download_data),
            ("Run Benchmarks", self.step2_run_benchmarks),
            ("Create Summary CSV", self.step3_create_summary_csv),
            ("Generate Figures", self.step4_generate_figures),
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                self.log(f"❌ Pipeline failed at step: {step_name}", "ERROR")
                return False
        
        # Success!
        elapsed = time.time() - start_time
        self.log("=" * 60)
        self.log("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        self.log(f"Total time: {elapsed:.1f} seconds")
        self.log(f"Results: {self.results_dir}")
        self.log(f"Figures: {self.figs_dir}")
        self.log("=" * 60)
        
        return True

def main():
    parser = argparse.ArgumentParser(description="FieldCert End-to-End Pipeline")
    parser.add_argument("--mode", choices=["test", "full"], default="test",
                       help="Run mode: 'test' for small validation, 'full' for complete evaluation")
    parser.add_argument("--base_dir", default="/workspace",
                       help="Base directory for data and results")
    parser.add_argument("--config_file", default=None,
                       help="Optional JSON config file to override defaults")
    
    args = parser.parse_args()
    
    # Select configuration
    if args.mode == "test":
        config = TEST_CONFIG
        test_mode = True
    else:
        config = FULL_CONFIG
        test_mode = False
    
    # Override with config file if provided
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            custom_config = json.load(f)
        config.update(custom_config)
    
    # Run pipeline
    pipeline = FieldCertPipeline(config, args.base_dir, test_mode)
    success = pipeline.run_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
