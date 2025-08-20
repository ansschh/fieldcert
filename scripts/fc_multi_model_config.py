#!/usr/bin/env python3
"""
Multi-Model Configuration for Comprehensive FieldCert Evaluation

This script defines configurations for multiple simulation methods and UQ approaches
to demonstrate FieldCert's superiority across different base models.

Usage:
    from fc_multi_model_config import get_comprehensive_config
    config = get_comprehensive_config()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from fc_comprehensive_eval import ModelConfig, UQMethodConfig, ExperimentConfig

def get_ifs_ensemble_config() -> ModelConfig:
    """ECMWF IFS Ensemble - operational weather model."""
    return ModelConfig(
        name="IFS_ENS",
        forecast_path="gs://weatherbench2/datasets/ifs_ens/2018-2022_0012_0p25_0p25.zarr",
        ensemble_size=50,
        description="ECMWF IFS Ensemble (Operational)",
        model_type="ensemble"
    )

def get_pangu_config() -> ModelConfig:
    """Pangu-Weather - AI weather model."""
    return ModelConfig(
        name="Pangu_Weather",
        forecast_path="gs://weatherbench2/datasets/pangu/2020-2022_0012_0p25.zarr",
        ensemble_size=1,
        description="Pangu-Weather AI Model",
        model_type="neural_operator"
    )

def get_graphcast_config() -> ModelConfig:
    """GraphCast - Google's neural weather model."""
    return ModelConfig(
        name="GraphCast",
        forecast_path="gs://weatherbench2/datasets/graphcast/2020-2022_0012_0p25.zarr",
        ensemble_size=1,
        description="GraphCast Neural Weather Model",
        model_type="neural_operator"
    )

def get_fourcastnet_config() -> ModelConfig:
    """FourCastNet - NVIDIA's neural weather model."""
    return ModelConfig(
        name="FourCastNet",
        forecast_path="gs://weatherbench2/datasets/fourcastnet/2020-2022_0012_0p25.zarr",
        ensemble_size=1,
        description="FourCastNet Neural Weather Model",
        model_type="neural_operator"
    )

def get_all_models() -> List[ModelConfig]:
    """Get all available simulation models."""
    models = [
        get_ifs_ensemble_config(),
        # Uncomment as datasets become available
        # get_pangu_config(),
        # get_graphcast_config(), 
        # get_fourcastnet_config(),
    ]
    return models

def get_fieldcert_configs() -> List[UQMethodConfig]:
    """Get FieldCert UQ method configurations."""
    return [
        UQMethodConfig(
            name="FieldCert_CRC",
            method_type="fieldcert",
            description="FieldCert with Conformal Risk Control",
            hyperparams={"margin_method": "grad_mag", "n_blocks": 5}
        ),
        UQMethodConfig(
            name="FieldCert_CRC_Temporal",
            method_type="fieldcert",
            description="FieldCert CRC with Temporal Blocks",
            hyperparams={"margin_method": "grad_mag", "n_blocks": 10}
        ),
    ]

def get_baseline_uq_configs() -> List[UQMethodConfig]:
    """Get baseline UQ method configurations."""
    return [
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
        UQMethodConfig(
            name="Split_CP",
            method_type="pixelwise_cp",
            description="Standard Split Conformal Prediction",
            hyperparams={"split_type": "standard"}
        ),
    ]

def get_all_uq_methods() -> List[UQMethodConfig]:
    """Get all UQ methods for comprehensive evaluation."""
    return get_fieldcert_configs() + get_baseline_uq_configs()

def get_comprehensive_config(
    variable: str = "10m_wind_speed",
    threshold: float = 20.0,
    years: List[int] = None,
    max_samples: int = None,
    output_dir: str = "results/comprehensive"
) -> ExperimentConfig:
    """
    Create comprehensive experiment configuration.
    
    Args:
        variable: Weather variable to evaluate
        threshold: Threshold for extreme events
        years: Years to include (default: [2020, 2021])
        max_samples: Maximum samples (None for full dataset)
        output_dir: Output directory
        
    Returns:
        ExperimentConfig with all models and UQ methods
    """
    if years is None:
        years = [2020, 2021]
        
    return ExperimentConfig(
        variable=variable,
        lead_hours=24,
        years=years,
        threshold=threshold,
        alpha=0.10,
        train_frac=0.7,
        max_samples=max_samples,
        output_dir=output_dir,
        models=get_all_models(),
        uq_methods=get_all_uq_methods()
    )

def get_quick_test_config() -> ExperimentConfig:
    """Get configuration for quick testing (500 samples, 1 year)."""
    return get_comprehensive_config(
        years=[2020],
        max_samples=500,
        output_dir="results/quick_test"
    )

def get_full_scale_config() -> ExperimentConfig:
    """Get configuration for full-scale evaluation (all data, 3 years)."""
    return get_comprehensive_config(
        years=[2020, 2021, 2022],
        max_samples=None,  # Full dataset
        output_dir="results/full_scale"
    )

def get_multi_variable_configs() -> List[ExperimentConfig]:
    """Get configurations for multiple weather variables."""
    variables = [
        ("10m_wind_speed", 20.0),
        ("2m_temperature", 35.0),  # 35°C extreme heat
        ("total_precipitation_6hr", 10.0),  # 10mm/6hr heavy rain
    ]
    
    configs = []
    for variable, threshold in variables:
        config = get_comprehensive_config(
            variable=variable,
            threshold=threshold,
            output_dir=f"results/multi_variable/{variable}"
        )
        configs.append(config)
        
    return configs

def print_experiment_matrix():
    """Print the full experiment matrix."""
    config = get_comprehensive_config()
    
    print("=== COMPREHENSIVE FIELDCERT EVALUATION MATRIX ===")
    print(f"\nBase Models ({len(config.models)}):")
    for i, model in enumerate(config.models, 1):
        print(f"  {i}. {model.name}: {model.description}")
        print(f"     Path: {model.forecast_path}")
        print(f"     Type: {model.model_type}, Ensemble: {model.ensemble_size}")
        
    print(f"\nUQ Methods ({len(config.uq_methods)}):")
    for i, method in enumerate(config.uq_methods, 1):
        print(f"  {i}. {method.name}: {method.description}")
        print(f"     Type: {method.method_type}")
        
    print(f"\nTotal Experiments: {len(config.models)} × {len(config.uq_methods)} = {len(config.models) * len(config.uq_methods)}")
    print(f"Dataset: {config.variable}, Years: {config.years}")
    print(f"Threshold: {config.threshold}, Alpha: {config.alpha}")
    
    print("\n=== EXPECTED OUTCOMES ===")
    print("FieldCert should demonstrate:")
    print("  • Lower False Positive Area (FPA) than baselines")
    print("  • Better spatial coverage (IoU) for extreme events")
    print("  • Consistent performance across all base models")
    print("  • Physics-aware uncertainty quantification")
    print("  • Superior calibration with conformal risk control")

if __name__ == "__main__":
    print_experiment_matrix()
