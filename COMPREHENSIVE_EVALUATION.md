# FieldCert-Weather Comprehensive Evaluation

This document describes the comprehensive evaluation framework for demonstrating FieldCert's superiority across multiple simulation methods and UQ approaches on the full WeatherBench-2 dataset.

## 🎯 Objective

**Demonstrate that FieldCert consistently outperforms other UQ methods regardless of the underlying base model (GraphCast, SFNO, neural operators, etc.).**

## 🚀 Quick Start

### Run Comprehensive Evaluation

```bash
# Quick test (500 samples, 1 year)
make comprehensive-quick

# Standard evaluation (2 years)
make comprehensive

# Full-scale evaluation (3 years, all data)
make comprehensive-full
```

### Custom Configuration

```bash
# Custom threshold and years
python scripts/fc_comprehensive_eval.py \
    --output results/custom \
    --variable 10m_wind_speed \
    --threshold 25.0 \
    --years 2020 2021 2022 \
    --max-samples 1000
```

## 📊 Evaluation Matrix

### Base Simulation Models

| Model | Type | Ensemble Size | Description |
|-------|------|---------------|-------------|
| **IFS_ENS** | Ensemble | 50 | ECMWF IFS Ensemble (Operational) |
| **GraphCast** | Neural Operator | 1 | Google's Neural Weather Model |
| **Pangu-Weather** | Neural Operator | 1 | AI Weather Model |
| **FourCastNet** | Neural Operator | 1 | NVIDIA's Neural Weather Model |

### UQ Methods

| Method | Type | Description |
|--------|------|-------------|
| **FieldCert_CRC** | Physics-Aware | FieldCert with Conformal Risk Control |
| **Global_Bump** | Baseline | Global Threshold Bump |
| **Morphological_CP** | Baseline | Morphological Conformal Prediction |
| **Pixelwise_CP** | Baseline | Pixelwise Split Conformal Prediction |

**Total Experiments**: 4 models × 4 UQ methods = **16 experiments**

## 🔬 Expected Results

FieldCert should demonstrate:

- ✅ **Lower False Positive Area (FPA)** than all baselines
- ✅ **Better spatial coverage (IoU)** for extreme events  
- ✅ **Consistent performance** across all base models
- ✅ **Physics-aware uncertainty quantification** using spatial gradients
- ✅ **Superior calibration** with conformal risk control

## 📈 Output Analysis

### Generated Files

```
results/comprehensive/
├── comprehensive_results.json          # Raw results
├── summary_fpa_mean.csv               # FPA comparison table
├── summary_fna_mean.csv               # FNA comparison table  
├── summary_iou_mean.csv               # IoU comparison table
├── fieldcert_superiority_analysis.csv # FieldCert vs baselines
└── plots/
    ├── fpa_vs_fna.png                 # Performance scatter plot
    ├── iou_comparison.png             # IoU bar chart
    └── fieldcert_superiority.png      # Improvement distributions
```

### Key Metrics

- **FPA (False Positive Area)**: Lower is better
- **FNA (False Negative Area)**: Lower is better  
- **IoU (Intersection over Union)**: Higher is better

### Publication-Ready Tables

LaTeX tables are automatically generated:
- `summary_fpa_mean.tex`
- `summary_fna_mean.tex` 
- `summary_iou_mean.tex`

## 🛠 Configuration

### Adding New Models

Edit `scripts/fc_multi_model_config.py`:

```python
def get_new_model_config() -> ModelConfig:
    return ModelConfig(
        name="NewModel",
        forecast_path="gs://path/to/model/data.zarr",
        ensemble_size=1,
        description="Description of New Model",
        model_type="neural_operator"
    )
```

### Adding New UQ Methods

```python
def get_new_uq_config() -> UQMethodConfig:
    return UQMethodConfig(
        name="NewUQ",
        method_type="new_method",
        description="Description of New UQ Method"
    )
```

## 🎛 Advanced Usage

### Multi-Variable Evaluation

```python
from fc_multi_model_config import get_multi_variable_configs

configs = get_multi_variable_configs()
for config in configs:
    evaluator = ComprehensiveEvaluator(config)
    evaluator.run_all_experiments()
```

### Custom Experiment Matrix

```python
from fc_comprehensive_eval import ComprehensiveEvaluator, ExperimentConfig
from fc_multi_model_config import get_comprehensive_config

# Custom configuration
config = get_comprehensive_config(
    variable="2m_temperature",
    threshold=35.0,  # 35°C extreme heat
    years=[2020, 2021, 2022],
    max_samples=None,  # Full dataset
    output_dir="results/temperature_extremes"
)

evaluator = ComprehensiveEvaluator(config)
results = evaluator.run_all_experiments()
evaluator.generate_analysis(results)
```

## 📊 Performance Benchmarks

### Runtime Estimates

| Configuration | Samples | Models | UQ Methods | Est. Runtime |
|---------------|---------|--------|------------|--------------|
| Quick Test | 500 | 1 | 4 | ~10 minutes |
| Standard | ~1500 | 1 | 4 | ~30 minutes |
| Full Scale | ~5000+ | 1 | 4 | ~2 hours |
| Multi-Model | ~1500 | 4 | 4 | ~2 hours |

### Memory Requirements

- **Quick Test**: ~2 GB RAM
- **Standard**: ~8 GB RAM  
- **Full Scale**: ~16 GB RAM
- **Multi-Model**: ~32 GB RAM

## 🔧 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `max_samples` or use `comprehensive-quick`
2. **Dataset Access**: Ensure proper GCS credentials
3. **Missing Dependencies**: Run `make setup-env`

### Debug Mode

```bash
# Enable verbose logging
export PYTHONPATH=/workspace/fieldcert/src
python scripts/fc_comprehensive_eval.py --output results/debug --max-samples 100
```

## 📚 Scientific Interpretation

### FieldCert Advantages

1. **Physics-Aware Margins**: Uses spatial gradients to capture physical structure
2. **Conformal Risk Control**: Provides finite-sample guarantees
3. **Model-Agnostic**: Works with any base forecast model
4. **Extreme Event Focus**: Optimized for rare, high-impact events

### Baseline Comparisons

- **Global Bump**: Simple but ignores spatial structure
- **Morphological CP**: Spatial but not physics-aware
- **Pixelwise CP**: Independent pixels, no spatial coherence

### Expected Publication Results

> "FieldCert demonstrates consistent superiority across all tested simulation methods, achieving X% lower false positive rates and Y% better spatial coverage compared to state-of-the-art baselines, while maintaining robust calibration guarantees for extreme weather events."

## 🎯 Next Steps

1. **Scale to More Models**: Add GraphCast, SFNO, other neural operators
2. **Multi-Variable Analysis**: Temperature, precipitation, pressure
3. **Regime Stratification**: Performance by season, region, weather type
4. **Ablation Studies**: Margin methods, block assignments, hyperparameters
5. **Real-Time Deployment**: Operational weather forecasting integration

## 🤝 Contributing

To add new models or UQ methods:

1. Implement data loading in `fc_comprehensive_eval.py`
2. Add configuration in `fc_multi_model_config.py`
3. Update this documentation
4. Test with `comprehensive-quick`
5. Submit pull request

---

**This comprehensive evaluation framework provides the foundation for demonstrating FieldCert's superiority across the entire landscape of modern weather forecasting methods.**
