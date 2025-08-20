# FieldCert End-to-End Pipeline Usage Guide

This guide explains how to use the complete FieldCert evaluation pipeline from data download to figure generation.

## 🚀 Quick Start

### Test Mode (Recommended First Run)
```bash
# Run small test to validate entire pipeline
python scripts/fc_end_to_end_pipeline.py --mode test --base_dir ./test_workspace

# Or with custom config
python scripts/fc_end_to_end_pipeline.py --mode test --config_file configs/test_config.json
```

### Full Production Mode
```bash
# Run complete evaluation (requires significant compute/storage)
python scripts/fc_end_to_end_pipeline.py --mode full --base_dir /workspace
```

## 📁 Directory Structure

After running, you'll have:
```
{base_dir}/
├── data/local_wb2/           # Downloaded Zarr datasets
│   ├── forecast/             # Forecast data
│   └── truth/                # ERA5 truth data
├── results/{mode}_pipeline/  # Benchmark results
│   ├── *.summary.json        # Per-run summaries
│   ├── *.times.jsonl         # Per-time metrics
│   ├── summary.csv           # Aggregated results
│   ├── examples/             # Case-study NPZ files
│   └── figs/                 # Generated figures
│       ├── F2_validity_*.pdf # Validity curves
│       ├── F3_frontier_*.pdf # Sharpness frontiers
│       ├── F6_case_*.pdf     # Case studies
│       └── *.json            # Figure data sidecars
```

## ⚙️ Configuration Options

### Test Configuration (Small & Fast)
- **Provider**: IFS ENS mean only
- **Variable**: 10m wind speed only  
- **Lead**: 24h only
- **Year**: 2020 only (single year)
- **Threshold**: Single threshold (15.0 m/s)
- **Alpha**: Single risk level (0.10)
- **Chunks**: Small (16 time steps)
- **Examples**: 2 case studies
- **Runtime**: ~10-30 minutes

### Full Configuration (Comprehensive)
- **Providers**: IFS ENS mean, GraphCast 2020
- **Variables**: 10m wind speed, 24h precipitation
- **Leads**: 24h, 48h, 72h
- **Years**: 2019-2022 (4 years)
- **Thresholds**: Multiple (15.0, 20.0, 25.0)
- **Alphas**: Multiple (0.05, 0.10, 0.20)
- **Chunks**: Large (64 time steps)
- **Examples**: 5 case studies
- **Runtime**: Several hours to days

## 🔧 Custom Configuration

Create a JSON config file to override defaults:

```json
{
  "provider": "graphcast2020",
  "variable": "total_precipitation_24hr",
  "lead_hours": [24, 48],
  "years": "2021-2021",
  "threshold": 10.0,
  "alpha": 0.05,
  "time_chunk": 32,
  "examples_topk": 3,
  "max_cases": 3
}
```

## 📊 Pipeline Steps

### Step 1: Data Download
- Downloads targeted Zarr subsets from WeatherBench-2
- Includes both forecast and aligned ERA5 truth data
- Uses chunked writes for memory safety
- **Output**: `{base_dir}/data/local_wb2/`

### Step 2: Benchmarking
- Runs FieldCert CRC + 4 baseline methods
- Processes data in memory-safe chunks
- Logs per-time metrics and case studies
- **Output**: JSON summaries + JSONL logs + NPZ examples

### Step 3: Summary Creation
- Aggregates individual JSON results into CSV
- Creates structured data for figure generation
- **Output**: `summary.csv`

### Step 4: Figure Generation
- Generates publication-ready figures (F2, F3, F6)
- Creates JSON sidecars with exact plotted data
- **Output**: PDF figures + JSON metadata

## 🎯 Expected Results

### Test Mode Success Indicators
- ✅ Downloads complete without errors
- ✅ All 5 UQ methods run successfully
- ✅ Metrics show reasonable values (FPA ≈ α)
- ✅ Figures generate without errors
- ✅ Total runtime < 1 hour

### Key Metrics to Check
- **FPA (False Positive Area)**: Should be ≈ α (e.g., 0.10)
- **FNA (False Negative Area)**: Lower is better
- **IoU (Intersection over Union)**: Higher is better
- **FieldCert should outperform baselines** in FNA/IoU

## 🐛 Troubleshooting

### Common Issues

**1. Dataset Access Errors**
```bash
# Check internet connection and retry
# WeatherBench-2 is publicly accessible via Google Cloud
```

**2. Memory Issues**
```bash
# Reduce time_chunk size in config
# Use test mode first to validate
```

**3. Import Errors**
```bash
# Install requirements
pip install -r requirements.txt
```

**4. Permission Errors**
```bash
# Ensure write permissions to base_dir
chmod -R 755 /workspace
```

### Debug Mode
Add `--verbose` flag (if implemented) or check individual script outputs:
```bash
# Test individual components
python scripts/wb2_download_local.py --help
python scripts/fc_run_local_benchmark_enhanced.py --help  
python scripts/figs_generate.py --help
```

## 🚀 Production Deployment

### RunPod/Cloud Setup
```bash
# 1. Clone repository
git clone https://github.com/ansschh/fieldcert.git
cd fieldcert

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run test pipeline
python scripts/fc_end_to_end_pipeline.py --mode test --base_dir /workspace

# 4. Run full pipeline (if test succeeds)
python scripts/fc_end_to_end_pipeline.py --mode full --base_dir /workspace
```

### Resource Requirements
- **Test Mode**: 8GB RAM, 10GB storage, 1-2 GPUs optional
- **Full Mode**: 32GB+ RAM, 100GB+ storage, 4+ GPUs recommended

## 📈 Scientific Validation

The pipeline should demonstrate:
1. **Risk Control**: FieldCert FPA ≈ target α
2. **Superior Sharpness**: FieldCert achieves lower FNA, higher IoU
3. **Spatial Fidelity**: Case studies show better extreme event capture
4. **Consistency**: Results hold across providers, variables, leads

## 📚 Next Steps

After successful pipeline completion:
1. **Analyze Results**: Review `summary.csv` and generated figures
2. **Scale Up**: Run full evaluation with more providers/variables
3. **Paper Writing**: Use figures F2, F3, F6 in publications
4. **Custom Analysis**: Use JSONL logs for detailed investigations

## 🤝 Support

For issues or questions:
1. Check this guide and troubleshooting section
2. Review individual script help messages
3. Examine log outputs for specific error messages
4. Test with minimal configuration first
