# FieldCert-Weather

A distribution-free, physics-aware calibration methodology for uncertainty quantification in weather forecast fields.

## Project Overview

FieldCert-Weather implements a robust framework for calibrating weather forecast fields with a focus on extreme event maps and field-level maxima. The methodology provides risk-controlled set-valued forecasts and functional intervals using WeatherBench-2 datasets and advanced forecast models.

## Turn-Key RunPod Pipeline

This repository includes a **complete, production-ready pipeline** for running FieldCert and baseline methods on WeatherBench-2 data in RunPod environments.

### Quick Start (RunPod)

1. **Clone and setup**:
   ```bash
   cd /workspace
   git clone https://github.com/ansschh/fieldcert.git
   cd fieldcert
   make setup  # Creates venv and installs dependencies
   ```

2. **Run complete pipeline**:
   ```bash
   make all  # Prepares data subset and runs all methods
   make results  # Display results summary
   ```

3. **Optional: Add EMOS probabilistic baseline**:
   ```bash
   make emos
   ```

### Pipeline Components

#### Scripts (`scripts/`)
- **`fc_prepare_subset.py`**: Streams WeatherBench-2 data and creates aligned forecast/truth NPZ subsets
- **`fc_run_crc_baselines.py`**: Runs FieldCert CRC and baseline methods (global bump, morphological CP, pixelwise CP)
- **`fc_run_emos.py`**: Trains and evaluates EMOS probabilistic baseline from IFS ensemble

#### Makefile Targets
- `make setup`: Create virtual environment and install dependencies
- `make subset`: Prepare WeatherBench-2 data subset
- `make baselines`: Run FieldCert CRC and baseline methods
- `make emos`: Run EMOS probabilistic baseline (optional)
- `make all`: Complete pipeline (subset + baselines)
- `make results`: Display results summary
- `make sweep-leads`: Run experiments across multiple lead times
- `make sweep-thresholds`: Run experiments across multiple thresholds

### Configuration

Customize experiments via environment variables:

```bash
# Example: 48-hour lead time, 25 m/s threshold
make all LEAD_HOURS=48 THRESHOLD=25.0 VARIABLE=10m_wind_speed

# Multi-year experiment
make all YEARS_SUBSET=2019-2020

# EMOS with more ensemble members
make emos MEMBERS=25 YEARS_CAL=2018-2019 YEARS_TEST=2020
```

### WeatherBench-2 Data Access

The pipeline **streams data directly** from Google Cloud Storage (no downloads required):
- **ERA5 (truth)**: `gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr`
- **IFS ENS mean**: `gs://weatherbench2/datasets/ens/2018-2022-240x121_equiangular_with_poles_conservative_mean.zarr`
- **IFS ENS full**: `gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr`

### Example Results

After running `make all && make results`:

```
Set-valued methods (wb2_10m_wind_speed_24h_2020):
  Threshold: 20.0
  Target FPA (alpha): 0.1

  fieldcert_crc  : FPA=0.0987, FNA=0.1234, IoU=0.7543
  global_bump    : FPA=0.1001, FNA=0.1456, IoU=0.7321
  morph_cp       : FPA=0.0995, FNA=0.1389, IoU=0.7398
  pixel_cp       : FPA=0.0992, FNA=0.1278, IoU=0.7489
```

## Repository Structure

```
fieldcert/
├── src/
│   ├── calibration/     # Core FieldCert methodology
│   ├── baselines/       # Baseline methods
│   └── eval/           # Evaluation utilities
├── scripts/            # RunPod pipeline scripts
├── Makefile           # Automated pipeline
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## Core Features

- **Distribution-free calibration** with finite-sample guarantees
- **Physics-aware margins** using gradient magnitude and divergence
- **Regime stratification** for handling covariate shift
- **Set-valued forecasts** with risk control
- **Comprehensive evaluation** metrics for both set-valued and probabilistic forecasts
- **Memory-efficient streaming** from WeatherBench-2
- **Production-ready code** with type annotations and error handling

## Advanced Usage

### Custom Experiments

```bash
# Multi-lead experiment
for lead in 24 48 72; do
  make all LEAD_HOURS=$lead YEARS_SUBSET=2020
done

# Variable comparison
for var in 10m_wind_speed 2m_temperature; do
  make all VARIABLE=$var THRESHOLD=20.0
done
```

### Data Sizes

- **Single subset NPZ** (1 variable, 1 lead, 1 year): ~100-500 MB
- **Memory usage** during processing: ~2-4 GB
- **No full dataset downloads** required (streaming only)

### Troubleshooting

- **Cold start latency**: First GCS access may be slow; subsequent reads are faster
- **Memory issues**: Reduce `YEARS_SUBSET` range or use fewer `MEMBERS` for EMOS
- **Missing variables**: Check available variables with `xarray.open_zarr(path).data_vars`

## Development

### Local Installation

```bash
git clone https://github.com/ansschh/fieldcert.git
cd fieldcert
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Testing

```bash
make test-setup  # Verify all dependencies
make list-files  # Show generated files
```

## Citation

If you use FieldCert-Weather in your research, please cite:

```bibtex
@software{fieldcert_weather,
  title={FieldCert-Weather: Distribution-free, Physics-aware Calibration for Weather Forecasts},
  author={FieldCert Contributors},
  year={2025},
  url={https://github.com/ansschh/fieldcert}
}
```

## License

[MIT License](LICENSE)
