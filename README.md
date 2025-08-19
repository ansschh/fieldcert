# FieldCert-Weather

A distribution-free, physics-aware calibration methodology for uncertainty quantification in weather forecast fields.

## Project Overview

FieldCert-Weather implements a robust framework for calibrating weather forecast fields with a focus on extreme event maps and field-level maxima. The methodology provides risk-controlled set-valued forecasts and functional intervals using WeatherBench-2 datasets and advanced forecast models.

## Repository Structure

- `src/calibration/`: Core calibration methodology including conformal risk control (CRC), scoring functions, morphology filters, physics-aware margins, and regime stratification utilities.
- `src/baselines/`: Baseline methods for comparison, including raw thresholding, global threshold bump, morphological CP, pixelwise CP, and EMOS.
- `src/eval/`: Evaluation utilities for set-based and probabilistic metrics, including WeatherBench-2 adapters.

## Features

- **Distribution-free calibration** with finite-sample guarantees
- **Physics-aware margins** using gradient magnitude and divergence
- **Regime stratification** for handling covariate shift
- **Set-valued forecasts** with risk control
- **Comprehensive evaluation** metrics for both set-valued and probabilistic forecasts

## Dependencies

- numpy
- scipy
- scikit-image
- scikit-learn
- xarray
- gcsfs (for WeatherBench-2 data access)
- zarr
- joblib (optional, for model persistence)

## Installation

```bash
# Clone the repository
git clone https://github.com/ansschh/fieldcert.git
cd fieldcert

# Install dependencies
pip install -r requirements.txt
```

## License

[MIT License](LICENSE)
