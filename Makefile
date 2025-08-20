# Makefile for FieldCert-Weather RunPod Pipeline
# Usage: make <target>

# Default configuration
VARIABLE ?= 10m_wind_speed
LEAD_HOURS ?= 24
THRESHOLD ?= 20.0
ALPHA ?= 0.10
YEARS_SUBSET ?= 2020
YEARS_CAL ?= 2019
YEARS_TEST ?= 2020
MEMBERS ?= 10

# Paths
DATA_DIR ?= /workspace/data
RESULTS_DIR ?= /workspace/results
SUBSETS_DIR ?= $(DATA_DIR)/subsets
SCRIPTS_DIR ?= scripts

# Python environment
PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON ?= $(VENV_DIR)/bin/python

# Derived names
SUBSET_NAME = wb2_$(VARIABLE)_$(LEAD_HOURS)h_$(YEARS_SUBSET)
SUBSET_FILE = $(SUBSETS_DIR)/$(SUBSET_NAME).npz
METRICS_FILE = $(RESULTS_DIR)/metrics_$(SUBSET_NAME).json
EMOS_FILE = $(RESULTS_DIR)/emos_$(VARIABLE)_$(LEAD_HOURS)h_$(YEARS_CAL)to$(YEARS_TEST).json

# Comprehensive evaluation on full dataset with multiple models and UQ methods
COMPREHENSIVE_OUTPUT = results/comprehensive
COMPREHENSIVE_YEARS = 2020 2021

.PHONY: help setup dirs clean all subset baselines emos results comprehensive comprehensive-quick comprehensive-full

help:
	@echo "FieldCert-Weather RunPod Pipeline"
	@echo ""
	@echo "Targets:"
	@echo "  setup      - Create virtual environment and install dependencies"
	@echo "  dirs       - Create necessary directories"
	@echo "  subset     - Prepare WeatherBench-2 data subset"
	@echo "  baselines  - Run FieldCert CRC and baseline methods"
	@echo "  emos       - Run EMOS probabilistic baseline (optional)"
	@echo "  results    - Display results summary"
	@echo "  all        - Run complete pipeline (subset + baselines)"
	@echo "  clean      - Clean generated files"
	@echo ""
	@echo "Configuration (override with VAR=value):"
	@echo "  VARIABLE=$(VARIABLE)"
	@echo "  LEAD_HOURS=$(LEAD_HOURS)"
	@echo "  THRESHOLD=$(THRESHOLD)"
	@echo "  ALPHA=$(ALPHA)"
	@echo "  YEARS_SUBSET=$(YEARS_SUBSET)"
	@echo "  YEARS_CAL=$(YEARS_CAL)"
	@echo "  YEARS_TEST=$(YEARS_TEST)"
	@echo "  MEMBERS=$(MEMBERS)"

setup: $(VENV_DIR)/pyvenv.cfg
	@echo "Setting up Python environment..."
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo "Setup complete!"

$(VENV_DIR)/pyvenv.cfg:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)

dirs:
	@echo "Creating directories..."
	mkdir -p $(SUBSETS_DIR) $(RESULTS_DIR)

subset: $(SUBSET_FILE)

$(SUBSET_FILE): dirs
	@echo "Preparing data subset: $(SUBSET_NAME)"
	$(VENV_PYTHON) $(SCRIPTS_DIR)/fc_prepare_subset.py \
		--variable $(VARIABLE) \
		--lead_hours $(LEAD_HOURS) \
		--years $(YEARS_SUBSET) \
		--out $(SUBSET_FILE)

baselines: $(METRICS_FILE)

$(METRICS_FILE): $(SUBSET_FILE)
	@echo "Running FieldCert CRC and baselines..."
	$(VENV_PYTHON) $(SCRIPTS_DIR)/fc_run_crc_baselines.py \
		--subset $(SUBSET_FILE) \
		--threshold $(THRESHOLD) \
		--alpha $(ALPHA) \
		--out_json $(METRICS_FILE)

emos: $(EMOS_FILE)

$(EMOS_FILE): dirs
	@echo "Running EMOS probabilistic baseline..."
	$(VENV_PYTHON) $(SCRIPTS_DIR)/fc_run_emos.py \
		--variable $(VARIABLE) \
		--lead_hours $(LEAD_HOURS) \
		--years_cal $(YEARS_CAL) \
		--years_test $(YEARS_TEST) \
		--members $(MEMBERS) \
		--threshold $(THRESHOLD) \
		--alpha $(ALPHA) \
		--out_json $(EMOS_FILE)

results:
	@echo "=== FieldCert-Weather Results Summary ==="
	@echo ""
	@echo "Set-valued methods ($(SUBSET_NAME)):"
	@echo "  Threshold: $(THRESHOLD)"
	@echo "  Target FPA (alpha): $(ALPHA)"
	@echo ""
	@if [ -f "$(RESULTS_DIR)/metrics_$(SUBSET_NAME).json" ]; then \
		python -c "import json; data=json.load(open('$(RESULTS_DIR)/metrics_$(SUBSET_NAME).json')); \
		[print(f'  {k:14}: FPA={v[\"fpa\"]:.4f}, FNA={v[\"fna\"]:.4f}, IoU={v[\"iou\"]:.4f}') for k,v in data.items() if isinstance(v, dict) and 'fpa' in v]"; \
	else \
		echo "  No results found. Run 'make all' first."; \
	fi
	@echo ""
	@if [ ! -f "$(RESULTS_DIR)/emos_$(SUBSET_NAME).json" ]; then \
		echo "No EMOS results found. Run 'make emos' to include probabilistic baseline."; \
	fi

all: subset baselines
	@echo "Pipeline complete! Run 'make results' to see summary."

# Multi-experiment targets
sweep-leads:
	@echo "Running lead time sweep..."
	@for lead in 24 48 72; do \
		echo "=== Lead time: $$lead hours ==="; \
		$(MAKE) all LEAD_HOURS=$$lead YEARS_SUBSET=$(YEARS_SUBSET); \
	done
	@echo "Lead time sweep complete!"

sweep-thresholds:
	@echo "Running threshold sweep..."
	@for thresh in 15.0 20.0 25.0; do \
		echo "=== Threshold: $$thresh ==="; \
		$(MAKE) all THRESHOLD=$$thresh YEARS_SUBSET=$(YEARS_SUBSET); \
	done
	@echo "Threshold sweep complete!"

clean:
	@echo "Cleaning generated files..."
	rm -rf $(SUBSETS_DIR)/*.npz
	rm -rf $(RESULTS_DIR)/*.json
	@echo "Clean complete!"

clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)

# Development targets
test-setup:
	@echo "Testing environment setup..."
	$(VENV_PYTHON) -c "import numpy, scipy, sklearn, xarray, gcsfs; print('All dependencies imported successfully!')"

list-files:
	@echo "Generated files:"
	@find $(DATA_DIR) $(RESULTS_DIR) -name "*.npz" -o -name "*.json" 2>/dev/null | sort || echo "No files found"

comprehensive:
	@echo "Running comprehensive FieldCert evaluation..."
	mkdir -p $(COMPREHENSIVE_OUTPUT)
	$(VENV_PYTHON) scripts/fc_comprehensive_eval.py \
		--output $(COMPREHENSIVE_OUTPUT) \
		--variable $(VARIABLE) \
		--threshold $(THRESHOLD) \
		--years $(COMPREHENSIVE_YEARS)
	@echo "Comprehensive evaluation complete!"
	@echo "Results saved to: $(COMPREHENSIVE_OUTPUT)"
	@echo "View summary tables: ls $(COMPREHENSIVE_OUTPUT)/*.csv"
	@echo "View plots: ls $(COMPREHENSIVE_OUTPUT)/plots/*.png"

comprehensive-quick:
	@echo "Running quick comprehensive evaluation (500 samples)..."
	mkdir -p $(COMPREHENSIVE_OUTPUT)
	$(VENV_PYTHON) scripts/fc_comprehensive_eval.py \
		--output $(COMPREHENSIVE_OUTPUT) \
		--variable $(VARIABLE) \
		--threshold $(THRESHOLD) \
		--years 2020 \
		--max-samples 500
	@echo "Quick comprehensive evaluation complete!"

comprehensive-full:
	@echo "Running FULL comprehensive evaluation (all data)..."
	mkdir -p $(COMPREHENSIVE_OUTPUT)
	$(VENV_PYTHON) scripts/fc_comprehensive_eval.py \
		--output $(COMPREHENSIVE_OUTPUT) \
		--variable $(VARIABLE) \
		--threshold $(THRESHOLD) \
		--years 2020 2021 2022
	@echo "FULL comprehensive evaluation complete!"
