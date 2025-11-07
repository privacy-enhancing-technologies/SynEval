# Differential Privacy Dashboard Generator - Configuration Guide

This guide explains how to configure your YAML files and organize your experiment directories to generate comprehensive differential privacy (DP) evaluation dashboards.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Directory Structure](#directory-structure)
4. [YAML Configuration](#yaml-configuration)
5. [Running Experiments](#running-experiments)
6. [Output Structure](#output-structure)
7. [Troubleshooting](#troubleshooting)

## Overview

The DP Dashboard Generator evaluates synthetic data and generates interactive HTML dashboards. It supports:

- **Single Dataset Evaluation**: Evaluate one synthetic dataset (no epsilon/DP required)
- **DP Evaluation**: Compare multiple privacy budgets (epsilon values)
- **Multiple Experiment Groups**: Compare different generation approaches

Three types of dashboards are generated:

- **Individual Dashboards**: Detailed metrics for a single experiment
- **In-Group Comparisons**: Compare different epsilon values within the same experiment group
- **Cross-Group Comparisons**: Compare multiple experiment groups

## Quick Start

### Scenario 1: Single Synthetic Dataset (No DP)

**You have:**
- One synthetic CSV file
- One real training CSV file  
- One real test CSV file

**Minimal directory structure:**
```
project_root/
├── configs/
│   └── simple_eval.yaml
├── data/
│   ├── synthetic.csv
│   ├── real_train.csv
│   └── real_test.csv
└── experiments/
    └── run_experiment.py
```

**Minimal configuration (`configs/simple_eval.yaml`):**
```yaml
version: 1

datasets:
  my_data:
    display_name: "My Dataset"
    default_columns:
      - feature1
      - feature2
      - target
    utility:
      target_column: target
      input_columns:
        - feature1
        - feature2

experiment_groups:
  - key: SINGLE_EXPERIMENT
    display_name: "My Synthetic Data Evaluation"
    dataset: my_data
    experiments:
      - epsilon: "none"  # No DP, just evaluation
        synthetic_path: data/synthetic.csv
        train_path: data/real_train.csv
        test_path: data/real_test.csv

evaluations:
  fidelity:
    enabled: true
  utility:
    enabled: true
  privacy:
    enabled: true
  diversity:
    enabled: true
  mia:
    enabled: false  # Disable if you don't have model checkpoints

privacy:
  skip_anonymeter: true  # Set to false if you want anonymeter evaluation

reporting:
  individual_reports: true
  in_group_comparisons: false  # Only one experiment, no comparison needed
  cross_group_comparisons: false
  reports_output_dir: reports/evaluation

execution:
  device: auto
  log_level: INFO

validation:
  check_files_exist: true
```

**Run it:**
```bash
python experiments/run_experiment.py \
    --config configs/simple_eval.yaml \
    --group SINGLE_EXPERIMENT \
    --epsilon none
```

**Result:** Individual dashboard at `reports/evaluation/SINGLE_EXPERIMENT/eps_none/individual_SINGLE_EXPERIMENT.html`

---

### Scenario 2: Single Epsilon Value from DP Experiment

**You have:**
- Synthetic data from one DP run (e.g., epsilon=10)
- Want to evaluate just this one run

**Directory structure:**
```
project_root/
├── configs/
│   └── single_epsilon.yaml
├── data/
│   └── epsilon_10/
│       ├── synthetic.csv
│       ├── train.csv
│       └── test.csv
```

**Configuration (`configs/single_epsilon.yaml`):**
```yaml
version: 1

datasets:
  my_data:
    display_name: "My Dataset"
    default_columns:
      - feature1
      - feature2
    utility:
      target_column: feature2
      input_columns:
        - feature1

experiment_groups:
  - key: DP_EPSILON_10
    display_name: "DP Synthetic Data (ε=10)"
    dataset: my_data
    experiments:
      - epsilon: "10"
        synthetic_path: data/epsilon_10/synthetic.csv
        train_path: data/epsilon_10/train.csv
        test_path: data/epsilon_10/test.csv

evaluations:
  fidelity:
    enabled: true
  utility:
    enabled: true
  privacy:
    enabled: true
  diversity:
    enabled: true
  mia:
    enabled: false

reporting:
  individual_reports: true
  in_group_comparisons: false
  cross_group_comparisons: false
  reports_output_dir: reports/evaluation

execution:
  device: auto
  log_level: INFO

validation:
  check_files_exist: true
```

**Run it:**
```bash
python experiments/run_experiment.py \
    --config configs/single_epsilon.yaml \
    --group DP_EPSILON_10 \
    --epsilon 10
```

---

### Scenario 3: Multiple Non-DP Experiments (Comparing Different Methods)

**You have:**
- Multiple synthetic datasets from different methods (e.g., GAN, VAE, Diffusion)
- No epsilon values
- Want to compare methods

**Directory structure:**
```
project_root/
├── configs/
│   └── compare_methods.yaml
└── data/
    ├── gan/
    │   ├── synthetic.csv
    │   ├── train.csv
    │   └── test.csv
    ├── vae/
    │   └── ...
    └── diffusion/
        └── ...
```

**Configuration (`configs/compare_methods.yaml`):**
```yaml
version: 1

datasets:
  my_data:
    display_name: "My Dataset"
    default_columns:
      - col1
      - col2
    utility:
      target_column: col2
      input_columns:
        - col1

experiment_groups:
  - key: GAN_METHOD
    display_name: "GAN-Generated Data"
    dataset: my_data
    experiments:
      - epsilon: "none"
        synthetic_path: data/gan/synthetic.csv
        train_path: data/gan/train.csv
        test_path: data/gan/test.csv

  - key: VAE_METHOD
    display_name: "VAE-Generated Data"
    dataset: my_data
    experiments:
      - epsilon: "none"
        synthetic_path: data/vae/synthetic.csv
        train_path: data/vae/train.csv
        test_path: data/vae/test.csv

  - key: DIFFUSION_METHOD
    display_name: "Diffusion-Generated Data"
    dataset: my_data
    experiments:
      - epsilon: "none"
        synthetic_path: data/diffusion/synthetic.csv
        train_path: data/diffusion/train.csv
        test_path: data/diffusion/test.csv

evaluations:
  fidelity:
    enabled: true
  utility:
    enabled: true
  privacy:
    enabled: true
  diversity:
    enabled: true
  mia:
    enabled: false

reporting:
  individual_reports: true
  in_group_comparisons: false  # No epsilon sweep per group
  cross_group_comparisons: true  # Compare methods
  reports_output_dir: reports/method_comparison

execution:
  device: auto
  log_level: INFO
```

**Run all methods:**
```bash
python experiments/run_dp_evaluation.py --config configs/compare_methods.yaml
```

**Or run one method:**
```bash
python experiments/run_experiment.py \
    --config configs/compare_methods.yaml \
    --group GAN_METHOD \
    --epsilon none
```

**Result:** Cross-group comparison dashboard comparing all three methods!

---

## Directory Structure

### For DP Experiments with Multiple Epsilons

```
project_root/
├── configs/
│   └── cross_group_dp.yaml          # Your configuration file
├── experiments/
│   ├── data/
│   │   ├── OUTPUT_GRID_ROW_DP_EXTREMA/
│   │   │   ├── dp_eps1/
│   │   │   │   └── stocks/
│   │   │   │       ├── ddpm_fake_stocks.csv           # Synthetic data
│   │   │   │       ├── samples/
│   │   │   │       │   ├── stocks_ground_truth_24_train.csv
│   │   │   │       │   └── stocks_ground_truth_24_test.npy
│   │   │   │       └── checkpoints/                   # Optional: for MIA
│   │   │   │           └── checkpoint-10.pt
│   │   │   ├── dp_eps10/
│   │   │   │   └── stocks/
│   │   │   │       └── ...
│   │   │   └── dp_eps100/
│   │   │       └── stocks/
│   │   │           └── ...
│   │   └── OUTPUT_GRID_WINDOW_DP_EXTREMA/
│   │       └── ...
│   ├── run_dp_evaluation.py
│   └── run_experiment.py
├── reports/
│   └── dp_evaluation/                # Generated dashboards appear here
└── run.py                            # SynEval main evaluator
```

### Key File Requirements

For each experiment (epsilon value), you need:

1. **Synthetic Data** (Required): CSV file containing synthetic data
   - Example: `ddpm_fake_stocks.csv`

2. **Training Data** (Required): CSV or NPY file
   - Example: `stocks_ground_truth_24_train.csv`

3. **Test Data** (Required): CSV or NPY file
   - Example: `stocks_ground_truth_24_test.npy`

4. **Model Checkpoint** (Optional, for MIA): PyTorch model file
   - Example: `checkpoint-10.pt`
   - Only needed if you want membership inference attack evaluation

---

## YAML Configuration

### Complete Configuration Template (Full DP Evaluation)

```yaml
version: 1

# ============================================================================
# DATASETS: Define your datasets and their metadata
# ============================================================================
datasets:
  stocks:                              # Dataset key (reference this in experiment_groups)
    display_name: "Stocks Time-Series" # Human-readable name
    default_columns:                   # Column names (for NPY files without headers)
      - Open
      - High
      - Low
      - Close
      - Adj_Close
      - Volume
    utility:                           # Utility evaluation configuration
      target_column: Close             # Column to predict
      input_columns:                   # Features for prediction
        - Open
        - Volume
        - Low
        - High
      selected_metrics:                # Specific utility metrics to run
        - tstr_accuracy
        - correlation_analysis

# ============================================================================
# EXPERIMENT GROUPS: Define your experiments
# ============================================================================
experiment_groups:
  - key: OUTPUT_GRID_ROW_DP_EXTREMA    # Unique identifier for this group
    display_name: "Row DP + Extrema Protection, 10k"  # Display name for dashboards
    dataset: stocks                    # Reference to dataset key above
    
    generator:                         # Pattern-based path generation
      type: pattern
      base_dir: experiments/data/OUTPUT_GRID_ROW_DP_EXTREMA
      epsilon_prefix: dp_eps           # Prefix for epsilon directories
      epsilons: ["1", "10", "100", "1000", "10000", "Inf"]  # Epsilon values to evaluate
      
      data:
        subdir: stocks                 # Subdirectory within each epsilon folder
        synthetic: ddpm_fake_stocks.csv          # Synthetic data filename
        train: samples/stocks_ground_truth_24_train.csv  # Train data path
        test: samples/stocks_ground_truth_24_test.npy    # Test data path
      
      model:                           # Optional: for MIA evaluation
        subdir: stocks/checkpoints
        file: checkpoint-10.pt

  - key: OUTPUT_GRID_WINDOW_DP_EXTREMA
    display_name: "Window DP + Extrema Protection, 10k"
    dataset: stocks
    # ... (similar structure)

# ============================================================================
# EVALUATIONS: Enable/disable evaluation dimensions
# ============================================================================
evaluations:
  fidelity:
    enabled: true
  utility:
    enabled: true
  privacy:
    enabled: true
  diversity:
    enabled: true
  mia:                                 # Membership Inference Attack
    enabled: true

# ============================================================================
# PRIVACY SETTINGS
# ============================================================================
privacy:
  skip_anonymeter: true                # Set to false to enable Anonymeter evaluations
  max_rows: null                       # Limit dataset size for faster evaluation

# ============================================================================
# MIA SETTINGS
# ============================================================================
mia:
  adapter: diffusion_ts                # MIA adapter type
  dataset: stocks                      # Dataset name for MIA
  n_shadow: 1000                       # Number of shadow samples

# ============================================================================
# REPORTING SETTINGS
# ============================================================================
reporting:
  individual_reports: true             # Generate individual experiment dashboards
  in_group_comparisons: true           # Generate in-group comparison dashboards
  cross_group_comparisons: true        # Generate cross-group comparison dashboard
  reports_output_dir: reports/dp_evaluation  # Output directory for dashboards
  formats:
    - html
    - json

# ============================================================================
# EXECUTION SETTINGS
# ============================================================================
execution:
  parallel_workers: 4                  # Number of parallel workers
  device: auto                         # 'auto', 'cpu', or 'cuda'
  enable_cache: true                   # Enable caching for faster re-runs
  cache_dir: ./cache
  log_level: INFO                      # DEBUG, INFO, WARNING, ERROR
  log_file: dp_evaluation.log
  continue_on_error: true              # Continue if one experiment fails
  save_intermediate: true

# ============================================================================
# VALIDATION SETTINGS
# ============================================================================
validation:
  check_files_exist: true              # Validate file existence before running
```

### Configuration Sections Explained

#### 1. Datasets Section

Define metadata for each dataset:

```yaml
datasets:
  YOUR_DATASET_NAME:
    display_name: "Human-Readable Name"
    default_columns:                   # Column names for NPY files
      - column1
      - column2
    utility:
      target_column: column_to_predict
      input_columns:
        - feature1
        - feature2
      selected_metrics:
        - tstr_accuracy                # Train on Synthetic, Test on Real
        - correlation_analysis
```

#### 2. Experiment Groups Section

Three ways to define experiments:

**Option A: Single Experiment (No DP / One Epsilon)**

```yaml
experiment_groups:
  - key: MY_EXPERIMENT
    display_name: "My Synthetic Data Evaluation"
    dataset: YOUR_DATASET_NAME
    experiments:
      - epsilon: "none"  # Or "10" for a single DP value
        synthetic_path: path/to/synthetic.csv
        train_path: path/to/train.csv
        test_path: path/to/test.csv
```

**Option B: Pattern-Based (Multiple Epsilons)**

```yaml
experiment_groups:
  - key: MY_EXPERIMENT
    display_name: "My Experiment Description"
    dataset: YOUR_DATASET_NAME
    generator:
      type: pattern
      base_dir: experiments/data/MY_EXPERIMENT
      epsilon_prefix: dp_eps           # Creates: dp_eps1, dp_eps10, etc.
      epsilons: ["1", "10", "100"]
      data:
        subdir: data_subfolder         # Optional subdirectory
        synthetic: synthetic.csv
        train: train.csv
        test: test.npy
      model:                           # Optional: for MIA
        subdir: models
        file: model.pt
```

This creates paths like:
- `experiments/data/MY_EXPERIMENT/dp_eps1/data_subfolder/synthetic.csv`
- `experiments/data/MY_EXPERIMENT/dp_eps10/data_subfolder/synthetic.csv`

**Option C: Explicit Multiple Experiments**

```yaml
experiment_groups:
  - key: MY_EXPERIMENT
    display_name: "My Experiment Description"
    dataset: YOUR_DATASET_NAME
    experiments:                       # List individual experiments
      - epsilon: "1"
        delta: "1e-5"                  # Optional
        synthetic_path: path/to/eps1/synthetic.csv
        train_path: path/to/train.csv
        test_path: path/to/test.csv
      - epsilon: "10"
        synthetic_path: path/to/eps10/synthetic.csv
        train_path: path/to/train.csv
        test_path: path/to/test.csv
      - epsilon: "100"
        synthetic_path: path/to/eps100/synthetic.csv
        train_path: path/to/train.csv
        test_path: path/to/test.csv
```

---

## Running Experiments

### Run Single Experiment

**For non-DP or single epsilon:**
```bash
python experiments/run_experiment.py \
    --config configs/your_config.yaml \
    --group YOUR_GROUP_KEY \
    --epsilon none  # Or specific epsilon value like "10"
```

**With additional options:**
```bash
# Skip MIA evaluation
python experiments/run_experiment.py \
    --config configs/your_config.yaml \
    --group MY_GROUP \
    --epsilon 10 \
    --disable-mia

# With delta specification (if multiple experiments share same epsilon)
python experiments/run_experiment.py \
    --config configs/your_config.yaml \
    --group MY_GROUP \
    --epsilon 10 \
    --delta 1e-5

# Debug mode
python experiments/run_experiment.py \
    --config configs/your_config.yaml \
    --group MY_GROUP \
    --epsilon 10 \
    --log-level DEBUG
```

### Run All Experiments in Config

Generate all dashboards at once:

```bash
python experiments/run_dp_evaluation.py --config configs/cross_group_dp.yaml
```

**Options:**
```bash
# Run in parallel
python experiments/run_dp_evaluation.py \
    --config configs/cross_group_dp.yaml \
    --parallel \
    --workers 8

# Set log level
python experiments/run_dp_evaluation.py \
    --config configs/cross_group_dp.yaml \
    --log-level DEBUG
```

---

## Output Structure

### Single Experiment Output

After running a single experiment:

```
reports/evaluation/
├── MY_GROUP/
│   └── eps_10/  # Or eps_none for non-DP
│       ├── results.json
│       ├── metadata.json
│       └── individual_MY_GROUP.html  # ← Your dashboard
└── summary.json
```

### Multiple Experiments Output

After running multiple experiments:

```
reports/dp_evaluation/
├── OUTPUT_GRID_ROW_DP_EXTREMA/
│   ├── eps_1/
│   │   ├── results.json              # Raw results
│   │   ├── metadata.json             # Dataset metadata
│   │   └── individual_OUTPUT_GRID_ROW_DP_EXTREMA.html  # Individual dashboard
│   ├── eps_10/
│   │   └── ...
│   └── in_group_comparison/
│       └── in_group_output_grid_row_dp_extrema.html    # In-group comparison
├── OUTPUT_GRID_WINDOW_DP_EXTREMA/
│   └── ...
├── cross_group_comparison/
│   └── cross_group_comparison_TIMESTAMP.html           # Cross-group comparison
└── summary.json                      # Overall summary
```

### Dashboard Types

1. **Individual Dashboard** (`individual_*.html`)
   - Detailed metrics for one epsilon value or non-DP dataset
   - Fidelity, utility, privacy, diversity scores
   - MIA results (if enabled)
   - Visualizations and comparisons

2. **In-Group Comparison** (`in_group_*.html`)
   - Compare different epsilon values within same experiment
   - Privacy-utility trade-off curves
   - Metrics across privacy budgets
   - Only generated when multiple epsilons exist in a group

3. **Cross-Group Comparison** (`cross_group_comparison_*.html`)
   - Compare different experimental approaches
   - Side-by-side metric comparisons
   - Best practices identification
   - Generated when multiple groups exist

---

## Troubleshooting

### Common Issues

#### 1. File Not Found Errors

**Problem:** `FileNotFoundError: Synthetic data not found`

**Solution:** Verify your paths match the YAML configuration:
```bash
# Check if file exists
ls experiments/data/OUTPUT_GRID_ROW_DP_EXTREMA/dp_eps10/stocks/ddpm_fake_stocks.csv

# Verify YAML paths match actual directory structure
```

#### 2. Column Mismatch Errors

**Problem:** `ValueError: synthetic data missing columns: ['Close']`

**Solution:** Ensure `default_columns` in your dataset configuration match your data:
```yaml
datasets:
  stocks:
    default_columns:  # Must match actual columns in your data
      - Open
      - High
      - Low
      - Close
```

#### 3. MIA Evaluation Failures

**Problem:** `MIA evaluation failed: Model checkpoint unavailable`

**Solution:**
- If you don't have model checkpoints, disable MIA:
  ```yaml
  evaluations:
    mia:
      enabled: false
  ```
- Or run single experiments with `--disable-mia` flag

#### 4. Memory Issues

**Problem:** `MemoryError` or system freezes

**Solution:**
```yaml
privacy:
  max_rows: 10000  # Limit dataset size

execution:
  parallel_workers: 2  # Reduce parallelism
```

#### 5. GPU/CUDA Errors

**Problem:** `CUDA out of memory`

**Solution:**
```yaml
execution:
  device: cpu  # Force CPU usage
```

#### 6. "No experiments found" Error

**Problem:** `No experiment found for group=MY_GROUP epsilon=10`

**Solution:**
- Check that your group key matches exactly (case-sensitive)
- Verify epsilon value format (string, not number)
- Check YAML syntax is valid:
  ```bash
  python -c "import yaml; yaml.safe_load(open('configs/your_config.yaml'))"
  ```

### Validation Tips

Before running experiments:

1. **Validate file paths:**
   ```python
   # Quick validation script
   import yaml
   from pathlib import Path
   
   config = yaml.safe_load(open('configs/your_config.yaml'))
   
   # For explicit experiments
   for group in config['experiment_groups']:
       if 'experiments' in group:
           for exp in group['experiments']:
               print(f"Checking {exp['synthetic_path']}: {Path(exp['synthetic_path']).exists()}")
   
   # For pattern-based experiments
   for group in config['experiment_groups']:
       if 'generator' in group:
           base_dir = Path(group['generator']['base_dir'])
           for eps in group['generator']['epsilons']:
               eps_dir = base_dir / f"{group['generator']['epsilon_prefix']}{eps}"
               print(f"Checking {eps_dir}: {eps_dir.exists()}")
   ```

2. **Test with single experiment:**
   ```bash
   # Test one experiment first
   python experiments/run_experiment.py \
       --config configs/your_config.yaml \
       --group YOUR_GROUP \
       --epsilon 10 \
       --log-level DEBUG
   ```

3. **Check log output:**
   ```bash
   # Review log file
   tail -f dp_evaluation.log
   ```

### Getting Help

If you encounter issues:

1. Enable debug logging: `--log-level DEBUG`
2. Check `dp_evaluation.log` for detailed error messages
3. Verify all file paths exist and are accessible
4. Ensure dataset configurations match your actual data structure
5. Test with a small subset of epsilons first

---

## Advanced Configuration

### Custom Metrics Selection

Select specific metrics to speed up evaluation:

```yaml
datasets:
  stocks:
    utility:
      selected_metrics:
        - tstr_accuracy  # Only run TSTR, skip correlation

evaluations:
  fidelity:
    enabled: true
    selected_metrics:
      - diagnostic
      - quality
      # Skip: text, numerical_statistics
```

### Metadata Configuration

For advanced dataset configurations:

```yaml
datasets:
  custom_dataset:
    metadata:
      info_json_path: path/to/metadata.json
      column_overrides:
        column_name:
          sdtype: numerical  # or categorical, text, datetime
          categories: [cat1, cat2]  # for categorical
```

### MIA Adapter Configuration

For custom MIA evaluation:

```yaml
mia:
  adapter: diffusion_ts  # or custom adapter name
  dataset: stocks
  n_shadow: 1000
  
experiment_groups:
  - key: MY_GROUP
    mia:  # Group-specific override
      adapter: custom_adapter
      dataset: custom_dataset
```

---

## Summary: Common Use Cases

### ✅ Single synthetic dataset, no DP
```yaml
experiment_groups:
  - key: MY_DATA
    dataset: my_dataset
    experiments:
      - epsilon: "none"
        synthetic_path: data/synthetic.csv
        train_path: data/train.csv
        test_path: data/test.csv
```
```bash
python experiments/run_experiment.py --config config.yaml --group MY_DATA --epsilon none
```

### ✅ One DP experiment (epsilon=10 only)
```yaml
experiment_groups:
  - key: DP_EPS10
    dataset: my_dataset
    experiments:
      - epsilon: "10"
        synthetic_path: data/eps10/synthetic.csv
        train_path: data/train.csv
        test_path: data/test.csv
```
```bash
python experiments/run_experiment.py --config config.yaml --group DP_EPS10 --epsilon 10
```

### ✅ Multiple epsilons, same method
```yaml
experiment_groups:
  - key: DP_SWEEP
    dataset: my_dataset
    generator:
      base_dir: data/dp_sweep
      epsilon_prefix: eps_
      epsilons: ["1", "10", "100"]
      data:
        synthetic: synthetic.csv
        train: train.csv
        test: test.csv
```
```bash
# Run all
python experiments/run_dp_evaluation.py --config config.yaml

# Or run one
python experiments/run_experiment.py --config config.yaml --group DP_SWEEP --epsilon 10
```

### ✅ Compare different methods (no DP)
```yaml
experiment_groups:
  - key: GAN
    experiments:
      - epsilon: "none"
        synthetic_path: data/gan/synthetic.csv
        # ...
  - key: VAE
    experiments:
      - epsilon: "none"
        synthetic_path: data/vae/synthetic.csv
        # ...
```
```bash
python experiments/run_dp_evaluation.py --config config.yaml
# Creates cross-group comparison dashboard
```

---

**Ready to generate your dashboards?**

```bash
# Single experiment
python experiments/run_experiment.py \
    --config configs/your_config.yaml \
    --group YOUR_GROUP \
    --epsilon YOUR_EPSILON

# All experiments
python experiments/run_dp_evaluation.py --config configs/your_config.yaml
```

Your HTML dashboards will appear in `reports/dp_evaluation/`! 🎉
