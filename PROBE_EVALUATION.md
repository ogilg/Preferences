# Flexible Probe Evaluation Workflow

This document describes the flexible probe evaluation system for training probes on multiple datasets and evaluating on held-out datasets.

## Overview

The system has three main components:

1. **Probe Training** (`train_probe_experiment.py`) - Train probes on datasets A, B, C
2. **Probe Evaluation** (`run_probe_evaluation.py`) - Evaluate trained probes on dataset D
3. **Orchestration** (`run_train_and_evaluate.py`) - Coordinate training and multiple evaluation runs

## Quick Start

### Step 1: Train Probes on Datasets A, B, C

Create a training config (see `configs/probe_training_example.yaml`):

```bash
python -m src.probes.train_probe_experiment --config configs/probe_training.yaml
```

This will:
- Train probes on all combinations of templates, datasets, seeds, and layers
- Save probe weights to `probe_data/manifests/{manifest_dir}/probes/probe_*.npy`
- Save metadata to `probe_data/manifests/{manifest_dir}/manifest.json`

### Step 2: Evaluate on Dataset D

First, ensure you have measurements (scores) for dataset D. These should be in a directory structure like:

```
results/experiments/{eval_experiment_id}/
├── post_task_stated/
│   └── post_task_stated_001_model_format_seedN/
│       ├── measurements.yaml
│       └── config.yaml
```

Then create an evaluation config (see `configs/probe_eval_example.yaml`):

```bash
python -m src.probes.run_probe_evaluation --config configs/probe_eval.yaml --output results/probe_eval_d.json
```

This will evaluate all specified probes on dataset D and save results.

### Step 3 (Optional): Train and Evaluate End-to-End

Run the full pipeline with orchestration:

```bash
python -m src.probes.run_train_and_evaluate \
  --train-config configs/probe_training.yaml \
  --eval-configs configs/probe_eval_1.yaml configs/probe_eval_2.yaml \
  --output-dir results/cross_dataset_eval/
```

This will:
1. Train all probes according to training config
2. Evaluate on all evaluation datasets
3. Print a summary table showing generalization across datasets

## Configuration Details

### ProbeTrainingConfig

Training configuration for probe training:

```yaml
experiment_name: my_experiment           # Name for this experiment
experiment_dir: results/experiments/...  # Path to measurement results
activations_path: probe_data/activations/activations.npz

# Define probe combinations: templates x datasets x seeds x layers
template_combinations:
  - [post_task_qualitative_013]
  - [post_task_stated_013]

dataset_combinations:
  - [wildchat]
  - [alpaca]
  - null  # All datasets

seed_combinations:
  - [0, 1]
  - [2]

layers: [16, 20, 24]

# Training parameters
cv_folds: 5
alpha_sweep_size: 5
manifest_dir: probe_data/manifests/my_experiment
```

### ProbeEvaluationConfig

Evaluation configuration:

```yaml
probe:
  manifest_dir: probe_data/manifests/my_experiment
  probe_ids: ["0001", "0002", "0003"]  # Which probes to evaluate

data:
  experiment_dir: results/experiments/eval_dataset/  # Where eval measurements live
  template: post_task_stated_013                     # Which template to evaluate on
  seeds: [0]                                         # Which seeds
  dataset_filter: null                               # Filter to specific dataset (optional)
  activations_path: probe_data/activations/activations.npz

output:
  results_file: results/probe_eval_results.json
```

## Core API

### `evaluate_probe_on_data()`

Core evaluation function:

```python
from src.probes.evaluate import evaluate_probe_on_data
import numpy as np

result = evaluate_probe_on_data(
    probe_weights=np.ndarray,        # [coef_1, ..., coef_n, intercept]
    activations=np.ndarray,          # (n_samples, n_features)
    scores=np.ndarray,               # (n_samples,)
    task_ids_data=np.ndarray,        # task IDs for activations
    task_ids_scores=list[str],       # task IDs for scores
)

# Returns:
# {
#     "r2": float,              # R² score
#     "mse": float,             # Mean squared error
#     "pearson_r": float,       # Pearson correlation
#     "n_samples": int,         # Number of samples used
#     "predictions": list,      # Model predictions
# }
```

### `run_evaluation()`

High-level evaluation runner:

```python
from src.probes.config import ProbeEvaluationConfig
from src.probes.run_probe_evaluation import run_evaluation

config = ProbeEvaluationConfig.from_yaml("configs/probe_eval.yaml")
results = run_evaluation(config)
```

### `run_train_and_evaluate()`

Orchestration function:

```python
from src.probes.config import ProbeTrainingConfig, ProbeEvaluationConfig
from src.probes.run_train_and_evaluate import run_train_and_evaluate

train_config = ProbeTrainingConfig.from_yaml("configs/probe_training.yaml")
eval_configs = [
    ProbeEvaluationConfig.from_yaml("configs/probe_eval_1.yaml"),
    ProbeEvaluationConfig.from_yaml("configs/probe_eval_2.yaml"),
]

summary = run_train_and_evaluate(train_config, eval_configs, output_dir="results/")
```

## Output Formats

### Training Output

Manifest saved to `manifest_dir/manifest.json`:

```json
{
  "experiment_name": "...",
  "experiment_dir": "...",
  "created_at": "...",
  "probes": [
    {
      "id": "0001",
      "file": "probes/probe_0001.npy",
      "templates": ["post_task_qualitative_013"],
      "layer": 16,
      "datasets": ["wildchat"],
      "seeds": [0, 1],
      "cv_r2_mean": 0.234,
      "cv_r2_std": 0.045,
      "n_measurement_instances": 450,
      "n_unique_tasks": 234,
      "best_alpha": 0.1,
      "train_test_gap": 0.034,
      "cv_stability": 0.92,
      "trained_at": "..."
    }
  ]
}
```

### Evaluation Output

Results saved to `results_file`:

```json
{
  "config": {
    "manifest_dir": "...",
    "probe_ids": ["0001", "0002"],
    "template": "post_task_stated_013",
    "seeds": [0],
    "dataset_filter": null
  },
  "created_at": "...",
  "probes": [
    {
      "id": "0001",
      "layer": 16,
      "trained_on_templates": ["post_task_qualitative_013"],
      "trained_on_datasets": ["wildchat"],
      "eval_metrics": {
        "r2": 0.156,
        "mse": 0.234,
        "pearson_r": 0.421,
        "n_samples": 234,
        "predictions": [0.1, 0.2, ...]
      }
    }
  ]
}
```

## Design Rationale

**Why separate training and evaluation?**
- Training is expensive (cross-validation, hyperparameter search)
- Evaluation is cheap (single forward pass)
- Decoupling allows flexible evaluation on any dataset with pre-computed measurements

**Why require pre-computed measurements?**
- Measurements involve API calls (expensive)
- Separates concerns: measurement pipeline vs. analysis pipeline
- User has full control over measurement parameters

**Why use configs?**
- Reproducibility: exact hyperparameters and data are recorded
- Flexibility: easy to run many combinations
- Traceability: results can be traced back to config

## Advanced Usage

### Held-One-Out Cross-Dataset Validation

Run leave-one-dataset-out validation: train on N-1 datasets, evaluate on the held-out dataset. Automatically generates all train/eval splits without creating individual config files.

```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 \
  --output-dir results/held_one_out/
```

This will:
1. Detect available datasets (ALPACA, BAILBENCH, MATH, WILDCHAT)
2. For each dataset:
   - Train probes on N-1 other datasets
   - Evaluate on the held-out dataset
3. Print summary table showing generalization per dataset
4. Save results to `results/held_one_out/hoo_summary.json`

Example output:
```
SUMMARY
================================================================================

Eval Dataset    # Probes     Median R²       Mean Pearson r
------------------------------------------------------------
alpaca          12           0.1234          0.3421
bailbench       12           0.0987          0.2891
math            12           0.1456          0.3812
wildchat        12           0.1123          0.3156
```

You can also specify specific datasets and skip training if rerunning:

```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 1 2 \
  --datasets alpaca math \
  --output-dir results/hoo_subset/ \
  --skip-training  # Use already-trained probes
```

**Base config format** (`configs/probe_training/hoo_base.yaml`):
- Use `DATASET` placeholder in paths (experiment_dir, manifest_dir)
- This gets substituted for each fold automatically
- No need to manually create N separate configs!

### Evaluate Cross-Dataset Generalization (Manual)

Train on datasets A, B and evaluate on C, D:

```bash
# 1. Train
python -m src.probes.train_probe_experiment --config train_ab.yaml

# 2. Measure dataset C and D
python -m src.running_measurements.run_measurements ...

# 3. Evaluate
python -m src.probes.run_train_and_evaluate \
  --train-config train_ab.yaml \
  --eval-configs eval_c.yaml eval_d.yaml
```

### Evaluate Specific Probe Subset

In evaluation config, specify only the probe IDs you want:

```yaml
probe:
  probe_ids: ["0001", "0003", "0005"]  # Only evaluate these
```

### Filter by Dataset

Evaluate probes only on specific tasks:

```yaml
data:
  dataset_filter: wildchat  # Only evaluate on WildChat tasks
```

### Use Different Activations

Point to alternative activation source:

```yaml
data:
  activations_path: probe_data/activations/activations_v2.npz
```

## Troubleshooting

**"Probe not found in manifest"**
- Check probe_ids in config match those in manifest.json
- Run training first to generate probes

**"Could not find measurement run"**
- Ensure measurements exist in `experiment_dir/post_task_stated/`
- Check directory names match template and seed patterns

**"No samples after dataset filter"**
- Check dataset_filter matches available datasets (WILDCHAT, ALPACA, MATH, etc.)
- Some probes may be trained on all datasets (None), use empty filter

**"Insufficient samples for evaluation"**
- Minimum 10 samples required
- Check that tasks overlap between probe training data and eval data
