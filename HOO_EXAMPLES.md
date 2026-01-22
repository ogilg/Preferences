# Held-One-Out Validation Examples

The `run_held_one_out.py` script enables efficient leave-one-dataset-out validation without creating individual config files. It automatically generates all train/eval splits and orchestrates the entire pipeline.

## Quick Start

### Basic Held-One-Out Validation

Run held-one-out validation with a single template:

```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 \
  --output-dir results/hoo_basic/
```

This will:
1. Auto-detect all datasets (ALPACA, BAILBENCH, MATH, WILDCHAT)
2. For each dataset:
   - Create temporary training config excluding that dataset
   - Train probes on the N-1 other datasets
   - Evaluate on the held-out dataset
3. Print summary table showing cross-dataset generalization
4. Save detailed results to `results/hoo_basic/hoo_summary.json`

### Multi-Template Validation

Train and evaluate separate probes for multiple templates:

```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base_multi_template.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 1 2 \
  --output-dir results/hoo_multi_template/
```

This trains probes for all template combinations in the config, then evaluates all of them. Useful for comparing template robustness across datasets.

## Command-Line Options

```
--base-config PATH              Base config YAML with DATASET placeholder (required)
--template NAME                 Template to evaluate on (required)
--seeds SEED [SEED ...]         Seeds for evaluation (default: [0])
--datasets DATASET [DATASET ...]  Specific datasets to validate (auto-detect if omitted)
--output-dir PATH               Directory for results (optional)
--skip-training                 Skip training, only run evaluation on existing probes
```

## Use Cases

### Case 1: Full Held-One-Out on New Training Run

Train new probes and evaluate on held-out datasets:

```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 \
  --output-dir results/hoo_full_run/
```

Time: ~4-8 hours (depends on dataset sizes and hardware)

### Case 2: Evaluate Already-Trained Probes

Reuse trained probes from a previous run:

```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 \
  --skip-training \
  --output-dir results/hoo_eval_only/
```

Time: ~5 minutes (evaluation is fast)

### Case 3: Subset of Datasets

Validate only on specific datasets:

```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 \
  --datasets alpaca math \
  --output-dir results/hoo_subset/
```

### Case 4: Multiple Seeds

Evaluate with multiple random seeds for robustness:

```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 1 2 3 4 \
  --output-dir results/hoo_robust/
```

### Case 5: Compare Templates

Create a config with multiple templates and evaluate each:

```bash
# First, update a config with multiple templates:
# template_combinations:
#   - [post_task_qualitative_013]
#   - [post_task_stated_013]
#   - [post_task_rating_013]

python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base_multi_template.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 \
  --output-dir results/hoo_templates/
```

## Base Config Format

Create a base config with `DATASET` placeholder:

```yaml
experiment_name: probe_hoo
experiment_dir: results/experiments/probe_hoo_DATASET      # DATASET gets substituted
activations_path: probe_data/activations/activations.npz
manifest_dir: probe_data/manifests/probe_hoo_DATASET       # DATASET gets substituted

template_combinations:
  - [post_task_qualitative_013]

dataset_combinations:
  - []  # Placeholder - will be filled by run_held_one_out.py

seed_combinations:
  - [0]

layers: [16, 20, 24]

cv_folds: 5
alpha_sweep_size: 5
```

The `DATASET` placeholder is automatically substituted with each held-out dataset name. No need to create separate configs!

## Output

### Summary Table

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

### Results JSON

Saved to `{output_dir}/hoo_summary.json`:

```json
{
  "created_at": "2025-01-21T10:30:45.123456",
  "config": {
    "template": "post_task_qualitative_013",
    "seeds": [0],
    "datasets": ["alpaca", "bailbench", "math", "wildchat"]
  },
  "folds": [
    {
      "eval_dataset": "alpaca",
      "training_datasets": ["bailbench", "math", "wildchat"],
      "manifest_dir": "probe_data/manifests/probe_hoo_alpaca",
      "results_file": "results/hoo_basic/probe_hoo_eval_alpaca.json",
      "n_probes": 12,
      "probes": [
        {
          "id": "0001",
          "layer": 16,
          "trained_on_templates": ["post_task_qualitative_013"],
          "trained_on_datasets": ["bailbench", "math", "wildchat"],
          "eval_metrics": {
            "r2": 0.1234,
            "mse": 0.4567,
            "pearson_r": 0.3421,
            "n_samples": 234,
            "predictions": [...]
          }
        },
        ...
      ]
    },
    ...
  ]
}
```

Each fold contains:
- Which dataset was held out
- Which datasets were used for training
- Evaluation metrics for all probes
- Full predictions for detailed analysis

## Interpreting Results

### Good Cross-Dataset Generalization

If a probe trains on datasets A, B, C and achieves:
- R² on held-out D ≈ R² on training datasets → good generalization
- Example: Train R² = 0.12, Eval R² = 0.11 ✓

### Poor Cross-Dataset Generalization

- R² on held-out D ≪ R² on training datasets → overfitting to specific datasets
- Example: Train R² = 0.25, Eval R² = 0.05 ✗

### Dataset-Specific Signals

Compare R² across eval datasets to find which datasets have clearer preference signals:

```
Eval Dataset    Median R²    (signals quality)
alpaca          0.15         ← Clearest signal
math            0.12         ← Medium signal
wildchat        0.08         ← Noisier
```

## Tips & Tricks

### Faster Prototyping

Use `--skip-training` to test evaluation logic without retraining:

```bash
# First run: train probes
python -m src.probes.run_held_one_out --base-config ... --template ...

# Subsequent runs: reuse probes, fast evaluation
python -m src.probes.run_held_one_out --base-config ... --template ... --skip-training
```

### Subset Validation

Test on a subset of datasets first:

```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 \
  --datasets alpaca math \
  --output-dir results/hoo_test/
```

### Multiple Seeds for Stability

Run with several seeds to check if results are stable:

```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 1 2 3 4 \
  --output-dir results/hoo_seeds/
```

Then compare median R² across seeds.

### Combining with Manual Evaluation

You can still use the individual evaluation script if needed:

```bash
# Use the generated manifests from held-one-out
python -m src.probes.run_probe_evaluation \
  --config configs/probe_eval.yaml \
  --output results/custom_eval.json
```

Edit the evaluation config to point to the manifest_dir from a specific fold.

## Troubleshooting

**"Could not find measurement run"**
- Ensure measurements exist: `results/experiments/probe_hoo_DATASET/post_task_stated/`
- Run measurements before running held-one-out

**"No samples after dataset filter"**
- The eval dataset may not have tasks with measurements
- Check that tasks exist in completions_with_activations.json

**"Probe not found in manifest"**
- Training may have failed silently
- Check logs from training subprocess

**Memory issues during training**
- Reduce `alpha_sweep_size` in base config
- Reduce number of layers
- Run on a machine with more RAM
