# Probe Evaluation Quick Reference

## One-Line Held-One-Out Validation

```bash
python -m src.probes.run_held_one_out --base-config configs/probe_training/hoo_base.yaml --template post_task_qualitative_013 --seeds 0 --output-dir results/hoo/
```

That's it! No need to create 4 separate training configs and 4 separate eval configs.

## What It Does

1. **Detects datasets** → ALPACA, BAILBENCH, MATH, WILDCHAT
2. **For each dataset:**
   - Trains probes on N-1 datasets
   - Evaluates on held-out dataset
3. **Prints summary table** showing generalization per dataset
4. **Saves results** to `results/hoo/hoo_summary.json`

## Common Commands

### Train & Evaluate (Full Pipeline)
```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 \
  --output-dir results/hoo_full/
```

### Evaluate Only (Reuse Trained Probes)
```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 \
  --skip-training \
  --output-dir results/hoo_eval/
```

### Multiple Seeds (Robustness Check)
```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 1 2 3 4 \
  --output-dir results/hoo_seeds/
```

### Specific Datasets Only
```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 \
  --datasets alpaca math \
  --output-dir results/hoo_subset/
```

### Multiple Templates
```bash
python -m src.probes.run_held_one_out \
  --base-config configs/probe_training/hoo_base_multi_template.yaml \
  --template post_task_qualitative_013 \
  --seeds 0 \
  --output-dir results/hoo_templates/
```

## Manual Evaluation (No Template Loop)

```bash
# Train once
python -m src.probes.train_probe_experiment --config configs/probe_training.yaml

# Evaluate on specific dataset
python -m src.probes.run_probe_evaluation --config configs/probe_eval.yaml
```

## Individual Workflows

### Train and Evaluate (Specific Config)
```bash
python -m src.probes.run_train_and_evaluate \
  --train-config configs/probe_training.yaml \
  --eval-configs configs/probe_eval.yaml \
  --output-dir results/cross_eval/
```

## Config Templates

### Base Held-One-Out Config
```yaml
experiment_name: probe_hoo
experiment_dir: results/experiments/probe_hoo_DATASET    # DATASET placeholder
activations_path: probe_data/activations/activations.npz
manifest_dir: probe_data/manifests/probe_hoo_DATASET     # DATASET placeholder

template_combinations:
  - [post_task_qualitative_013]

dataset_combinations:
  - []  # Auto-configured

seed_combinations:
  - [0]

layers: [16, 20, 24]

cv_folds: 5
alpha_sweep_size: 5
```

### Evaluation Config
```yaml
probe:
  manifest_dir: probe_data/manifests/my_probe
  probe_ids: []  # Empty = all probes

data:
  experiment_dir: results/experiments/my_experiment
  template: post_task_qualitative_013
  seeds: [0]
  dataset_filter: null  # or specific dataset

output:
  results_file: results/probe_eval.json
```

## Output Locations

```
results/hoo_basic/
├── hoo_summary.json           # All results (folds + metrics)
├── probe_hoo_eval_alpaca.json
├── probe_hoo_eval_bailbench.json
├── probe_hoo_eval_math.json
└── probe_hoo_eval_wildchat.json

probe_data/manifests/
├── probe_hoo_alpaca/
│   ├── manifest.json
│   └── probes/
│       ├── probe_0001.npy
│       ├── probe_0002.npy
│       └── ...
├── probe_hoo_bailbench/
├── probe_hoo_math/
└── probe_hoo_wildchat/
```

## Key Metrics

In results JSON, look at:

```json
"eval_metrics": {
  "r2": 0.15,              # R² score (0-1, higher is better)
  "mse": 0.45,             # Mean squared error (lower is better)
  "pearson_r": 0.38,       # Pearson correlation (0-1)
  "n_samples": 234,        # Number of eval samples
  "predictions": [...]     # Model predictions
}
```

## Interpretation

- **Good generalization**: R² on held-out dataset ≈ train R²
- **Poor generalization**: R² on held-out dataset ≪ train R²
- **Dataset quality**: Compare R² across eval datasets
  - Higher R² = clearer preference signal in that dataset

## File Organization

```
configs/probe_training/
├── hoo_base.yaml                    # Single template HOO
├── hoo_base_multi_template.yaml     # Multiple templates HOO
└── probe_training_example.yaml      # Manual training example

src/probes/
├── run_held_one_out.py             # Orchestrator (NEW!)
├── run_train_and_evaluate.py        # Manual orchestrator
├── run_probe_evaluation.py          # Single eval runner
├── train_probe_experiment.py        # Training script
└── evaluate.py                      # Core evaluation function
```

## Documentation

- `PROBE_EVALUATION.md` — Full documentation
- `HOO_EXAMPLES.md` — Detailed held-one-out examples
- `QUICK_REFERENCE.md` — This file!
