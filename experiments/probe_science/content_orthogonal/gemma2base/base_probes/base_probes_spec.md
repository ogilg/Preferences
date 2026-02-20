# Probing Gemma-2 27B base activations for preferences

Follow-up to the parent experiment (`../gemma2base_spec.md`). Train standard probes on Gemma-2 27B base (`google/gemma-2-27b`) activations and compare to Gemma-3 27B IT. No content-orthogonal projection — just straight probing.

## Why this matters

If a base model's activations predict preferences nearly as well as the IT model's, it suggests the evaluative structure is already present in pretraining. If performance drops substantially, the evaluative representations may be shaped by alignment training.

## Method

**Do not re-implement any of the steps below.** The extraction pipeline, probe training, and evaluation code all exist. The only new artifacts are two small YAML config files.

### Step 1: Extract activations

Use the existing extraction pipeline: `python -m src.probes.extraction.run <config.yaml>`.

Create a new config at `configs/extraction/gemma2_27b_prompt_last.yaml` modeled on `configs/extraction/gemma3_27b_prompt_last.yaml`:

```yaml
model: gemma-2-27b          # HuggingFace ID: google/gemma-2-27b (base, NOT IT)
backend: huggingface

n_tasks: 30000
task_origins:
  - wildchat
  - alpaca
  - math
  - bailbench
  - stress_test
seed: 42

selectors: [prompt_last]
layers_to_extract: [0.25, 0.5, 0.6, 0.7, 0.8, 0.9]

batch_size: 32
save_every: 1000
```

Fractional layers map to absolute layers via the model's layer count (46 layers for Gemma-2 27B). Load in bf16 to keep memory manageable (~54GB).

Output goes to `activations/gemma_2_27b/activations_prompt_last.npz`.

### Step 2: Train probes

Use the existing probe pipeline: `python -m src.probes.experiments.run_dir_probes --config <config.yaml>`.

Create a new config at `configs/probes/gemma2_27b_std_raw.yaml` modeled on `configs/probes/gemma3_3k_std_raw.yaml`:

```yaml
experiment_name: gemma2_27b_std_raw

run_dir: results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0

activations_path: activations/gemma_2_27b/activations_prompt_last.npz

output_dir: results/probes/gemma2_27b_std_raw

layers: [11, 23, 27, 32, 36, 41]   # 0.25/0.5/0.6/0.7/0.8/0.9 of 46 layers

modes: [ridge, bradley_terry]

cv_folds: 5
alpha_sweep_size: 10

standardize: true
```

Note: `standardize: true` is important — use standardized activations to match the Gemma-3 baseline (`gemma3_3k_std_raw`).

### Step 3: Compare

Compare probe R² across layers to the Gemma-3 27B IT baseline:

| Layer (fractional) | Gemma-3 27B IT R² |
|--------------------|-------------------|
| 0.5 (L31)         | 0.863             |
| 0.7 (L43)         | 0.840             |
| 0.9 (L55)         | 0.835             |

## Existing code — use it, don't rewrite it

- **Extraction**: `src/probes/extraction/run.py` — config-driven, handles batching, checkpointing, resume
- **Activation loading**: `src/probes/core/activations.py` — `load_activations()` with task ID filtering and layer selection
- **Probe training**: `src/probes/core/linear_probe.py` — Ridge with CV alpha sweep (`train_and_evaluate`)
- **Probe orchestration**: `src/probes/experiments/run_dir_probes.py` — loads scores, loads activations, runs probe sweep
- **Evaluation**: `src/probes/core/evaluate.py` — `evaluate_probe_on_data`, cross-template transfer

## Data

Same preference scores as all other Gemma-3 probe experiments:
- `results/experiments/gemma3_3k_run2/` — Thurstonian scores and pairwise comparisons
- `results/probes/gemma3_3k_std_raw/` — Gemma-3 baseline probe results for comparison