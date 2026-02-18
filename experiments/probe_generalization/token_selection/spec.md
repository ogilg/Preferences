# Token Selection Ablation: prompt_last vs prompt_mean

## Question

Does averaging over all prompt tokens (prompt_mean) yield better or worse preference probes than using the last prompt token (prompt_last)?

## Setup

**Model:** Gemma 3 27B
**Probe type:** Ridge only
**Targets:** Thurstonian scores (standardized, raw — no demeaning)
**Layers:** [31, 43, 55] (0.5, 0.7, 0.9 fractional)

**Train:** 3k run at `results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0`
**Eval:** 4k run at `results/experiments/gemma3_4k_pre_task/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0`

Alpha sweep (Pearson r on sweep half of eval set) done independently per selector. Final metrics (R², pairwise accuracy) on the other half of the eval set.

## Steps

### 1. Extract prompt_mean activations

`prompt_mean` selector already implemented in `src/models/base.py` and tested.

Create `configs/extraction/gemma3_27b_prompt_mean.yaml` — identical to `configs/extraction/gemma3_27b_prompt_last.yaml` but with `selectors: [prompt_mean]`. Run on RunPod:

```bash
python -m src.probes.extraction.run configs/extraction/gemma3_27b_prompt_mean.yaml --resume
```

Output: `activations/gemma_3_27b/activations_prompt_mean.npz`.

### 2. Train probes and evaluate

Use `src/probes/experiments/eval_on_heldout.py` (NOT `run_dir_probes.py`). CLI:

```bash
python -m src.probes.experiments.eval_on_heldout --config <config>.yaml
```

Run both selectors fresh for a fair comparison. Create two configs — they differ only in `activations_path` and naming.

`configs/probes/heldout_eval_gemma3_token_sel_prompt_mean.yaml`:

```yaml
experiment_name: heldout_eval_gemma3_token_sel_prompt_mean
train_run_dir: results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0
eval_run_dir: results/experiments/gemma3_4k_pre_task/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0
activations_path: activations/gemma_3_27b/activations_prompt_mean.npz
output_dir: results/probes/heldout_eval_gemma3_token_sel_prompt_mean
layers: [31, 43, 55]
standardize: true
alpha_sweep_size: 10
eval_split_seed: 42
```

For `prompt_last`, create an identical config but with `activations_path: activations/gemma_3_27b/activations_prompt_last.npz` and names adjusted to `..._prompt_last`.

### 3. Report

Side-by-side table of final_r and final_acc by layer and selector.
