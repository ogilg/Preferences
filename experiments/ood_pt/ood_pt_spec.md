# OOD System Prompts with Pre-Trained Model Probe

## Goal

Re-run OOD experiments 1a-1d using the Gemma 3 27B pre-trained (PT) model and its probe, instead of the instruct model. The PT probe predicts instruct preferences at r=0.770 (vs IT's r=0.864). The question is whether this pre-training-derived preference direction also tracks artificially induced preference shifts from system prompts — shifts the PT model never encountered during training.

## Background

The OOD experiments (report at `experiments/ood_system_prompts/ood_system_prompts_report.md`) showed that a probe trained on natural preferences generalises to system-prompt-induced shifts. The EOT variant (`experiments/ood_eot/`) tested whether EOT-position probes improve this. This experiment asks a different question: does a probe trained on *pre-trained model activations* (no post-training) also track these shifts?

Original results (IT prompt_last, L31):

| Experiment | Overall r | On-target r |
|------------|----------|-------------|
| 1a: Category preference | 0.61 | 0.90 |
| 1b: Hidden preference | 0.65 | 0.95 |
| 1c: Crossed preference | 0.66 | 0.86 |
| 1d: Competing preference | 0.78 | 0.88 |

PT probe performance (heldout, predicting IT preferences):

| Layer | Heldout r |
|-------|-----------|
| L31 (0.5 depth) | 0.770 |
| L37 (0.6 depth) | 0.764 |
| L55 (0.9 depth) | 0.765 |

Cross-model pilot (`scripts/ood_pt/cross_probe_eval.py`, n=29,996 shared tasks at L31):

- Probe cosine similarity (IT vs PT): **0.13** — very different directions
- PT probe on PT acts vs IT probe on IT acts: **r=0.87** — both recover similar preference orderings
- PT probe on IT acts: **r=0.81** — PT direction transfers well to IT activations
- IT probe on PT acts: **r=0.46** — IT direction does not transfer well to PT activations
- Transfer is asymmetric: PT→IT works much better than IT→PT

## Methodology

Same behavioral data, same system prompts, same tasks, same analysis pipeline. Two things change:

1. **Model**: Gemma 3 27B PT (`gemma-3-27b-pt`) instead of IT
2. **Probe**: `results/probes/gemma3_pt_10k_heldout_std_raw/probes/probe_ridge_L31.npy` (trained on PT activations, IT preference scores, r=0.770)

Token position is `prompt_last` — the PT model has no chat template and no `<end_of_turn>` token. System prompts are prepended as plain text (the extraction config handles this via `system_prompt` field, which the PT model receives as a prefix).

## What we reuse

- **Behavioral data**: `results/ood/{category,hidden,crossed,competing}_preference/pairwise.json`. No re-measurement needed — these are instruct model choices, which define the ground truth we're predicting.
- **System prompts and tasks**: AL configs in `configs/measurement/active_learning/ood_exp1{a,b,c,d}/`. Each yaml has `measurement_system_prompt`, and either `custom_tasks_file` (1b-1d) or `include_task_ids_file` (1a).
- **PT probe**: `results/probes/gemma3_pt_10k_heldout_std_raw/probes/probe_ridge_L31.npy` (trained on 10k PT activations, heldout r=0.770).

## Step 1: Extract PT activations under system prompts (GPU)

Add a `run_ood_pt_extractions()` function to `scripts/run_all_extractions.py` that calls `run_ood("activations/ood_pt", [31], ["prompt_last"])` with `model="gemma-3-27b-pt"`.

This requires a small modification to `run_ood()` — currently the model is hardcoded to `gemma-3-27b`. Add a `model` parameter (default `"gemma-3-27b"`).

```bash
python -m scripts.run_all_extractions_pt
```

(Or add `run_ood_pt_extractions()` to `scripts/run_all_extractions.py` and call it from a thin wrapper script.)

Output: `activations/ood_pt/{exp1_category,exp1_prompts}/{condition}/activations_prompt_last.npz`

| Exp | Conditions | Tasks/condition | Forward passes |
|-----|-----------|----------------|---------------|
| 1a | 13 (12 persona + baseline) | 50 (standard pool) | 650 |
| 1b | 17 (16 targeted + baseline) | 48 (custom tasks) | 816 |
| 1c | 17 (16 targeted + baseline) | 48 (crossed tasks) | 816 |
| 1d | 17 (16 competing + baseline) | 48 (crossed tasks) | 816 |
| **Total** | | | **3,098** |

~20 min on A100. Same directory structure as the IT OOD extractions.

## Step 2: Analysis

`scripts/ood_pt/analyze_pt.py` — copy of `scripts/ood_eot/analyze_eot.py`, parameterized to run three conditions:

1. **PT probe on PT activations** (`activations/ood_pt/`, PT probe) — the main question: does the PT model's preference direction track OOD shifts?
2. **PT probe on IT activations** (`activations/ood/`, PT probe) — does the pre-training direction remain readable in post-trained activations under system prompts?
3. **IT probe on PT activations** (`activations/ood_pt/`, IT probe at `results/probes/gemma3_10k_heldout_std_raw/probes/probe_ridge_L31.npy`) — does the post-training direction exist in base model activations under system prompts?

This completes the full 2x2: {IT probe, PT probe} x {IT acts, PT acts}. The IT probe on IT acts results already exist in the original OOD report.

```bash
python scripts/ood_pt/analyze_pt.py --exp all
```

For each experiment it runs all three conditions through the same pipeline: load behavioral rates, score activations with the relevant probe, correlate deltas. All analysis is at L31 only.

Note: conditions 2 and 3 (cross-model) require no new GPU extraction — IT activations at `activations/ood/` already exist from the original OOD experiment. These can run locally after step 1 completes.

Output: `experiments/ood_pt/analysis_results.json` (summary) and `analysis_results_full.json`.

## Deliverable

Report at `experiments/ood_pt/ood_pt_report.md` with:
- Scatter plots (behavioral delta vs probe delta) for all four experiments, for each of the three new conditions
- Full 2x2 comparison table: {IT probe, PT probe} x {IT acts, PT acts} OOD correlations for all four experiments
- Discussion: does pre-training already encode enough preference structure to track OOD shifts, or does post-training add critical signal? Is transfer symmetric (PT probe on IT acts vs IT probe on PT acts)?
