# OOD System Prompts with EOT Probes

## Goal

Re-run OOD experiments 1a–1d using probes trained on `<end_of_turn>` activations instead of `prompt_last`. The EOT probes generalise much better across topics (1.8% gap vs 8.8%), so the question is whether this also improves OOD tracking of artificially induced preferences.

## Background

The original OOD experiments (report at `experiments/ood_system_prompts/ood_system_prompts_report.md`) showed prompt_last probes track system-prompt-induced preference shifts. But all activations were extracted at `prompt_last`. To use EOT probes, we need EOT activations under each system prompt condition.

Original results (prompt_last, L31):

| Experiment | Overall r | On-target r |
|------------|----------|-------------|
| 1a: Category preference | 0.61 | 0.90 |
| 1b: Hidden preference | 0.65 | 0.95 |
| 1c: Crossed preference | 0.66 | 0.86 |
| 1d: Competing preference | 0.78 | 0.88 |

## Methodology

This must be an exact replication of the original OOD experiments — same behavioral data, same system prompts, same tasks, same analysis pipeline, same ground truth labels. The ONLY thing that changes is the token position (eot instead of prompt_last) and the probe (trained on EOT activations). Any difference in results is attributable solely to the token position.

## What we reuse

- **Behavioral data**: `results/ood/{category,hidden,crossed,competing}_preference/pairwise.json`. No re-measurement needed.
- **System prompts and tasks**: AL configs in `configs/measurement/active_learning/ood_exp1{a,b,c,d}/`. Each yaml has `measurement_system_prompt`, and either `custom_tasks_file` (1b-1d) or `include_task_ids_file` (1a) to specify which tasks to extract.
- **EOT probe**: `results/probes/heldout_eval_gemma3_eot/probes/probe_ridge_L31.npy` (trained on 10k, evaluated on 4k heldout, r=0.868).

## Step 1: Extract EOT activations (GPU)

`scripts/run_all_extractions.py` has a shared `run_ood(act_base, layers, selectors)` that iterates the AL configs for exp1a-1d, reads `measurement_system_prompt` / `custom_tasks_file` / `include_task_ids_file` from each, and runs `src.probes.extraction.extract.run_extraction()` with `--resume`.

```bash
python -c "from scripts.run_all_extractions import run_ood_eot_extractions; run_ood_eot_extractions()"
```

This calls `run_ood("activations/ood_eot", [31], ["eot"])`, which extracts at the `eot` selector, layer 31 only, writing to `activations/ood_eot/{exp1_category,exp1_prompts}/{condition}/activations_eot.npz`.

| Exp | Conditions | Tasks/condition | Forward passes |
|-----|-----------|----------------|---------------|
| 1a | 13 (12 persona + baseline) | 50 (standard pool) | 650 |
| 1b | 17 (16 targeted + baseline) | 48 (custom tasks) | 816 |
| 1c | 17 (16 targeted + baseline) | 48 (crossed tasks) | 816 |
| 1d | 17 (16 competing + baseline) | 48 (crossed tasks) | 816 |
| **Total** | | | **3,098** |

~20 min on A100. 1b, 1c, and 1d share `exp1_prompts/` because their condition IDs don't overlap (targeted vs crossed vs compete prefixes).

## Step 2: Analysis

`scripts/ood_eot/analyze_eot.py` is the EOT counterpart of `scripts/ood_system_prompts/analyze_ood.py`. Same per-experiment analysis logic — condition filtering (targeted vs competing, `hidden_*` vs `crossed_*` task ID prefixes), on-target/off-target splits for 1d, etc.

```bash
python scripts/ood_eot/analyze_eot.py --exp all
```

For each experiment it:
1. Loads behavioral rates from `results/ood/*/pairwise.json` via `compute_p_choose_from_pairwise()`
2. Scores activations with the EOT probe via `compute_deltas()` — loads each condition's `activations_eot.npz`, applies `weights @ acts + bias`, computes `probe_delta = cond_score - baseline_score` per task
3. Correlates with behavioral deltas via `correlate_deltas()` — Pearson r, Spearman r, sign agreement, permutation p

Output: `experiments/ood_eot/analysis_results.json` (summary) and `analysis_results_full.json` (with raw delta arrays for plotting).

## Deliverable

Report at `experiments/ood_eot/ood_eot_report.md` with:
- reproduce the scatter plots from section3_draft.md of the blog post. And compute the Pearson correlations on and off target.
- Side-by-side comparison: prompt_last vs EOT probe OOD correlations for all four experiments
