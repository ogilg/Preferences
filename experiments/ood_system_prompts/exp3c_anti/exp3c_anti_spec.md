# Exp 3C: Minimal Pairs Anti-Prompt Activations

## Goal

Extract activations for the "anti" (version C) minimal pairs conditions and run the selectivity analysis comparing pro (A) vs anti (C) probe deltas on the target task.

## Background

Exp 3 showed that single-sentence interest additions to role biographies produce selective probe responses: the target-matched task has mean probe rank 6.7/50 (65% in top 5). But we only extracted A (pro) and B (neutral) conditions. The C (anti) conditions — where the sentence explicitly expresses dislike for the target — have behavioral data but no activations.

The original minimal pairs v7 report showed that A vs C behavioral deltas are even more specific than A vs B (10.5x vs 6.9x specificity, 100% hit rate). The key question: does the probe also show this enhanced specificity for A vs C?

## What to do

### 1. Extract C condition activations

```bash
python -m scripts.ood_system_prompts.extract_ood_activations --exp exp3c
```

This extracts 20 conditions (midwest + brooklyn × 10 targets, version C) × 50 tasks at layers 31, 43, 55. Saves to `activations/ood/exp3_minimal_pairs/{condition_id}/activations_prompt_last.npz`. The baseline already exists from the original exp3 extraction.

### 2. Run selectivity analysis with A, B, and C

Update `scripts/ood_system_prompts/plot_exp3_selectivity.py` to:
- Include C conditions alongside A conditions
- For each (base_role, target): compare the target task's probe rank under A (pro) vs C (anti)
- Key metric: does A push the target task UP in probe rank and C push it DOWN?
- Compute A-C probe delta specificity (analogous to A-C behavioral specificity from v7 report)

### 3. Rerun full exp3 analysis with C conditions included

Update `scripts/ood_system_prompts/analyze_ood.py` exp3 to include C conditions (change `selected_versions` from `{"A", "B"}` to `{"A", "B", "C"}`), rerun, and update the report numbers.

### 4. Generate plots

- Selectivity comparison: A vs C probe rank distributions (side by side with A vs B)
- Per-target: A probe delta vs C probe delta for the target task (should be opposite signs)

### 5. Update the report

Add the A vs C selectivity results to the Exp 3 section of `experiments/ood_system_prompts/ood_system_prompts_report.md`. Regenerate the task examples JSON (`scripts/ood_system_prompts/dump_top_deltas.py`) with C conditions included.

## Expected results

- C conditions should show negative probe deltas on the target task (anti suppresses)
- A vs C probe specificity should exceed A vs B (matching the behavioral pattern)
- Target task probe rank under C should be near 50/50 (bottom of the list)

## Data dependencies

- Behavioral data: `results/ood/minimal_pairs_v7/behavioral.json` (all 127 conditions including C — already exists)
- Probe: `results/probes/gemma3_10k_heldout_std_demean/probes/probe_ridge_L{31,43,55}.npy`
- Existing activations: `activations/ood/exp3_minimal_pairs/` (baseline + 40 A/B conditions)
- Config: `configs/ood/prompts/minimal_pairs_v7.json`
