# Exp 3C: Minimal Pairs Anti-Prompt Activations

## Goal

Extract activations for the "anti" (version C) minimal pairs conditions on RunPod, then transfer them back locally for analysis.

## Background

Exp 3 showed that single-sentence interest additions to role biographies produce selective probe responses: the target-matched task has mean probe rank 6.7/50 (65% in top 5). But we only extracted A (pro) and B (neutral) conditions. The C (anti) conditions — where the sentence explicitly expresses dislike for the target — have behavioral data but no activations.

The original minimal pairs v7 report showed that A vs C behavioral deltas are even more specific than A vs B (10.5x vs 6.9x specificity, 100% hit rate). The key question: does the probe also show this enhanced specificity for A vs C?

## What to do

### 1. Extract C condition activations (RunPod)

```bash
python -m scripts.ood_system_prompts.extract_ood_activations --exp exp3c
```

This extracts 20 conditions (midwest + brooklyn × 10 targets, version C) × 50 tasks at layers 31, 43, 55. Saves to `activations/ood/exp3_minimal_pairs/{condition_id}/activations_prompt_last.npz`. The baseline already exists from the original exp3 extraction.

### 2. Transfer activations back locally

Sync the new C condition activation files from RunPod back to the local machine.

### 3. Run analysis locally

Once activations are local, run the selectivity and full exp3 analysis (steps from the original spec: selectivity with A/B/C, updated plots, report updates).

## Expected results

- C conditions should show negative probe deltas on the target task (anti suppresses)
- A vs C probe specificity should exceed A vs B (matching the behavioral pattern)
- Target task probe rank under C should be near 50/50 (bottom of the list)

## Data dependencies

- Behavioral data: `results/ood/minimal_pairs_v7/behavioral.json` (all 127 conditions including C — already exists)
- Probe: `results/probes/gemma3_10k_heldout_std_demean/probes/probe_ridge_L{31,43,55}.npy`
- Existing activations: `activations/ood/exp3_minimal_pairs/` (baseline + 40 A/B conditions)
- Config: `configs/ood/prompts/minimal_pairs_v7.json`
