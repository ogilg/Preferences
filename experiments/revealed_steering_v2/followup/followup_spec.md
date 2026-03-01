# Revealed Steering v2 — Follow-up

## Motivation

The v2 experiment established that differential steering along the L31 probe direction causally shifts pairwise choices (steering effect = +0.166 at mult=+0.02), but two limitations weaken the analysis:

1. **Baseline resolution:** Only 10 trials per pair (5 per ordering) → 11 possible P(A) values. 41% of pairs saturate at P(A)=0.0 or 1.0, making the steerability-vs-decidedness analysis unreliable.
2. **Pair coverage:** Only 300 pairs, all selected as borderline from the original measurement. We don't know if steering effects differ for non-borderline pairs or hold across a broader sample.

## Design

### Pairs

**500 total:**
- 300 original pairs from `experiments/steering/replication/fine_grained/results/pairs.json`
- 200 new pairs from the full 10k task pool (no P(A) filter), selected by `scripts/revealed_steering_v2/select_decided_pairs.py`
  - Constraints: |delta_mu| < 2, both tasks have activations, not already in the 300
  - Stratified: 20 per mu_bin (10 bins)
  - New pair IDs: pair_0300 through pair_0499

New pairs stored at `experiments/revealed_steering_v2/followup/pairs_200_new.json`.

Combined pairs file: `experiments/revealed_steering_v2/followup/pairs_500.json` — the 300 original + 200 new, with updated baseline P(A) from the 20-trial measurement below.

### Phase 1: Baseline recompute (20 trials)

Collect P(A) for all 500 pairs with 20 trials (10 per ordering) at t=1.0 using the standard measurement pipeline. This gives 21 possible P(A) values per pair instead of 11.

- Template: `completion_preference` (canonical)
- Temperature: 1.0
- Max new tokens: 256
- Model: Gemma 3 27B (local HF, bfloat16)
- Steering: coef=0 (no steering — just baseline measurement)
- 500 pairs × 20 trials = 10,000 generations → ~7 hours

Store baseline P(A) per pair in `pairs_500.json` as `p_a_baseline_20` (to distinguish from the old 10-trial baseline).

### Phase 2: Steering sweep (probe direction only)

All 15 multipliers from v2:
```
[-0.15, -0.10, -0.07, -0.05, -0.03, -0.02, -0.01,
 0.0,
 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
```

- 10 trials per pair per multiplier (5 per ordering AB/BA)
- Differential steering mode (same as v2)
- 500 pairs × 15 multipliers × 10 trials = 75,000 generations → ~52 hours

No random direction control — already validated in v2 Phase 3.

### Measurement protocol

Same as v2:
- Template: `completion_preference` (canonical)
- Temperature: 1.0
- Max new tokens: 256
- Differential steering: +direction on task A span, -direction on task B span
- Ordering counterbalancing: both AB and BA
- Response parsing: prefix match + semantic parser fallback via OpenRouter
- Checkpoint after each (multiplier) block to a JSONL file

### Per-trial data format

Same as v2 checkpoint format:
```json
{
  "pair_id": "pair_0042",
  "task_a_id": "alpaca_1234",
  "task_b_id": "competition_math_5678",
  "coefficient": 1056.5,
  "multiplier": 0.02,
  "condition": "probe",
  "sample_idx": 3,
  "ordering": 0,
  "choice_original": "a",
  "choice_presented": "b",
  "raw_response": "Task B: To solve this problem..."
}
```

### Source data

- Thurstonian CSV: `results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_80fa9dc8.csv`
- Activations: `activations/gemma_3_27b/activations_prompt_last.npz`
- Probe: `results/probes/gemma3_10k_heldout_std_raw/` → `ridge_L31`
- Original pairs: `experiments/steering/replication/fine_grained/results/pairs.json`
- New pairs: `experiments/revealed_steering_v2/followup/pairs_200_new.json`

## Analysis plan

1. **Dose-response** — P(choose A) vs multiplier for all 500 pairs. Compare old 300 vs new 200 subsets.
2. **Steerability vs decidedness (properly resolved)** — With 20-trial baseline, decidedness = |P(A) - 0.5| has 10 distinct levels (vs 5 before). Plot per-pair steerability (max |shift| across multipliers) against decidedness.
3. **Ordering effect analysis** — Steering effect = (ordering_diff − baseline_ordering_diff) / 2. Same metric as v2 but with 500 pairs.
4. **Borderline vs non-borderline** — Compare steerability distributions for the original 300 (pre-selected borderline) vs the new 200 (unrestricted). If evaluative representations causally drive choice, steering should work regardless of how decided the pair is.

## GPU requirements

Same as v2: 1× H100 80GB (or A100 80GB).

## Estimated runtime

- Phase 1 (baseline): ~7 hours
- Phase 2 (steering): ~52 hours
- Total: ~59 hours (~2.5 days)
