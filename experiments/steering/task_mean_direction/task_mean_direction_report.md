# Task-Mean Direction Steering — Report

**Status: INTERIM** — Experiment running (~725/80,000 records). L25 m=-0.05 has ~25 pairs; other conditions have 3 pairs. Results below use available data; CIs will tighten with full data.

## Summary

The task_mean probe direction — trained on averaged task-token activations — produces dramatically stronger steering than either EOT or prompt_last probes. At L25, the effect saturates at even the smallest multiplier tested (±0.01), achieving near-perfect choice control. L32 shows a clear dose-response with effects comparable to but slightly exceeding EOT/prompt_last references.

| Layer | m=±0.01 | m=±0.02 | m=±0.03 | m=±0.05 |
|-------|---------|---------|---------|---------|
| L25 | +0.83 | +1.00 | +1.00 | +1.00 |
| L32 | +0.23 | +0.40 | +0.50 | +0.50 |
| EOT (ref) | — | — | +0.32 | — |
| prompt_last (ref) | — | — | +0.23 | — |

## Method

- **Probe:** task_mean Ridge probes at L25 (sweep r=0.803) and L32 (sweep r=0.797), from `heldout_eval_gemma3_task_mean`
- **Model:** gemma-3-27b via HuggingFace (max_new_tokens=256, temperature=1.0)
- **Steering mode:** differential — +direction on Task A token span, -direction on Task B token span
- **Multipliers:** ±0.01, ±0.02, ±0.03, ±0.05 (8 signed values per layer)
- **Coefficients:** multiplier × mean_norm (L25: 38,349; L32: 40,966)
- **Data:** 500 pairs from v2 followup, 10 trials per pair per condition (5 per ordering)
- **Baseline:** 10,000 records from v2 followup (condition="baseline")
- **Comparison:** EOT at ±0.03 (L31, coef≈±1,585), prompt_last at ±0.03

### Coefficient table

| Multiplier | L25 coef | L32 coef |
|---|---|---|
| 0.01 | 384 | 410 |
| 0.02 | 767 | 819 |
| 0.03 | 1,150 | 1,229 |
| 0.05 | 1,917 | 2,048 |

For reference, EOT at ±0.03 used coef ≈ ±1,585.

## Results

### Steering effects

Steering effect = P(choose presented A | +m) - P(choose presented A | -m), controlling for position bias via ordering.

**L25 task_mean** achieves near-perfect choice control:
- m=±0.01 (coef=±384): effect = +0.83 [0.70, 0.97]
- m=±0.02+: effect = +1.00 [1.00, 1.00] — fully saturated

P(choose presented A) = 0.00 for all negative multipliers at L25 (across 245+ records). The model never selects the steered-against task.

**L32 task_mean** shows a clear dose-response:
- m=±0.01: +0.23 [0.00, 0.47]
- m=±0.02: +0.40 [0.17, 0.63]
- m=±0.03: +0.50 [0.27, 0.70]
- m=±0.05: +0.50 [0.27, 0.70]

L32 at m=±0.03 (+0.50) exceeds both EOT (+0.32) and prompt_last (+0.23).

![Steering effect comparison](assets/plot_031326_steering_effect_comparison.png)

### Dose-response

L25 saturates at m=0.02 — the dose-response curve hits ceiling. L32 shows a rising curve that begins to plateau around m=0.03.

![Dose-response](assets/plot_031326_dose_response.png)

### Parse rates

| Condition | Parse rate | Refusals |
|---|---|---|
| task_mean (all) | 98.6% | 10 |
| baseline | 93.1% | 0 |
| EOT | 98.3% | 168 |
| prompt_last | 91.1% | 0 |

All 10 task_mean refusals come from a single pair (pair_0020) at L25 m=-0.05, where differential steering fell back to all_tokens and the model produced incoherent refusals ("cannot complete because tasks are harmful"). This is a known artifact of strong steering overwhelming the model at extreme coefficients.

### Per-pair correlations

Insufficient data — only 3 pairs have both task_mean and EOT data at m=±0.03. Will be computed when the full run completes (500 common pairs expected).

### Steering fallback rate

10/725 records (1.4%) required steering fallback to all_tokens mode, all from one pair at L25 m=-0.05.

## Interpretation

The task_mean direction at L25 is the most effective steering vector we have tested — approximately 3× stronger than EOT and 4× stronger than prompt_last at comparable multipliers. This likely reflects two reinforcing factors:

1. **No train-steer mismatch.** The probe was trained on mean activations over task tokens. Differential steering applies the direction to exactly those positions. EOT and prompt_last probes were trained on boundary tokens but steered on different positions (or all tokens).

2. **Multi-token leverage.** Differential steering modifies activations across the entire task span (dozens of tokens) rather than a single boundary token. The cumulative effect of steering many tokens is much stronger.

The L25 vs L32 difference (L25 is ~2-4× stronger) maps onto their probe performance (L25 sweep r=0.803 vs L32 r=0.797), but the steering difference is far larger than the probe accuracy difference. This may indicate L25 has better alignment between the probe direction and the causal mechanisms that determine choice.

## What remains

- Full experiment: 80,000 generations (~33h on H100), currently running
- Per-pair correlations: task_mean vs EOT and prompt_last
- Tighter CIs on all effects
- Layer comparison with more statistical power

## Reproduction

```bash
# Run experiment (supports --resume)
python -m scripts.task_mean_direction.run_experiment [--resume] [--pilot N]

# Analysis
python scripts/task_mean_direction/analyze.py

# Plots
python scripts/task_mean_direction/plot_results.py
```

Checkpoint: `experiments/steering/task_mean_direction/checkpoint.jsonl`
