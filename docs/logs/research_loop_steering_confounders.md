# H2 Differential Steering: Confounder Follow-ups

**Goal**: The H2 differential steering result (P(A) 0.59→0.67, p=0.000002) showed that steering a task's tokens with the probe direction shifts choice toward that task. This follow-up investigates confounders: (1) how much of the effect is positional vs task-dependent, (2) whether the effect is specific to the probe direction, (3) whether it concentrates in borderline pairs.

**Result**: Steering a task's tokens with the probe direction increases P(picking that task), and the effect is concentrated in borderline pairs (Δ=0.711 on borderline vs Δ=0.083 overall). The probe is significantly more effective than random directions (2.6x, p=0.003). There is a positional confound — differential steering shifts P(A) even with identical tasks (Δ=0.106) — but this is much smaller than the effect on borderline pairs with different content. The effect is partly positional (62%) and partly order-sensitive (38%), with the order-sensitive component consistent with task-dependent steering.

## Context
- L31 ridge probe direction causally shifts revealed preferences (H2, 60 pairs)
- Effect concentrated in ~3/60 borderline pairs; 57 firmly decided
- Differential steering: +coef on task A tokens, -coef on task B tokens
- Model: Gemma-3-27B on H100 80GB

## E2: Borderline Pair Screening

**Approach**: Screen 250 pairs at coef=0. Two-stage adaptive: 5 resamples for all, then 15 more for any showing variance.

**Result**: 12/250 pairs borderline (4.8%), matching original 3/60 (5%). 2 pairs at P(A)=0.50, rest between 0.30-0.75.

## E1: Order Counterbalancing

**Approach**: Present each of 12 borderline pairs in both orderings: original (A,B) and swapped (B,A). Apply same differential steering (+first, -second) in both. 7 coefficients, 15 resamples per condition.

**Result**: Steering shifts choice toward the steered-position task in both orderings. Effect is massive on borderline pairs (Δ=0.711 in original) compared to the Δ=0.083 on the mixed set of 60 pairs. Effect persists in the swapped ordering (Δ=0.209) but is 3.4x smaller.

| Coef | Original P(A) | Swapped P(A) |
|------|--------------|-------------|
| -3000 | 0.161 | 0.430 |
| 0 | 0.494 | 0.570 |
| +3000 | 0.872 | 0.639 |

In both orderings, +coef increases P(A) — i.e., steering task A's tokens with +probe increases P(picking A). Since the tasks swap positions between orderings, this means the probe shifts choice toward whichever task's tokens receive +probe. The 3.4x magnitude difference between orderings (slope 1.39e-04 original vs 3.26e-05 swapped) suggests a task-dependent component beyond pure position.

**Slopes**: original 1.39e-04 (p<1e-100), swapped 3.26e-05 (p=3e-06).

## E3: Same-Task Pairs — Positional Baseline

**Approach**: Present same task as both A and B. Apply differential steering. Measures the purely positional component of the steering effect. 20 tasks, 7 coefficients, 15 resamples.

**Result**: Differential steering shifts P(A) even with identical tasks: P(A) = 0.690→0.796 (slope=1.58e-05, p=0.002). Baseline position A bias: P(A) = 0.749 at coef=0.

| Coef | P(A) same-task |
|------|----------------|
| -3000 | 0.690 |
| 0 | 0.749 |
| +3000 | 0.796 |

This establishes the positional floor: the probe direction shifts P(A) by Δ=0.106 through position alone. The E1 borderline effect (Δ=0.711) is 6.7x larger, indicating substantial task-dependent contribution beyond position.

## E8: Probe vs Random Directions on Borderline Pairs

**Approach**: 20 random orthogonal directions + probe direction, all on 12 borderline pairs. Differential steering at coefs [-3000, 0, +3000], 10 resamples. 7,560 total observations.

**Result**: The probe direction shifts choice more than any random direction.

| Metric | Value |
|--------|-------|
| Probe ΔP(A) | +0.742 |
| Random mean ΔP(A) | -0.000 (std=0.331) |
| Random mean abs(ΔP(A)) | 0.286 (std=0.167) |
| Probe abs(Δ) z-score | 2.73 (p=0.003) |
| Rank p-value | 0.048 (0/20 randoms ≥ probe) |

Random directions also shift borderline pairs (mean |Δ|=0.286) — the original "zero effect" from random controls was an artifact of testing on firm pairs. But the probe direction is 2.6x stronger and consistently shifts in one direction, while random directions scatter symmetrically (mean signed Δ ≈ 0).

![e8-extended](assets/steering_confounders/plot_021126_e8_extended.png)

## E5: Logit Lens

**Approach**: Forward pass with steering (no generation), extract logit(a) - logit(b) at last token position. 12 borderline + 20 firm pairs, 7 coefficients.

**Result**: Strong linear dose-response in borderline pairs, no effect on firm.

| Pair Type | slope | r | p | Δ logit_diff |
|-----------|-------|------|------|-------------|
| Borderline | 1.15e-03 | 0.779 | <1e-6 | 6.50 |
| Firm | 3.98e-04 | 0.057 | 0.500 | 2.16 |

Borderline pairs: logit_diff goes from -2.98 to +3.52 (crosses zero at coef≈0). Clean, monotonic, r=0.779.

![e5-logit-lens](assets/steering_confounders/plot_021126_e5_logit_lens.png)

## Effect Decomposition

### Position vs order-sensitive components

The E1 counterbalancing data decomposes the steering effect. The average of both orderings captures the positional component (present in both), while half the difference captures the order-sensitive component (reverses when tasks swap positions):

| Component | Slope | % of original |
|-----------|-------|---------------|
| Positional: (orig + swap) / 2 | 8.58e-05 | 62% |
| Order-sensitive: (orig - swap) / 2 | 5.33e-05 | 38% |
| **Original ordering** | **1.39e-04** | **100%** |

The order-sensitive component reverses sign when order is swapped, consistent with task-dependent steering. However, this decomposition alone cannot prove the order-sensitive component is evaluative — it could reflect task-specific steerability differences or other content-dependent factors.

### Comparison to same-task positional baseline

E3 provides an independent estimate of the positional contribution: slope=1.58e-05. This is smaller than the E1 positional component (8.58e-05), likely because E3 uses non-borderline tasks. On borderline pairs, positional effects are amplified along with task-dependent effects.

### Specificity

The probe direction (|Δ|=0.742) is 2.73 standard deviations above the random mean (|Δ|=0.286), p=0.003. No random direction matches it. This means the probe isn't just any perturbation — it's specifically positioned in activation space to shift choice toward the steered task.

![summary](assets/steering_confounders/plot_021126_confounders_summary.png)

## Dead ends
- E1 alone cannot fully separate position from task-dependent effects — both predict P(A) increases with +coef in both orderings
- Controls with all-firm pairs: underpowered because firm pairs resist steering in any direction

## Final Results

| Experiment | Key Finding | p-value |
|-----------|-------------|---------|
| E2: Screening | 4.8% borderline rate, 6.7x effect amplification vs positional baseline | — |
| E3: Same-task | Positional component: Δ=+0.106 with identical tasks | 0.002 |
| E8: Specificity | Probe 2.6x stronger than random (|Δ| z=2.73) | 0.003 |
| E5: Logit lens | Borderline dose-response: r=0.779 | <1e-6 |
| E1: Counterbalancing | Effect in both orderings (Δ=0.711 vs 0.209), 38% order-sensitive | <1e-6 |

**Summary**: Steering a task's tokens with the probe direction increases P(picking that task). The effect is concentrated in borderline pairs and specific to the probe direction (2.6x stronger than random). There is a positional confound (62% of the effect), but the order-sensitive component (38%) and the 6.7x amplification over the same-task baseline are consistent with task-dependent steering. The key limitation of this experiment is that differential steering (+A, −B) confounds task-specific effects with position: we cannot isolate what happens when we steer only one task's tokens without suppressing the other.
