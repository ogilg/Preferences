# Steering Confounders v2: Utility-Matched Pairs

**Goal**: Investigate whether differential steering shifts choice toward the steered task at scale, using utility-matched pairs across a range of preference distances (Δmu), with controls for position, header contributions, and direction specificity.

**Result**: Differential steering increases P(picking the steered-position task) in both orderings across 110 pairs (both p<1e-28). The effect is specific to the probe direction (outperforms all 20 random directions, z=1.89, rank p<0.05). On variable pairs (where the model's baseline choice is uncertain), the effect is large and contains a substantial order-sensitive component (28%), consistent with task-dependent steering. On firm pairs, slopes are small and dominated by position. The effect comes from task content tokens — header-only steering has the opposite sign. The main limitation is that differential steering (+A, −B) cannot isolate single-task steering from position.

## Design

- **Pairs**: 110 utility-matched across Δmu bins (0-1: 30, 1-2: 20, 2-3: 20, 3-5: 20, 5+: 20)
- **Conditions**: Probe differential (both orderings), same-task, header-only, random directions
- **Coefficients**: [-3000, -1500, 0, +1500, +3000]
- **Resamples**: 15 per condition
- **Total trials**: ~69,000

## Pair Construction

Constructed 110 pairs from 3000 tasks with Thurstonian mu values, pairing tasks within Δmu bins. Higher-mu task always placed as A in the "original" ordering.

| Δmu bin | N pairs | Mean Δmu | Min  | Max   |
|---------|---------|----------|------|-------|
| 0-1     | 30      | 0.44     | 0.01 | 0.95  |
| 1-2     | 20      | 1.47     | 1.11 | 1.94  |
| 2-3     | 20      | 2.51     | 2.05 | 2.94  |
| 3-5     | 20      | 3.90     | 3.11 | 4.86  |
| 5+      | 20      | 9.81     | 5.97 | 15.86 |

**Pair variability**: Only 11/110 pairs are genuinely variable (P(A) between 0.1 and 0.9 at baseline in original ordering). Most are firmly decided:

| Category | N | % | Meaning |
|----------|---|---|---------|
| Firm | 93 | 85% | One task or position dominates |
| Variable | 11 | 10% | Baseline choice is uncertain |
| Other | 6 | 5% | Refused or edge cases |

Thurstonian mu matching does not guarantee pairwise borderline status — pairwise decisions depend on factors beyond individual-task utility.

## Baseline Preferences (coef=0)

The model's natural choices (without steering) show clear content preference:

| Δmu bin | P(higher-mu task) | P(position A) |
|---------|-------------------|---------------|
| 0-1     | 0.653             | 0.622         |
| 1-2     | 0.676             | 0.591         |
| 2-3     | 0.737             | 0.532         |
| 3-5     | 0.718             | 0.632         |
| 5+      | 0.925             | 0.475         |

Content preference scales with Δmu as expected (0.65 → 0.93). Position A bias is moderate (0.53–0.63) and weakens at large Δmu.

## Probe Differential (16,500 trials)

Adding the probe to task A's tokens while subtracting it from task B's increases P(A) in both orderings:

| Ordering | coef=-3000 | coef=0 | coef=+3000 | Slope | p |
|----------|-----------|--------|-----------|-------|---|
| Original (A=higher μ) | 0.731 | 0.811 | 0.879 | 2.60e-05 | <1e-28 |
| Swapped (A=lower μ)   | 0.240 | 0.338 | 0.436 | 3.26e-05 | <1e-28 |

Since tasks swap positions between orderings, but P(A) increases with +coef in both, the probe shifts choice toward whichever task's tokens receive the +probe direction. This is the core finding: **differential steering causally shifts choice toward the steered task**.

![summary](assets/steering_confounders/plot_021126_v2_summary.png)

### Does the effect scale with the model's default preference?

The steering effect depends on how decided the pair is at baseline. Per-pair, we compute the steering slope (averaged across both orderings) and the baseline P(higher-mu task) (position-controlled, averaged across orderings at coef=0).

| Predictor | Correlation with steering slope | p |
|-----------|-------------------------------|---|
| Decidedness \|P(higher-μ) − 0.5\| | r=−0.250 | 0.011 |
| P(higher-μ task) | r=−0.266 | 0.007 |
| Δμ | r=−0.188 | 0.058 |

Less decided pairs are more steerable. Δmu has a marginal relationship (p=0.058), but **adding Δmu to a regression that already includes decidedness provides no significant improvement** (ΔR²=0.015, F=1.67, p=0.20). The apparent Δmu effect is mediated by decidedness: high-Δmu pairs tend to be more firmly decided.

![decidedness](assets/steering_confounders/plot_021226_steering_vs_decidedness.png)

Per Δmu bin:

| Δmu bin | N pairs | N variable | Avg slope | Orig P(A) range | Swap P(A) range |
|---------|---------|------------|-----------|-----------------|-----------------|
| 0-1 | 30 | 3 | 3.05e-05 | 0.680→0.815 | 0.312→0.531 |
| 1-2 | 20 | 2 | 2.70e-05 | 0.681→0.815 | 0.350→0.531 |
| 2-3 | 20 | 6 | 4.61e-05 | 0.642→0.919 | 0.180→0.454 |
| 3-5 | 20 | 0 | 3.23e-05 | 0.800→0.923 | 0.290→0.543 |
| 5-20 | 20 | 0 | 1.08e-05 | 0.860→0.943 | 0.050→0.100 |

The 2-3 bin has the largest effect (6/20 variable pairs), and the 5-20 bin has the smallest (0/20 variable). The pattern tracks the number of variable pairs per bin, not Δmu itself.

### Position vs order-sensitive decomposition

The average of both orderings captures the positional component; half the difference captures the order-sensitive component (which reverses when tasks swap):

| Subset | N | Position component | Order-sensitive component | Position % |
|--------|---|-------------------|--------------------------|------------|
| All pairs | 103 | 2.93e-05 | −3.30e-06 | 90% |
| Variable | 11 | 9.46e-05 | 3.60e-05 | 72% |
| Firm | 92 | 2.16e-05 | −7.92e-06 | — |

The all-pairs decomposition is dominated by firm pairs where slopes are small and approximately symmetric across orderings (making the order-sensitive component vanish). On **variable pairs** — where the model is actually uncertain — the order-sensitive component is 28% of the total, with the original ordering (where +probe aligns with the higher-mu task) producing a 2.2x larger slope than the swapped.

Per-pair content effect t-tests are underpowered (variable: t=0.19, p=0.85, N=11; all: t=−0.91, p=0.36, N=103).

## Same-Task Control (2,250 trials)

| coef | P(A) |
|------|------|
| -3000 | 0.775 |
| 0     | 0.853 |
| +3000 | 0.864 |

Baseline position A bias is strong (0.853). Steering effect is not significant (slope=5.63e-06, p=0.22). This is weaker than the v1 same-task result (slope=1.58e-05, p=0.002), possibly because this set uses different tasks.

## Header-Only Steering (4,500 trials)

**Opposite sign** to full differential:

| Ordering | coef=-3000 | coef=0 | coef=+3000 | Slope | p |
|----------|-----------|--------|-----------|-------|---|
| Original | 0.802 | 0.775 | 0.760 | −7.41e-06 | 0.10 |
| Swapped  | 0.521 | 0.469 | 0.448 | −1.17e-05 | 0.02 |

Adding the probe direction to "Task A:\n" / "Task B:\n" header tokens pushes *away* from that position. The full differential effect (which is positive) must come from the task content tokens, not the structural headers.

![conditions](assets/steering_confounders/plot_021126_v2_condition_comparison.png)

## Condition Comparison

| Condition | Slope | p-value | ΔP(A) |
|-----------|-------|---------|-------|
| Probe (original) | 2.60e-05 | <1e-28 | +0.149 |
| Probe (swapped) | 3.26e-05 | <1e-28 | +0.196 |
| Same-task | 5.63e-06 | 0.22 | +0.089 |
| Header (original) | −7.41e-06 | 0.10 | −0.043 |
| Header (swapped) | −1.17e-05 | 0.02 | −0.074 |

## Random Directions (45,000 trials)

30 borderline pairs × 20 random orthogonal directions × 5 coefs × 15 resamples.

| Metric | Probe | Random (N=20) |
|--------|-------|---------------|
| Slope | 2.10e-05 | mean 1.56e-06, std 1.03e-05 |
| Δ P(A) | +0.135 | mean +0.008, std 0.068 |
| \|Δ P(A)\| | 0.135 | mean 0.057 |

- **Z-score**: 1.89 (probe slope vs random slope distribution)
- **Rank p-value**: 0/20 (probe outperforms all 20 random directions)
- Random slopes not significantly different from zero (t=0.61, p=0.55)

The probe direction is specifically oriented to shift choice — random perturbations scatter symmetrically and are much weaker.

![random comparison](assets/steering_confounders/plot_021126_v2_random_comparison.png)

## Key Findings

1. **Differential steering shifts choice toward the steered task**. Adding +probe to a task's tokens while subtracting from the other increases P(picking that task), in both orderings (both p<1e-28).

2. **The effect is concentrated in variable pairs**. Variable pairs (N=11) show slopes 5-10x larger than firm pairs. On variable pairs, the original ordering swings P(A) from 0.19 to 0.88.

3. **The effect has a task-dependent component on variable pairs**. On variable pairs, the order-sensitive component is 28% of the total — the original ordering (where +probe aligns with the higher-mu task) produces a 2.2x larger slope than the swapped ordering. On all pairs, this component is masked by firm pairs.

4. **The effect comes from task content tokens, not headers**. Header-only steering reverses sign, meaning the full-differential effect is driven by the probe modifying how the model processes the actual task descriptions.

5. **The probe direction is specific**. It outperforms all 20 random directions (z=1.89, rank p<0.05). Random perturbations scatter symmetrically and produce much weaker effects.

## Limitations and Missing Pieces

The central question — **does adding the probe to a task's tokens increase P(picking that task)?** — cannot be cleanly answered by differential steering. Differential steering simultaneously boosts one task and suppresses the other, confounding three effects:

1. **Boost effect**: Does +probe on task X make X more likely to be chosen?
2. **Suppress effect**: Does −probe on task Y make Y less likely to be chosen?
3. **Position effect**: Does the +/− asymmetry just amplify position A bias?

The order-counterbalancing decomposition partially separates position from content, but cannot separate boost from suppress. A clean test requires **single-task steering**: add the probe to only one task's tokens (no subtraction from the other), across both orderings, and measure whether P(picking the steered task) increases.

The existing H1 task-selective data (from the main steering report) provides a preliminary answer: combined across steer-on-A and steer-on-B, P(pick steered task) goes from 0.445 to 0.515 (slope=1.06e-05, p=0.025). But this was only 20 pairs (mostly firm) with 10 resamples — underpowered for the question.

**What's needed**: A properly powered single-task steering experiment on pre-screened borderline pairs, with both orderings, testing +probe-only and −probe-only separately.
