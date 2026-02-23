# Random Direction Control

## Goal

Test whether the steering effects observed in the replication experiment are specific to the probe direction, or whether any unit vector in activation space shifts pairwise choices equally well.

This is the highest-priority follow-up from both the original experiment and the replication. If random directions produce comparable shifts, the effect is a generic consequence of perturbing activations at task token positions, not evidence that the probe encodes evaluative information.

## Design

Use the **exact same setup** as Phase 1 of the replication experiment, but replace the probe direction with random unit vectors. Everything else is identical: same 77 borderline pairs, same coefficient range, same conditions, same prompt, same model.

### Random directions

Generate N=5 random unit vectors in the L31 activation space (dim=5376). Each vector is sampled from a standard normal distribution and normalized to unit length. Use seeds [100, 101, 102, 103, 104] for reproducibility.

5 directions gives reasonable power: if the probe-specific effect is +9pp (replication Phase 1 diff_ab), and random directions produce 0pp on average, then 5 random directions with 77 pairs each gives >95% power to detect the difference at α=0.05 (the probe is tested once, random directions are averaged over 5 runs).

### Conditions

Run only the conditions that showed the clearest effects in Phase 1:

| Condition | What | Motivation |
|---|---|---|
| boost_a | +direction on task A's tokens | Main single-task condition |
| diff_ab | +direction on A, -direction on B | Strongest effect in original and replication |
| control | No steering (coef=0) | Baseline |

Skip boost_b, suppress_a, suppress_b, diff_ba — they are structurally redundant (as confirmed in the replication report).

### Coefficients

Use only the moderate coefficient that showed the clearest effect: **+2641** (5% of L31 activation norm). Skip +5282 which caused reversals.

Also run at **-2641** as a sign check — if the probe direction is special, reversing the random direction should have the same (null) effect, but reversing the probe direction should flip the sign.

So coefficients: **[-2641, 0, +2641]**

### Resamples

15 resamples per condition × ordering (same as replication).

### Trial budget

77 pairs × 2 orderings × (3 conditions × 3 coefficients) × 15 resamples × 5 random directions = **~311,850 trials**

That's a lot. At ~7 trials/sec ≈ 12.4 hours.

**Optimization:** Since we already have the control (coef=0) data from the replication, we can skip coef=0 for random directions. Also, run fewer resamples (10 instead of 15) since we're averaging over 5 directions anyway. This gives:

77 pairs × 2 orderings × (2 conditions × 2 nonzero coefficients) × 10 resamples × 5 directions = **~123,200 trials** ≈ 4.9 hours.

**Further optimization:** Run 3 random directions first. If all 3 show near-zero effects, stop. If any shows comparable effects to the probe, run all 5 for proper statistical comparison.

## Comparison with probe

Re-run the probe direction at the same reduced design (2 conditions, 2 coefficients, 10 resamples) as a within-experiment comparison. This avoids cross-experiment confounds (different model state, GPU temperature, etc.). Cost: 77 × 2 × 4 × 10 = 6,160 additional trials.

## Analysis plan

### Primary analysis

For each direction (probe + 5 random), compute:
1. Aggregate P(pick steered task) at coef=+2641 for boost_a and diff_ab
2. Per-pair slope (regression of P(steered) on coefficient)
3. Fraction of pairs with positive slopes

### Comparison

- **Probe vs random mean**: Is probe shift > mean random shift? (One-sample t-test, probe shift - mean(random shifts))
- **Probe rank**: Where does the probe's per-pair slope distribution rank among the 5 random directions? If probe is best of 6, that's p ≤ 1/6 = 0.17 (suggestive but not significant). If probe is clearly separated, that's stronger.
- **Effect size comparison**: Cohen's d for probe shifts vs random direction shifts
- **Distribution comparison**: KS test on per-pair slope distributions (probe vs pooled random)

### Success criterion

- If random directions produce shifts near zero (|mean| < 3pp) while probe produces ~9pp: **probe is specific**
- If random directions produce comparable shifts (~7-9pp): **effect is generic** (any perturbation at task tokens shifts choices)
- If random directions produce moderate shifts (~4-6pp): **partial specificity** (probe has above-generic effect but the generic component is large)

## Data dependencies

All data from the replication experiment is available on the pod:
- `experiments/steering/replication/results/pairs.json` — the 77 borderline pairs
- `experiments/steering/replication/results/screening.json` — baseline P(A) data
- `experiments/steering/replication/results/steering_phase1.json` — probe steering results for comparison
- `results/probes/gemma3_10k_heldout_std_raw/` — probe manifest (for activation dimensionality)

## Parameters

```yaml
model: gemma-3-27b (H100 80GB)
layer: 31
activation_dim: 5376
n_random_directions: 5 (3 initially, extend to 5 if needed)
random_seeds: [100, 101, 102, 103, 104]
coefficients: [-2641, 0, +2641]
conditions: [boost_a, diff_ab, control]
resamples: 10
pairs: 77 borderline pairs from replication Phase 1
temperature: 1.0
max_new_tokens: 8
prompt: same as replication
```

## Infrastructure

Uses the same steering infrastructure as the replication. The only new code needed is generating random unit vectors:

```python
rng = np.random.default_rng(seed)
direction = rng.standard_normal(activation_dim)
direction = direction / np.linalg.norm(direction)
```

No new hooks or model extensions needed — use the same `position_selective_steering` and `differential_steering` hooks, just with a different direction vector.
