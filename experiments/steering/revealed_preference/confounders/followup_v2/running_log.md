# Followup v2: Running Log

## Setup
- Workspace: `experiments/steering/revealed_preference/confounders/followup_v2/`
- GPU: H100 80GB
- Model: gemma-3-27b (HuggingFace)
- Probe: ridge_L31 (layer 31, 5376-dim)
- Thurstonian mu file: 3000 tasks with mu ∈ [-10, 10]

## Step 1: Pair Construction

Constructed 110 utility-matched pairs from 3000 tasks with Thurstonian mu values.

| Δmu bin | N pairs | Mean Δmu | Min  | Max   |
|---------|---------|----------|------|-------|
| 0-1     | 30      | 0.44     | 0.01 | 0.95  |
| 1-2     | 20      | 1.47     | 1.11 | 1.94  |
| 2-3     | 20      | 2.51     | 2.05 | 2.94  |
| 3-5     | 20      | 3.90     | 3.11 | 4.86  |
| 5-20    | 20      | 9.81     | 5.97 | 15.86 |

All 220 tasks are unique (no reuse). Higher-mu task is always placed as A.

## Step 2: Pilot experiment

Ran all 4 conditions on small subsets. Pipeline validates:
- Probe differential: 5 pairs, 3 coefs, 5 resamples — works, 21% null rate (refusals from adversarial tasks)
- Same-task: 10 tasks — works, 10% null
- Header-only: 3 borderline pairs — works, 0% null
- Random: 3 borderline pairs × 20 dirs — works, 0% null

Refusals are pair-specific (pair 4 = 100% refused, adversarial content), not coefficient-dependent.
Borderline pairs (Δmu 0-1 bin) have 0% refusal rate.

## Step 3: Full-scale experiment

### Probe differential (16,500 trials, ~40 min)

| Ordering | coef=-3000 | coef=-1500 | coef=0 | coef=+1500 | coef=+3000 |
|----------|-----------|-----------|--------|-----------|-----------|
| Original | 0.731 | 0.762 | 0.811 | 0.855 | 0.879 |
| Swapped  | 0.240 | 0.281 | 0.338 | 0.378 | 0.436 |

Both orderings show positive slope (toward position A). Quick decomposition:
- slope_orig ≈ 2.47e-5, slope_swap ≈ 3.27e-5
- Position component = (orig+swap)/2 = 2.87e-5 (dominant)
- Content component = (orig-swap)/2 = -0.40e-5 (small negative)

### Same-task (2,250 trials, ~5 min)

| coef | P(A) | N |
|------|------|---|
| -3000 | 0.775 | 377 |
| -1500 | 0.836 | 371 |
| 0     | 0.853 | 361 |
| +1500 | 0.858 | 359 |
| +3000 | 0.864 | 361 |

Strong baseline position bias (0.853). Steering adds small Δ=+0.089.

### Header-only (4,500 trials, ~10 min)

| Ordering | coef=-3000 | coef=-1500 | coef=0 | coef=+1500 | coef=+3000 |
|----------|-----------|-----------|--------|-----------|-----------|
| Original | 0.802 | 0.790 | 0.775 | 0.757 | 0.760 |
| Swapped  | 0.521 | 0.483 | 0.469 | 0.443 | 0.448 |

Slope is NEGATIVE in both orderings! Opposite to full differential. Header-only steering pushes
AWAY from position A. Meaning: the task content tokens drive the positive effect, not the headers.

### Random directions (running, ~90 min)

### Key intermediate finding: pair categorization

Most pairs are firmly decided despite Δmu matching:

| Category | N | % | Description |
|----------|---|---|-------------|
| Content-decided | 62 | 56% | One task always wins regardless of position |
| Position A-decided | 17 | 15% | Position A always wins |
| Position B-decided | 3 | 3% | Position B always wins |
| Variable | 24 | 22% | Show actual variability |
| Refused | 4 | 4% | Model refuses pair |

Variable pairs by Δmu bin: 0-1: 7, 1-2: 3, 2-3: 11, 3-5: 3, 5-20: 0

The Δmu 2-3 bin has the MOST variable pairs (11/20), not the 0-1 bin (7/30).
Thurstonian mu matching does NOT guarantee pairwise borderline status.

### Variable pairs decomposition (N=24)

| Ordering | P(A) at -3000 | P(A) at +3000 | Δ | Slope |
|----------|---------------|---------------|------|-------|
| Original | 0.458 | 0.857 | +0.400 | 7.37e-05 |
| Swapped  | 0.211 | 0.660 | +0.449 | 7.77e-05 |

Position component: 7.97e-05 (97%)
Content component: -2.45e-06 (3%)

Both orderings have nearly identical slopes → effect is overwhelmingly positional.

### Random directions (45,000 trials, ~110 min)

Random (N=20 dirs):
- Mean Δ: +0.008, Std: 0.068
- Mean |Δ|: 0.057
- Range: [-0.119, +0.127]

Probe vs random:
- Probe Δ (borderline, original): +0.135
- Z-score: 2.04
- Rank p: 0/20 = 0.000
- Random slopes not significantly ≠ 0 (t=0.61, p=0.55)

## Step 4: Final Analysis

Full analysis saved to `analyze_results.py` output. Key tables in main log.
Plots saved to `docs/logs/assets/steering_confounders/`:
- `plot_021126_v2_summary.png` — 6-panel overview
- `plot_021126_v2_condition_comparison.png` — all conditions on borderline pairs
- `plot_021126_v2_random_comparison.png` — probe vs 20 random directions

## Summary

The H2 differential steering effect is predominantly positional (~90-97% of total effect).
The probe direction is specific (z=2.04 vs random) but specific at encoding position, not evaluative content.
The model's natural preferences ARE content-driven (P(higher-mu)=0.74 at baseline), but the probe direction doesn't capture this mechanism in the steering context.

