# EOT Steering Running Log

## Setup
- Branch: research-loop/eot_steering
- GPU: 1x H100 80GB
- All data files present locally

## Pilot Run (10 pairs, 3 conditions, 2 resamples = 120 trials)
- Fixed dimension mismatch: probe shape (5377,) includes intercept, need [:-1] for weights (5376,)
- Fixed stratified sampling to respect n_pairs parameter
- Batched resamples with generate_with_hook_n: 149.5s -> 76.8s (2x speedup)
- Parse rate: 100% at control and +0.03, 95% at -0.03
- Pipeline validated, results noisy due to small sample

## Full Run
- 500 pairs x 7 conditions x 2 orderings x 5 resamples = 35,000 trials
- Completed in 163.2 min (9792s), ~0.28s/trial effective
- Parse rates: 93.9-97.3% (all >90%), positive multipliers have slightly lower parse rates

## Results Summary
- **Null result on task preference steering**: P(high-mu) flat at ~0.71 across all conditions
- Spearman r = 0.036, p = 0.94 (no dose-response)
- Max steering effect: 0.7pp (CI includes zero)
- No ordering bias shifts from steering
- 1/4 success criteria passed (only parse rate)

### Notable observations
- Strong task preference signal: P(high-mu) = 0.717 at control
- Ordering bias ≈ +0.43 explained entirely by task preference: P(A|AB) + P(A|BA) ≈ 1.0
- Subtle positional trend: both P(A|AB) and P(A|BA) increase monotonically with multiplier
  - This suggests steering shifts position-A salience but not task-quality evaluation
  - Effect is symmetric: ~4pp shift from lowest to highest multiplier, cancels out in P(high-mu)
