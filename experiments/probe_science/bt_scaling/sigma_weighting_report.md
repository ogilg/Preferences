# Sigma-weighted Ridge probes

## Question

Thurstonian fitting produces per-task sigma (posterior uncertainty). Does weighting Ridge regression by inverse sigma improve out-of-sample probe performance?

## Setup

- **Train**: 3K tasks (gemma3_3k_run2), 0 overlap with val
- **Val**: 4K held-out tasks (gemma3_4k_pre_task)
- **Activations**: gemma-3-27b, prompt_last token, layers [15, 31, 37, 43, 49, 55]
- **Alpha selection**: swept 50 alphas on val R² directly (no CV)
- **Weighting modes**: none, 1/σ² (inverse_variance), 1/σ (inverse_sigma)

Train sigma stats: mean=1.06, median=0.54, min=0.23, max=7.39.

## Results

| Layer | Mode | best_α | train_R² | val_R² | val_r | val_acc |
|-------|------|--------|----------|--------|-------|---------|
| 15 | none | 1.5e+03 | 0.8390 | 0.4291 | 0.7056 | 0.6884 |
| 15 | 1/σ² | 8.3e+03 | 0.7967 | 0.4263 | 0.6973 | 0.6862 |
| 15 | 1/σ | 2.7e+03 | 0.8238 | 0.4371 | 0.7046 | 0.6895 |
| 31 | none | 1.5e+03 | 0.9449 | 0.6216 | 0.8424 | 0.7551 |
| 31 | 1/σ² | 4.7e+03 | 0.9266 | 0.6240 | 0.8376 | 0.7515 |
| 31 | 1/σ | 2.7e+03 | 0.9343 | 0.6275 | 0.8420 | 0.7543 |
| 37 | none | 1.5e+03 | 0.9396 | 0.6018 | 0.8311 | 0.7447 |
| 37 | 1/σ² | 6.3e+03 | 0.9164 | 0.6029 | 0.8252 | 0.7423 |
| 37 | 1/σ | 2.7e+03 | 0.9284 | 0.6061 | 0.8289 | 0.7437 |
| 43 | none | 1.2e+03 | 0.9467 | 0.5922 | 0.8241 | 0.7421 |
| 43 | 1/σ² | 4.7e+03 | 0.9198 | 0.5932 | 0.8178 | 0.7379 |
| 43 | 1/σ | 2.0e+03 | 0.9339 | 0.5970 | 0.8223 | 0.7392 |
| 49 | none | 1.5e+03 | 0.9397 | 0.5868 | 0.8178 | 0.7397 |
| 49 | 1/σ² | 6.3e+03 | 0.9145 | 0.5874 | 0.8113 | 0.7355 |
| 49 | 1/σ | 2.7e+03 | 0.9275 | 0.5914 | 0.8153 | 0.7371 |
| 55 | none | 1.5e+03 | 0.9409 | 0.5843 | 0.8168 | 0.7383 |
| 55 | 1/σ² | 4.7e+03 | 0.9189 | 0.5879 | 0.8112 | 0.7340 |
| 55 | 1/σ | 2.7e+03 | 0.9281 | 0.5913 | 0.8156 | 0.7368 |

**Averages across layers:**

| Mode | train_R² | val_R² | val_r | val_acc |
|------|----------|--------|-------|---------|
| none | 0.9251 | 0.5693 | 0.8063 | 0.7347 |
| 1/σ² | 0.8988 | 0.5703 | 0.8001 | 0.7312 |
| 1/σ | 0.9126 | 0.5751 | 0.8048 | 0.7334 |

## Findings

- **1/σ² hurts.** Aggressive downweighting of uncertain tasks reduces correlation and pairwise accuracy across all layers. It also forces much higher regularization (alpha ~4-6x larger).
- **1/σ is marginal.** Small val R² gain (+0.6pp average) but slightly worse on val_r and pairwise accuracy vs unweighted. Not consistently better on any single metric.
- **Not worth the complexity.** The effect is small enough that it's noise-level. Most tasks have similar sigma (median 0.54 vs mean 1.06 — a few high-sigma outliers drive the mean up), so weighting changes little.

## Decision

Keep `sigma_weighting` config flag for completeness but default to `"none"`. Not pursuing further.
