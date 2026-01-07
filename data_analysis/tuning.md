# Thurstonian Hyperparameter Tuning

Sequential tuning of optimization parameters, varying one at a time.

## 1. mu_bounds

Varied: ±5, ±10, ±20, ±50, ±1000 (with log_sigma_bounds fixed at ±3)

Results (n=24 prompt templates):
- Rankings stable across configs (mean Spearman ρ = 0.978)
- Tighter bounds (±5) occasionally hit constraints
- Very wide bounds (±1000) slower convergence

**Choice: ±10**

## 2. log_sigma_bounds

Varied: ±1, ±2, ±3, ±5, ±10 (with mu_bounds fixed at ±10)

Results (n=24 prompt templates):
- Rankings stable across configs (mean Spearman ρ = 0.981)
- Tighter bounds converge faster
- ±2 has slightly higher correlations with other configs

**Choice: ±2**

## 3. sigma_init

Varied: 0.1, 0.5, 1.0, 2.0, 5.0 (with mu_bounds=±10, log_sigma_bounds=±2)

Results (n=24 prompt templates):
- Rankings fairly stable (mean Spearman ρ = 0.951)
- Higher init values converge faster in median but with wider variance
- σ_init=1.0 has slightly better correlations and converges well

**Choice: 1.0**

## Final Parameters

- mu_bounds: (-10, 10)
- log_sigma_bounds: (-2, 2)
- sigma_init: 1.0
