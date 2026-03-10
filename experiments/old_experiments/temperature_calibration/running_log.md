# Temperature Calibration — Running Log

## Session start: 2026-02-20

### Setup
- Branch: research-loop/temperature_calibration (already on it)
- Configs exist at configs/measurement/temperature_sweep/temp_{03,05,07,10,13}.yaml
- Analysis script exists at scripts/temperature_sweep/analyze.py
- Assets dir: experiments/temperature_calibration/assets/
- .env copied from Preferences parent project

### Data availability
- Gemma 3 27B activations (activations_prompt_last.npz) NOT present locally (gitignored)
- 4K eval run measurements available at results/experiments/gemma3_4k_pre_task/...
- Temperature sweep results dir does not exist yet

### Measurements completed

All 5 temperatures measured with fresh API calls (no cache contamination):
- temp_03: 6142 unique-pair measurements, 1250 pairs ✓ (0 cache hits)
- temp_05: 6212 measurements, 1250 pairs ✓ (0 cache hits, re-run after cache contamination)
- temp_07: 6094 measurements, 1250 pairs ✓ (0 cache hits, re-run after cache contamination)
- temp_10: 6143 measurements, 1250 pairs ✓ (0 cache hits)
- temp_13: 6236 measurements, 1250 pairs ✓ (0 cache hits)

Note: cache contamination issue — cache key does not include temperature, so earlier temp runs would have been reused for later temps. Fixed by clearing cache between each run. Contaminated temp_05/07 results archived in temp_05_contaminated/temp_07_contaminated.

### Activations
- Gemma 3 27B activations NOT available locally (gitignored, on RunPod)
- Used Gemma 2 27B base activations as proxy (40% of 500 sweep tasks, 100% of 4K eval tasks)
- Layer 32 (gemma2) ≈ Layer 31 (gemma3) in terms of relative depth

### Analysis results (2026-02-20)

Summary table:
| Temp | Consistency | Entropy | NLL    | Probe R² (G2 proxy) |
|------|------------|---------|--------|---------------------|
| 0.3  | 0.9796     | 0.0473  | 780.72 | 0.3389 |
| 0.5  | 0.9768     | 0.0494  | 943.14 | 0.3474 |
| 0.7  | 0.9808     | 0.0377  | 991.30 | 0.3099 |
| 1.0  | 0.9749     | 0.0529  | 807.83 | 0.3367 |
| 1.3  | 0.9756     | 0.0526  | 1275.90| 0.3344 |

Key insight: T=0.3 has very high max sigma (7.39) vs T=0.7 (1.42) — extremely few reversals at T=0.3 lead to unconstrained utility estimates for extreme tasks.

Ranking stability: Very high (0.846-0.876) across all temperature pairs — temperature mainly affects scale/noise, not ordering.

Probe R² (gemma-2 proxy): T=0.5 peaks (0.347), T=0.7 lowest (0.310). Interpretation unclear due to proxy model mismatch.

