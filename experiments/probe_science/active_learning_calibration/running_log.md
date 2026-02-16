# Active Learning Calibration — Running Log

## Data exploration

- Run dir: `results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/`
- 117,435 total measurements, 23,500 unique pairs, 3000 tasks
- Config: `initial_degree=5`, `batch_size=2000`, `n_samples=5`, `convergence_threshold=0.995`
- 9 iterations, converged at rank correlation 0.995
- Iteration structure (reconstructed from measurement order):
  - Iter 0 (d-regular init): ~7500 unique pairs × 5 samples = ~37,500 measurements
  - Iters 1-8: ~2000 unique pairs each × 5 samples = ~10,000 measurements each
  - Total: 7500 + 8×2000 = 23,500 unique pairs ✓
- Almost all pairs have exactly 5 measurements (23,440 of 23,500)
- Rank correlations: [0.0, 0.958, 0.975, 0.984, 0.989, 0.992, 0.993, 0.994, 0.995]

## Analysis 1: Iteration truncation

| Iter | Measurements | Unique pairs | Comp/task | Rank corr | CV R² | Pairwise acc |
|------|-------------|-------------|-----------|-----------|-------|-------------|
| 1 | 37,500 | 7,505 | 12.5 | 0.888 | 0.640 | 0.724 |
| 2 | 47,500 | 9,508 | 15.8 | 0.926 | 0.707 | 0.730 |
| 3 | 57,500 | 11,508 | 19.2 | 0.950 | 0.753 | 0.730 |
| 4 | 67,500 | 13,509 | 22.5 | 0.966 | 0.788 | 0.736 |
| 5 | 77,500 | 15,512 | 25.8 | 0.977 | 0.807 | 0.736 |
| 6 | 87,500 | 17,516 | 29.2 | 0.984 | 0.826 | 0.737 |
| 7 | 97,500 | 19,535 | 32.5 | 0.991 | 0.844 | 0.736 |
| 8 | 107,500 | 21,540 | 35.8 | 0.995 | 0.857 | 0.738 |
| 9 | 117,435 | 23,500 | 39.1 | 1.000 | 0.864 | 0.740 |

Key observations:
- Rank correlation converges quickly: 0.888 after iter 1, 0.95 by iter 3
- CV R² keeps climbing steadily: 0.640 → 0.864 (no plateau)
- Pairwise accuracy plateaus early: 0.724 → 0.740 (only 1.6pp gain after iter 1)
- Pairwise accuracy is essentially flat after iter 2 (~0.730-0.740)

## Analysis 2: Random subsampling of comparisons

| Fraction | Pairs | Comp/task | Rank corr (mean±std) | PW acc (mean±std) |
|----------|-------|-----------|---------------------|-------------------|
| 0.1 | 2,350 | 3.9 | 0.427±0.011 | 0.697±0.004 |
| 0.2 | 4,700 | 7.8 | 0.710±0.014 | 0.717±0.005 |
| 0.3 | 7,050 | 11.8 | 0.865±0.008 | 0.725±0.005 |
| 0.4 | 9,400 | 15.7 | 0.932±0.002 | 0.730±0.002 |
| 0.5 | 11,750 | 19.6 | 0.962±0.001 | 0.735±0.004 |
| 0.6 | 14,100 | 23.5 | 0.977±0.001 | 0.737±0.003 |
| 0.7 | 16,450 | 27.4 | 0.985±0.001 | 0.738±0.001 |
| 0.8 | 18,800 | 31.3 | 0.991±0.000 | 0.738±0.002 |
| 0.9 | 21,150 | 35.3 | 0.996±0.000 | 0.738±0.001 |
| 1.0 | 23,500 | 39.1 | 1.000 | 0.741 |

Note: R² values from Analysis 2 are unreliable due to non-shuffled KFold with alphabetically sorted task IDs creating degenerate folds. Pairwise accuracy used KFold(shuffle=True) and is correct.

Key observations:
- Random subsampling shows same pairwise acc plateau (~0.725 at 30% pairs, 0.738 at 70%)
- Rank correlation is much worse at low fractions vs iteration truncation (0.43 vs 0.89 at similar comparisons-per-task), because active learning prioritizes informative pairs
- Pairwise accuracy converges similarly in both analyses — the active learning doesn't help much for probe accuracy

## Analysis 3: p/q threshold sensitivity

Skipped per spec fallback — requires replaying select_next_pairs() which iterates over C(3000,2) ≈ 4.5M pair combinations, making it computationally infeasible within session time.
