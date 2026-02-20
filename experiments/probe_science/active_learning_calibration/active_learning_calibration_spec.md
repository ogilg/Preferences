# Active Learning Calibration

## Problem

We're planning a 10K-task preference measurement run. The active learning algorithm has parameters (`initial_degree`, `batch_size`, `p_threshold`, `q_threshold`, `max_iterations`) that were set to reasonable defaults for the 3K run but haven't been tuned. Before spending on a large run, we should understand how these settings affect Thurstonian utility quality and downstream probe accuracy.

We already have a 3K run with 117K comparisons across 9 iterations. We can subsample this data to simulate different parameter regimes — for free.

## Goal

Determine how many comparisons per task and how many active learning iterations are needed for stable utilities and good probe accuracy, so we can set parameters for the 10K run with confidence.

## Success Criteria

1. A curve showing probe pairwise accuracy (held-out) vs comparisons-per-task, with a clear "diminishing returns" point
2. A curve showing Thurstonian utility stability (rank correlation with full-data utilities) vs iterations
3. A recommendation for the 10K run: target comparisons-per-task and max_iterations

## Method

All analyses use the existing 3K run data. No API calls needed.

### Analysis 1: Iteration truncation

For N = 1, 2, ..., 9 (all completed iterations):
1. Take only the pairs queried in iterations 1..N
2. Refit Thurstonian utilities on those pairs
3. Compute Spearman rank correlation against full-data (all 9 iterations) utilities
4. Train Ridge probe (5-fold task-level CV) on the truncated utilities
5. Evaluate held-out pairwise accuracy on all test-fold pairs (using full pairwise data for evaluation)

This shows how quickly utilities and probe accuracy converge with more active learning iterations.

### Analysis 2: Random subsampling of comparisons

For fractions f = 0.1, 0.2, ..., 1.0 of total comparisons:
1. Randomly subsample f × 117K comparisons (preserving pair identity — sample pairs, not individual comparisons)
2. Refit Thurstonian utilities
3. Compute rank correlation against full-data utilities
4. Train Ridge probe (5-fold CV) and evaluate held-out pairwise accuracy
5. Repeat 5 times with different random seeds, report mean ± std

This shows the relationship between total comparison budget and quality, independent of the active learning selection strategy.

### Analysis 3: p/q threshold sensitivity

Using the stored per-iteration Thurstonian fits, simulate the pair selection step with different (p, q) combinations:
- p_threshold ∈ {0.1, 0.2, 0.3, 0.5}
- q_threshold ∈ {0.1, 0.2, 0.3, 0.5}

For each (p, q), replay the active learning selection from iteration 2 onward (iteration 1 is always random). Count how many unique tasks get queried and the degree distribution after a fixed number of iterations. Report the Gini coefficient of the degree distribution — lower is more balanced.

This doesn't require refitting (just replaying selection logic), so it's very fast.

## Data

- **Pairwise measurements**: `results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/`
- **Activations**: `activations/gemma_3_27b/activations_prompt_last.npz`
- **Per-iteration pair logs**: stored in the active learning output within the run_dir (check `active_learning.yaml` or the iteration logs for which pairs were queried in each iteration)

## Existing Infrastructure

- `src/fitting/thurstonian_fitting/thurstonian.py` — `fit_thurstonian()` for refitting on subsampled data
- `src/probes/core/linear_probe.py` — Ridge probe training with CV
- `src/probes/bradley_terry/training.py` — `pairwise_accuracy_from_scores()` for evaluation
- `src/fitting/thurstonian_fitting/active_learning.py` — `select_next_pairs()` for replaying selection with different thresholds
- Existing probe training configs in `configs/probes/` for reference

## Implementation Notes

- Layer 31 only — this is a calibration exercise, not a full probe comparison.
- For Analysis 1, check how per-iteration pairs are stored. They may be in the active_learning output as iteration logs, or you may need to reconstruct from the measurements file + timestamps.
- For Analysis 2, subsample at the pair level (keep all comparisons for a selected pair, or drop the pair entirely) to avoid partial-information artifacts.
- For Analysis 3, you only need to replay `select_next_pairs()` — no model fitting required, just the selection logic with different threshold parameters applied to the stored per-iteration Thurstonian fits.

## Fallbacks

- If per-iteration pair logs aren't available, reconstruct iteration membership from the measurements file (pairs are logged in order of querying).
- If Analysis 3 is hard to implement cleanly (requires extracting intermediate Thurstonian fits), skip it — Analyses 1 and 2 are the most important for setting the comparison budget.
