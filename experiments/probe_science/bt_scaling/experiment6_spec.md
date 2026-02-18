# Experiment 6: Task Scaling — Does More Tasks Help?

## Motivation

Ridge probes have 5,376 features but only ~2,400 training tasks (after CV split). This is heavily overparameterized. Prior experiments scaled *pairs* but that primarily improves Thurstonian score quality, not the number of regression training points. Here we directly test whether more tasks improves probe accuracy.

## Design

Subsample tasks at fractions {0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0} of 3,000. For each fraction:

1. Sample N tasks uniformly at random
2. Keep all pairs and measurements between those tasks
3. Fit Thurstonian scores on those measurements
4. Train Ridge (with per-fraction alpha sweep) and BT+scaled (with per-fraction lambda sweep) on the subsampled tasks
5. Evaluate on all remaining tasks (3,000 − N) using their pairwise comparisons

Train/test split is over tasks, not pairs. Test set is always large regardless of training fraction.

**Important:** Sweep regularization at each fraction — don't reuse full-data hyperparameters.

3 random seeds. Report mean ± std across seeds.

## Data

Same as Experiments 1–5.

## Cost

Zero API calls.
