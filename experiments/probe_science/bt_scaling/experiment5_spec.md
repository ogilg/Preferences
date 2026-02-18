# Experiment 5: Random vs AL Pair Selection — Is It Only at the Margin?

## Motivation

Experiment 4 found that adding 2K BT-selected pairs to 23.5K existing pairs *hurts* accuracy (-2.0pp BT, -0.4pp Ridge), while 2K random pairs *help* (+0.9pp BT, +1.3pp Ridge). But the existing 23.5K pairs were collected via Thurstonian AL. Two open questions:

1. **Is random better than Thurstonian AL too?** Experiment 4 tested BT uncertainty selection. The Thurstonian AL strategy used to collect the original data is different (12-22% pair overlap with BT, per Experiment 2). Maybe Thurstonian AL is fine and only BT selection is bad.

2. **Is this only true at the margin?** After 23.5K AL pairs, the probe is already well-trained — maybe AL pairs are valuable early (when coverage is sparse) but random catches up or surpasses at the margin. If so, the Experiment 4 result is less concerning.

## Design

We have 117K raw measurements (23.5K unique pairs × ~5 repeats), collected over 9 AL iterations. We can't change which pairs were measured, but we can **simulate** random vs AL by subsampling pairs at different data sizes and comparing accuracy.

### Experiment 5a: Random subsample vs AL-order subsample

For each data fraction f in {0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0}:

- **AL-order**: Take the first f × 23,500 pairs in the order they were collected (iterations 1, 2, ...). This is what AL actually produced at that point in time.
- **Random**: Take f × 23,500 pairs sampled uniformly at random from all 23.5K.

For each subsample, train BT+StandardScaler (λ=0.193) and Ridge+Thurstonian (α=1374), evaluate on 5-fold held-out pairwise accuracy.

This tests: did AL-ordered pairs give better probes than random subsets of the same size? If AL helped, the AL-order curve should be above the random curve, especially early.

**Requires**: Reconstructing iteration order from the measurements file. The AL ran 9 iterations: iteration 1 had ~7,500 pairs (d-regular init, degree 5), iterations 2-9 had 2,000 pairs each. Total = 7,500 + 8 × 2,000 = 23,500.

### Experiment 5b: Marginal value at different data sizes

Take the first K pairs (AL-ordered) as a "base" dataset, then add 2,000 pairs from two sources:
- **AL-next**: The next 2,000 pairs from the AL sequence
- **Random**: 2,000 random pairs from the remaining pool

Test at K = {5000, 10000, 15000, 20000} (where available from the 9-iteration sequence).

This directly tests whether the Experiment 4 pattern (random > AL at the margin) holds at all data sizes or only when the base is large.

## Data

Same as Experiments 1-4:
- Measurements: `scripts/active_learning_calibration/measurements_fast.json`
- Activations: `activations/gemma_3_27b/activations_prompt_last.npz`, layer 31
- 3,000 tasks, 23,500 unique pairs

Need to reconstruct iteration order. The measurements JSON has raw comparisons but may not have iteration labels. If not available, we can reconstruct from the AL config: iteration 1 = d-regular graph (degree 5, seed from config), subsequent iterations selected by Thurstonian uncertainty.

## Evaluation

Same 5-fold task-level CV, same folds (seed=42) as Experiments 1-4. Report mean ± std across folds. 3 random seeds for the random subsampling conditions.

## Expected outcome

- If AL helps early but not at the margin: AL-order curve above random at small f, converging or crossing at large f. This would explain Experiment 4 (marginal AL pairs are low-value) while validating the original AL strategy.
- If random is always as good as AL: The AL strategy never provided value over random sampling, and the d-regular initialization was doing all the work.
- If AL is always better: Experiment 4's negative result is specific to BT uncertainty selection and doesn't generalize.

## Cost

Zero API calls. All subsampling from existing data.
