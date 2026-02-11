# Fair BT vs Ridge Comparison — Report

## Summary

The previously reported 9pp BT advantage (0.844 vs 0.758) was an artifact of data leakage in the BT evaluation. On fair task-level k-fold splits, **Ridge outperforms BT by ~3pp** across all layers, reversing the original ranking.

## Method

Both BT and Ridge were evaluated on identical held-out test pairs using task-level 5-fold cross-validation:

- **3000 tasks** split into 5 folds (600 tasks each), same folds for both methods
- **Test pairs**: only pairs where both tasks are in the held-out fold (~920 pairs/fold)
- **Hyperparameters** selected once on fold 0, fixed for folds 1-4
- **Metric**: weighted pairwise accuracy on test pairs (identical for both methods)

The key difference from the original evaluation: BT's lambda sweep previously split *pairs* not *tasks*, so a task's activations could appear in both train and validation. This new setup ensures complete task-level separation.

## Results

### Layer 31 (primary)

| Fold | BT acc | Ridge acc | Thurstonian acc | Test pairs |
|------|--------|-----------|-----------------|------------|
| 0 | 0.717 | 0.752 | 0.879 | 933 |
| 1 | 0.729 | 0.771 | 0.888 | 965 |
| 2 | 0.724 | 0.737 | 0.872 | 920 |
| 3 | 0.721 | 0.731 | 0.855 | 902 |
| 4 | 0.706 | 0.740 | 0.838 | 916 |
| **Mean** | **0.719** | **0.746** | **0.866** | |
| **Std** | **0.008** | **0.014** | **0.018** | |

### All layers

| Layer | BT (mean ± std) | Ridge (mean ± std) | Gap |
|-------|------------------|--------------------|-----|
| 31 | 0.719 ± 0.008 | 0.746 ± 0.014 | +2.7pp Ridge |
| 43 | 0.700 ± 0.025 | 0.733 ± 0.018 | +3.4pp Ridge |
| 55 | 0.702 ± 0.027 | 0.732 ± 0.022 | +2.9pp Ridge |

![Per-fold comparison](assets/plot_021126_bt_ridge_per_fold_L31.png)

![Across layers](assets/plot_021126_bt_ridge_across_layers.png)

### Before vs after

| Method | Unfair (old) | Fair (new) | Change |
|--------|-------------|------------|--------|
| BT     | 0.844       | 0.719      | -12.5pp |
| Ridge  | 0.758       | 0.746      | -1.2pp  |

Ridge's number barely moved because the original 5-fold CV on tasks was already a fair evaluation. BT's number dropped by 12.5pp, confirming the leakage hypothesis.

## Hyperparameters

Selected on fold 0 via sweep:

| Layer | BT λ | Ridge α |
|-------|------|---------|
| 31    | 10   | 2154    |
| 43    | 100  | 2154    |
| 55    | 1000 | 2154    |

## Discussion

1. **The original comparison was unfair**: BT's 0.844 was essentially training accuracy. After removing leakage, BT performs ~12pp worse.

2. **Ridge is the better probe method**: Ridge consistently outperforms BT by ~3pp across all layers and folds. This makes sense — Ridge trains on Thurstonian utility scores (which aggregate information across all comparisons for each task), while BT trains on individual pairwise comparisons. Ridge benefits from the Thurstonian model's noise reduction.

3. **Both methods are far below the Thurstonian ceiling** (0.866): The gap between probe predictions and Thurstonian scores suggests that activations capture a limited fraction of the preference signal. The probes explain ~72-75% of pairwise outcomes, while the Thurstonian model explains ~87%.

4. **Statistical power is adequate**: ~900+ test pairs per fold, with consistent results across folds.

## Methodology notes

- Ridge uses StandardScaler fitted on training data; test activations are transformed with the same scaler before evaluation. BT uses raw activations (no standardization).
- No de-meaning — this is a raw comparison of the two probe methods.
- Thurstonian μ serves as a non-probe reference: it uses the actual fitted utility scores (not activations) and represents the information ceiling for pairwise prediction given the measurement noise.
