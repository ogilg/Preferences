# Fair BT vs Ridge Comparison

**Goal**: Run both BT and Ridge on the same task-level k-fold splits and report held-out pairwise accuracy for each. Determine whether the 9pp gap (BT 0.844 vs Ridge 0.758) is an artifact of data leakage.

**Result**: The gap was entirely an artifact. On fair task-level splits, Ridge beats BT by ~2.7pp (0.746 vs 0.719 on L31). Both are far below the Thurstonian ceiling (0.866).

## Setup

- **Data**: Gemma 3 27B, 3000 tasks, 23096 unique pairs, 113247 measurements
- **Layers**: 31, 43, 55 (focus on L31)
- **Folds**: k=5 task-level splits (600 tasks per fold)
- **HP selection**: Fold 0 only (per-layer sweep), fixed for folds 1-4
- **Metric**: Weighted pairwise accuracy on test pairs (both tasks in held-out fold)
- **Standardization**: Ridge uses StandardScaler fitted on train data; BT uses raw activations
- **No de-meaning**: Raw comparison

## Baseline

Previous (unfair) comparison:

| Method | Accuracy | Note |
|--------|----------|------|
| Ridge  | 0.758    | 5-fold CV, held-out tasks genuinely unseen |
| BT     | 0.844    | Training accuracy after lambda sweep on pair splits (leakage) |

## Iteration 1: Fair k-fold comparison

**Approach**: Task-level 5-fold CV where both methods are evaluated on the exact same held-out test pairs per fold. Fixed a standardization bug where Ridge weights trained on scaled features were being applied to unscaled test activations.

### Per-fold results (Layer 31)

| Fold | BT acc | Ridge acc | Thurstonian acc | Test pairs | Test measurements |
|------|--------|-----------|-----------------|------------|-------------------|
| 0 | 0.7167 | 0.7523 | 0.8785 | 933 | 4578 |
| 1 | 0.7288 | 0.7705 | 0.8878 | 965 | 4724 |
| 2 | 0.7243 | 0.7369 | 0.8724 | 920 | 4523 |
| 3 | 0.7212 | 0.7308 | 0.8550 | 902 | 4447 |
| 4 | 0.7057 | 0.7401 | 0.8384 | 916 | 4475 |
| **Mean** | **0.7193** | **0.7461** | **0.8664** | | |
| **Std** | **0.0079** | **0.0141** | **0.0176** | | |

Ridge beats BT on every fold for L31. ~900-960 test pairs per fold provides adequate statistical power.

### All layers summary

| Layer | BT (mean ± std) | Ridge (mean ± std) | Thurstonian (mean ± std) |
|-------|-----------------|--------------------|-----------------------|
| 31 | 0.719 ± 0.008 | 0.746 ± 0.014 | 0.866 ± 0.018 |
| 43 | 0.700 ± 0.025 | 0.733 ± 0.018 | 0.866 ± 0.018 |
| 55 | 0.702 ± 0.027 | 0.732 ± 0.022 | 0.866 ± 0.018 |

Ridge beats BT consistently across all layers. Layer 31 is best for both methods.

### Hyperparameters (selected on fold 0)

| Layer | BT λ | Ridge α |
|-------|------|---------|
| 31    | 10   | 2154    |
| 43    | 100  | 2154    |
| 55    | 1000 | 2154    |

BT needs progressively more regularization at later layers.

![Per-fold comparison (L31)](assets/bt_fair_comparison/plot_021126_bt_ridge_per_fold_L31.png)

![Across layers](assets/bt_fair_comparison/plot_021126_bt_ridge_across_layers.png)

## Dead ends
- Initial run had a per-layer HP bug: `best_lambda` was a single scalar overwritten in the layer loop, so folds 1-4 used layer 55's lambda for all layers. Fixed with per-layer dicts.
- Initial run also had a standardization bug: Ridge weights from scaled feature space applied to unscaled test activations. Fixed by fitting StandardScaler on train data and transforming all activations before Ridge evaluation.

## Final results

| Method | Unfair (old) | Fair (new) | Difference |
|--------|-------------|------------|------------|
| BT     | 0.844       | 0.719      | -0.125 (was train acc with leakage) |
| Ridge  | 0.758       | 0.746      | -0.012 (was already fairly evaluated) |

| Metric    | BT    | Ridge | Gap   |
|-----------|-------|-------|-------|
| L31 mean  | 0.719 | 0.746 | +2.7pp Ridge |
| L43 mean  | 0.700 | 0.733 | +3.4pp Ridge |
| L55 mean  | 0.702 | 0.732 | +2.9pp Ridge |

**Key insight**: The original 9pp BT advantage was entirely an artifact of data leakage. BT's reported accuracy was inflated because its lambda sweep split *pairs* not *tasks*, allowing task activations to appear in both train and validation. On fair task-level splits, the ranking reverses: Ridge beats BT by ~3pp across all layers. Ridge's accuracy barely changed (0.758 → 0.746), confirming the original Ridge evaluation was already roughly fair. BT's accuracy dropped dramatically (0.844 → 0.719), confirming the leakage hypothesis.

Both methods are far below the Thurstonian ceiling (0.866), suggesting significant room for improvement — but neither method has a meaningful edge in capturing preference information from activations.
