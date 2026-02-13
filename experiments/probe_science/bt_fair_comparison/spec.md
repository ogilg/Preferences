# Fair BT vs Ridge Comparison

## Problem

The current BT vs Ridge comparison is unfair. Ridge pairwise accuracy (0.758) comes from 5-fold CV where held-out *tasks* are genuinely unseen. BT pairwise accuracy (0.844) is training accuracy — the lambda sweep splits *pairs* not *tasks*, so a task's activations appear in both train and val. After lambda selection, BT retrains on all pairs. The 9pp gap may be an artifact of this leakage.

## Goal

Run both methods on the same task-level k-fold splits and report held-out pairwise accuracy for each. This gives a fair comparison of how well each method predicts preferences for unseen tasks.

## Success Criteria

A table like:

| Method | Fold-mean held-out pairwise accuracy | Std |
|--------|--------------------------------------|-----|
| Ridge  | ?                                    | ?   |
| BT     | ?                                    | ?   |

Plus a per-fold breakdown and a plot showing both methods across folds.

## Method

### Task-level k-fold splits

Split the 3000 tasks into k=5 folds (same folds for both methods). For each fold:

- **Train tasks**: tasks in the other 4 folds
- **Test tasks**: tasks in this fold

### Hyperparameter selection (once, on fold 0 only)

Run the regularization sweep on fold 0's train split for both methods, then fix those hyperparameters for all 5 folds:

- **Ridge**: alpha sweep via `train_and_evaluate` on fold-0 train tasks. Fix best α for folds 1–4.
- **BT**: lambda sweep via `train_bt` on fold-0 train pairs. Fix best λ for folds 1–4.

This avoids 5× sweep cost while still selecting hyperparameters on held-out data (fold 0's internal validation).

### Ridge evaluation (on each fold)

1. Train Ridge at the fixed α on train-task activations → Thurstonian μ (`train_at_alpha` with `standardize=True`)
2. Predict μ for test tasks
3. For all pairs where *both* tasks are in the test set, compute pairwise accuracy: does the higher-predicted-μ task match the empirical winner?

### BT evaluation (on each fold)

1. Filter pairs to only those where both tasks are in the train set
2. Train BT at the fixed λ on all train-only pairs (`train_bt_fixed_lambda`)
3. For all pairs where *both* tasks are in the test set, compute pairwise accuracy using the learned weight vector

### Key constraint

Both methods must be evaluated on the **exact same test pairs** in each fold — pairs where both tasks are in the held-out fold. This is what makes the comparison fair.

## Data

- **Activations**: `activations/gemma_3_27b/activations_prompt_last.npz` (3000 tasks, layers 31/43/55)
- **Thurstonian scores**: `results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/` (thurstonian CSV)
- **Pairwise measurements**: same run_dir (measurements.yaml)
- **Topics** (for task-ID lookup): `src/analysis/topic_classification/output/topics_v2.json`

## Existing Infrastructure

Most of the machinery exists:

- **`PairwiseActivationData.split_by_groups`** (`src/probes/bradley_terry/data.py`): splits pairs by task group membership (both tasks must be in same partition). Currently used for HOO by topic. You'll need to create a "fold" grouping where each task maps to its fold index.

- **`train_bt`** (`src/probes/bradley_terry/training.py`): BT training with lambda sweep. Use on fold 0 to select best λ.

- **`train_bt_fixed_lambda`** (`src/probes/bradley_terry/training.py`): BT training at a fixed lambda. Use on folds 1–4.

- **`weighted_accuracy`** (`src/probes/bradley_terry/training.py`): computes pairwise accuracy from a weight vector.

- **`train_and_evaluate`** (`src/probes/core/linear_probe.py`): Ridge with alpha sweep + CV. Use on fold 0 to select best α.

- **`train_at_alpha`** (`src/probes/core/linear_probe.py`): Ridge at fixed alpha. Use on folds 1–4.

- **`pairwise_accuracy_from_scores`** (`src/probes/bradley_terry/training.py`): computes pairwise accuracy from scalar scores (useful for evaluating Ridge predictions as pairwise accuracy).

- **`hoo_bt.py`** (`src/probes/experiments/hoo_bt.py`): shows how to use `split_by_groups` + `train_bt` + `weighted_accuracy` together. Good reference for the BT evaluation loop.

## Implementation Notes

- Use `numpy.random.Generator` with a fixed seed to assign tasks to folds (shuffle task indices, assign fold = index % k).
- Hyperparameters are selected once on fold 0 and fixed for all folds. The pair-level split inside `train_bt` for fold 0's lambda sweep is fine — it's contained within training data and the test fold is completely unseen.
- Layer 31 is the primary layer (highest R² historically). Run all three layers but focus analysis on L31.
- No de-meaning — this is a raw comparison of the two probe methods.
- Report weighted pairwise accuracy (weighted by number of comparisons per pair) to be consistent with existing BT metrics.

## Fallbacks

- If the number of test pairs per fold is small (because both tasks must be in the same fold), report the total count and discuss statistical power. With 3000 tasks in 5 folds (600 per fold), there should be ~C(600,2) potential test pairs minus those never compared, so likely 500-2000 test pairs per fold — should be sufficient.
- If results are inconclusive (e.g., both methods within 1-2pp), that's a valid finding — report it and discuss what it means.
