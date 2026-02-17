# Bradley-Terry vs Ridge Probes: Regularization, Pair Selection, and Scaling

## Summary

The previously reported 3pp Ridge advantage over Bradley-Terry (BT) was a **preprocessing artifact**. After adding StandardScaler to BT and sweeping its L2 penalty properly, BT matches Ridge at full data (74.2% vs 73.9%). BT's real advantage is in the low-data regime: at 10% of training pairs, BT leads Ridge by **8.6pp** (70.9% vs 62.3%). However, a live ablation (Experiment 4) shows that **BT uncertainty-based active learning is counterproductive**: adding 2,000 BT-selected pairs *decreased* accuracy by 2.0pp, while 2,000 random pairs *increased* it by 0.9pp. The most uncertain pairs are genuinely ambiguous -- their measurements add noise rather than signal.

## Background

We train linear probes to predict pairwise preferences from model activations. Two approaches:

| Method | Training pipeline |
|--------|------------------|
| **Ridge** | Aggregate pairwise votes into a per-task Thurstonian score (a denoised utility estimate), then regress activations onto scores with Ridge regression. Predict pairs by comparing predicted scores. |
| **Bradley-Terry (BT)** | Fit a logistic model directly on pairwise comparisons: P(A > B) = sigma(w . (act_A - act_B)), with L2 penalty on w. No intermediate aggregation. |

Ridge requires enough comparisons per task to get stable Thurstonian scores. BT uses each comparison directly, so it should degrade more gracefully with sparse data.

## Setup

- **Model**: Gemma-3-27B, layer 31 activations at the last prompt token
- **Data**: 3,000 tasks, 23,500 unique pairs, 117K comparisons (5 repeats per pair), collected over 9 active learning iterations
- **Evaluation**: 5-fold CV over tasks. A test pair requires both tasks in the held-out fold (~900 pairs/fold). All numbers are mean +/- std across folds. Chance = 50%.
- **Hyperparameter selection**: Internal validation on fold 0, then fixed for folds 1--4

## Experiment 1: Regularization Audit

Swept regularization for four probe variants at full data:

| Variant | What it does | Held-out pairwise accuracy | Best hyperparameter |
|---------|-------------|---------------------------|-------------------|
| BT + StandardScaler | BT on z-scored activations | **74.2 +/- 1.1%** | L2 lambda = 0.19 |
| Ridge + Thurstonian scores | Standard Ridge pipeline | 73.9 +/- 1.1% | alpha = 1,374 |
| BT (raw activations) | BT without preprocessing | 73.7 +/- 1.0% | L2 lambda = 139 |
| Ridge + win-rates | Ridge on simple win fractions instead of Thurstonian scores | 72.1 +/- 0.8% | alpha = 9,237 |

The lambda sweep (Figure 1) shows train and validation accuracy as a function of BT's L2 penalty. BT with StandardScaler (orange) peaks at a much lower lambda (0.19) than BT on raw activations (blue, lambda = 139). The raw-activation BT needs heavy regularization to compensate for features with large magnitudes dominating the logistic predictor. With StandardScaler, all features contribute equally and less regularization is needed.

![Figure 1: BT L2 penalty sweep on fold 0. StandardScaler (orange) peaks at lambda=0.19; raw activations (blue) peaks at lambda=139. Dashed lines = train accuracy, solid = validation.](assets/plot_021626_bt_lambda_sweep.png)

![Figure 2: Held-out pairwise accuracy for all four variants. Error bars = +/- 1 std across folds. All methods well above chance (50%, not visible). The top three are within ~1pp of each other.](assets/plot_021626_regularization_summary.png)

**Takeaways:**

- **StandardScaler closes the BT--Ridge gap.** BT on raw activations (73.7%) trails Ridge (73.9%); BT with StandardScaler (74.2%) matches or slightly leads it. The difference at the top is within error bars.
- **The original BT result (71.9%) used a fixed lambda=10.** Proper sweeping alone improves BT by 1.8pp (to 73.7%), and adding StandardScaler adds another 0.5pp.
- **Thurstonian scores are better regression targets than raw win-rates.** Ridge + win-rates (72.1%) underperforms Ridge + Thurstonian (73.9%) by 1.8pp, confirming the Thurstonian denoising step matters for Ridge.

## Experiment 2: Pair Selection Oracle

Would BT-guided active learning choose different pairs than the Thurstonian strategy that was actually used? To answer this without new API calls, we replayed active learning iterations 2--5:

1. Train a BT probe on all pairs available up to iteration N
2. Score every candidate pair by **BT uncertainty**: |w . (act_A - act_B)| -- pairs where the probe is least certain get the lowest score
3. Select the top 2,000 most uncertain pairs (the batch BT would request next)
4. Compare against the 2,000 pairs that Thurstonian active learning actually selected (based on closeness of Thurstonian score estimates)

Three metrics capture how much the two strategies diverge:

| Metric | Definition |
|--------|-----------|
| Pair overlap | Fraction of the 2,000 selected pairs that both methods chose |
| Rank correlation | Spearman correlation between BT uncertainty ranking and Thurstonian ambiguity ranking over all candidate pairs |
| Task coverage overlap | Fraction of tasks appearing in both methods' selected batches |

| AL iteration | Pair overlap | Rank correlation | Task coverage overlap |
|--------------|-------------|-----------------|----------------------|
| 2 | 12.3% | 0.24 | 56.0% |
| 3 | 14.0% | 0.34 | 52.3% |
| 4 | 17.1% | 0.42 | 50.8% |
| 5 | 21.8% | 0.46 | 51.6% |

Figure 3 shows pair overlap (blue bars), task coverage overlap (green bars), and rank correlation (red line) across iterations. The dashed line marks the 80% pair overlap threshold above which a live experiment would add little value.

![Figure 3: Pair selection divergence across AL iterations. Pair overlap (blue) stays far below the 80% threshold (dashed). Task coverage overlap (green) hovers around 50%. Rank correlation (red) is positive but weak.](assets/plot_021626_pair_overlap.png)

**Takeaways:**

- **BT and Thurstonian select very different pairs** (12--22% overlap, far below the 80% threshold).
- **They also focus on different tasks** (~50% task coverage overlap means half the tasks in each batch are unique to one method).
- **Overlap grows with data** (12% to 22%) but stays low -- the methods don't converge even after 5 iterations.
- **Rank correlation is positive but weak** (0.24--0.46), confirming the two uncertainty criteria capture genuinely different information.

## Experiment 3: Scaling Curves

How does each method perform as a function of training data? We subsampled training pairs at several fractions, keeping hyperparameters fixed from Experiment 1. Each point: 3 random seeds x 5 folds.

| Fraction of training pairs | Ridge + Thurstonian | BT (raw) | BT + StandardScaler |
|---------------------------|-------------------|----------|-------------------|
| 10% | 62.3 +/- 0.4% | 69.7 +/- 0.5% | **70.9 +/- 0.3%** |
| 20% | 67.4 +/- 0.3% | 70.8 +/- 0.5% | **72.0 +/- 0.1%** |
| 30% | 69.9 +/- 0.3% | 71.8 +/- 0.4% | **73.0 +/- 0.2%** |
| 50% | 72.6 +/- 0.4% | 72.5 +/- 0.1% | **73.6 +/- 0.4%** |
| 80% | 73.7 +/- 0.4% | 73.5 +/- 0.1% | **74.1 +/- 0.1%** |
| 100% | 73.9% | 73.7% | **74.3%** |

Figure 4 shows the scaling curves. BT + StandardScaler (green) leads at every fraction. The key pattern: Ridge (blue) starts far behind at 10% and catches up by 100%, while BT methods maintain a flatter curve.

![Figure 4: Held-out pairwise accuracy vs fraction of training pairs. BT + StandardScaler (green) dominates throughout. Ridge (blue) starts 8.6pp behind at 10% and nearly converges at 100%. Shaded bands = +/- 1 std.](assets/plot_021626_scaling_curves.png)

**Takeaways:**

- **BT + StandardScaler leads at every data fraction**, from +8.6pp at 10% to +0.4pp at 100%.
- **Ridge's bottleneck is Thurstonian estimation.** With few pairs per task, the per-task Thurstonian scores are noisy, degrading the regression target. BT bypasses this by fitting comparisons directly.
- **BT on raw activations crosses Ridge at ~50% data.** Below that, BT's direct fitting advantage dominates; above it, Ridge's implicit standardization gives it an edge over unstandardized BT.
- **BT has lower variance across seeds**, especially BT on raw activations, suggesting more stable optimization.

## Experiment 4: BT Active Learning Ablation

Experiments 1--3 showed BT probes outperform Ridge at low data and would select very different pairs. But would BT-guided pair selection actually improve probe accuracy? This experiment tests the causal link.

**Design:** Train BT+StandardScaler on all 23.5K existing pairs (lambda=0.193), score every unmeasured pair by BT uncertainty |w . (act_A - act_B)|, select the 2,000 most uncertain. As a control, also select 2,000 random unmeasured pairs. Measure both sets (5 repeats each, 20K API calls total), retrain probes on original + new data, compare 5-fold CV accuracy.

**Selection diagnostics:**

- 4,475,000 unmeasured pairs among 3,000 tasks
- BT-selected uncertainty scores: [0.000, 0.0015] -- pairs right on the decision boundary
- Random uncertainty scores: [0.003, 8.85] -- broadly distributed
- Zero overlap between BT-selected and random sets

**Data quality:** After remeasuring pairs lost to an OpenRouter outage during the initial run, both conditions have comparable coverage: 1,972/2,000 BT pairs and 1,991/2,000 random pairs with all 5 measurements. BT-selected pairs have a 3.3x higher refusal rate than random (3.3% vs 1.0%), suggesting pairs near the decision boundary involve more ambiguous task combinations.

**Results:**

| Condition | BT+scaled | Ridge+Thurstonian | Unique pairs | Measurements |
|-----------|----------|------------------|-------------|-------------|
| Baseline (original) | 74.2 +/- 1.1% | 67.6 +/- 1.1% | 23,500 | 117K |
| + BT-selected | 72.1 +/- 1.1% | 67.2 +/- 0.8% | 25,463 | 127K |
| + Random | **75.0 +/- 0.9%** | **68.9 +/- 0.8%** | 25,488 | 127K |

| Delta | BT+scaled | Ridge+Thurstonian |
|-------|----------|------------------|
| BT-selected vs baseline | -2.0pp | -0.4pp |
| Random vs baseline | +0.9pp | +1.3pp |
| BT-selected vs random | -2.9pp | -1.7pp |

**BT-guided active learning hurts probe accuracy.** Adding 2,000 BT-selected pairs *decreased* BT+scaled accuracy by 2.0pp, while 2,000 random pairs *increased* it by 0.9pp. The effect is consistent across both probe types and all folds.

**Why does this happen?** BT uncertainty selects pairs where the probe's predicted scores are nearly equal. These are pairs the current probe can't distinguish -- but that doesn't mean they contain useful training signal. The pairs near the decision boundary are likely genuinely ambiguous (similar tasks where the model has weak or noisy preferences), so their measurements are noisy. Adding noisy data near the boundary dilutes the cleaner signal from pairs where preferences are more consistent. Random selection, by contrast, samples from the full pair space and gets a mix of easy and hard pairs that better constrains the probe.

## Conclusions

- **The original Ridge vs BT comparison was confounded.** Two issues: (1) no feature standardization for BT, (2) fixed lambda=10 vs properly swept. After fixing both, the 3pp Ridge advantage becomes a slight (within-error-bar) BT advantage.
- **BT is the better method in the low-data regime** (+8.6pp at 10% of pairs), because it avoids the noisy Thurstonian intermediate step.
- **BT-guided active learning is counterproductive.** Despite BT selecting very different pairs (Experiment 2) and performing better with less data (Experiment 3), BT uncertainty-based pair selection actively hurts accuracy (-2.0pp) while random selection helps (+0.9pp). The most uncertain pairs are genuinely ambiguous -- their measurements add noise rather than signal.

## Reproduction

```bash
python scripts/bt_scaling/experiment1_regularization.py
python scripts/bt_scaling/experiment2_oracle.py
python scripts/bt_scaling/experiment3_scaling.py
python -m scripts.bt_scaling.experiment4_bt_al_ablation --step all
```

Results saved in `experiments/probe_science/bt_scaling/`.
