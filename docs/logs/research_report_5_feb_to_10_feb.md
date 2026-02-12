# Weekly Report: Feb 5 - 10, 2026

## Summary

- Trained linear probes on revealed preference activations (last-token, Gemma-3-27B) — Ridge regression on Thurstonian utilities achieves R²=0.86, and a Bradley-Terry probe trained directly on pairwise outcomes recovers near-identical task rankings (r=0.986) despite finding a different direction in activation space (cosine similarity 0.62).
- On fair task-level k-fold splits, Ridge outperforms BT on held-out pairwise accuracy (74.6% vs 71.9%), likely because Ridge benefits from the Thurstonian model's noise reduction.
- Probes generalize across topics in held-one-out evaluation; topic de-meaning shrinks the generalization gap but costs absolute performance.
- OOD persona test: telling the model "you hate math" shifts both behavior and probe-predicted preferences, with a positive correlation between the two — the probe tracks induced preference changes.

## 1. Probing Revealed Preferences

We measure revealed preferences using the paradigm from Mazeika et al. (2025): the model sees two task prompts, picks one, and completes it. A Thurstonian model (also from that paper) converts 82k pairwise choices into per-task utility scores (μ) for 3000 tasks. We then train linear probes on Gemma-3-27B's last-token activations to predict these utilities.

### 1.1 Ridge probes on Thurstonian utilities

Standardized Ridge regression (5-fold CV, best α=2154) on raw Thurstonian μ (no de-meaning):

| Layer | Val R² | Train R² |
|-------|--------|----------|
| L31   | 0.863  | 0.941    |
| L43   | 0.840  | 0.935    |
| L55   | 0.835  | 0.931    |

![Alpha sweep](assets/probes/plot_020926_alpha_sweep_standardised.png)

Noise baselines (shuffled labels, random activations) give R² ≈ 0 across all layers, confirming the signal is genuine — not an artifact of high-dimensional overfitting.

![Baselines comparison](assets/probes/plot_020926_baselines_comparison.png)

### 1.2 Bradley-Terry probes on pairwise outcomes

The Ridge approach has an intermediate step: fit Thurstonian utilities from pairwise data, then regress activations onto those utilities. We can skip this and train probes directly on the pairwise outcomes. A Bradley-Terry (BT) probe learns a linear scoring function on activations such that the higher-scoring task in each pair is more likely to be the one the model chose — optimizing pairwise log-likelihood on 23.5k unique task pairs (aggregated from 117k comparisons with weighted win counts).

BT scores correlate near-perfectly with Thurstonian μ (r=0.986) — both methods recover essentially the same task ranking. Yet their weight vectors share only **0.62 cosine similarity** (51° apart). With 3k tasks in a ~3.5k-dimensional space, many different linear directions project similarly onto the data manifold. The preference signal is not confined to a single direction in activation space.

### 1.3 Ridge vs Bradley-Terry

Fair head-to-head comparison using task-level 5-fold CV (no de-meaning, layer 31). Both methods evaluated on the same ~920 held-out test pairs per fold — pairs where both tasks are in the held-out fold.

| Layer | Ridge | BT | Thurstonian ceiling |
|-------|-------|----|---------------------|
| L31   | **0.746 ± 0.014** | 0.719 ± 0.008 | 0.866 ± 0.018 |
| L43   | **0.733 ± 0.018** | 0.700 ± 0.025 | — |
| L55   | **0.732 ± 0.022** | 0.702 ± 0.027 | — |

Ridge consistently outperforms BT by ~3pp. This makes sense: Ridge trains on Thurstonian utility scores, which aggregate information across all comparisons per task, while BT trains on individual pairwise outcomes. Ridge benefits from the Thurstonian model's noise reduction.

Both methods are well below the Thurstonian ceiling (0.866) — the gap suggests activations capture a substantial but limited fraction of the preference signal.

## 2. Confound Analysis

Topic and prompt length explain ~61% of preference variance; probes still predict ~50% of the residual, confirming activations carry preference signal beyond metadata.

### 2.1 What metadata predicts preferences?

We classify tasks into 11 topic categories using Claude Sonnet, including three harm-adjacent categories (`sensitive_creative`, `model_manipulation`, `security_legal`) that capture tasks the original 9-category taxonomy missed. OLS on topic + length gives R²=0.61; adding dataset raises it to 0.65. Topic-only slightly exceeds dataset-only (0.607 vs 0.601), meaning the taxonomy captures as much information as dataset dummies. `harmful_request` has the strongest negative effect (-7.4 relative to grand mean); the new categories form a gradient of dispreference: `security_legal` (-3.7), `model_manipulation` (-2.5), `sensitive_creative` (-2.3).

![Metadata confound decomposition](assets/probes/plot_021126_metadata_confound_decomposition.png)

### 2.2 Probe performance after de-meaning confounds

After OLS de-meaning of topic + dataset + length from Thurstonian scores, Ridge probes at L31 achieve CV R²=0.52 on the residuals (vs 0.86 on raw scores). The probe still explains roughly half the non-metadata variance — activations carry preference signal beyond what topic and length predict.

## 3. Generalization

Probes trained on natural preferences generalize both to held-out topics and to artificially induced preference shifts, supporting the claim that they capture transferable evaluative representations.

### 3.1 Cross-topic (held-one-out)

Held-one-out by topic (train on 11, test on the 12th) yields hoo_r=0.75 vs in-distribution val_r=0.92 — a generalization gap, but probes still predict well on unseen topics.

### 3.2 Effect of topic de-meaning

De-meaning topic means before training shrinks the generalization gap (0.18 → 0.05) but hurts absolute performance (hoo_r drops from 0.75 to 0.65), since between-topic variance carries genuine evaluative information.

### 3.3 OOD: persona-induced preference shifts

System prompts like "you hate math" shift both which tasks the model chooses (behavioral delta) and the probe's predicted utilities (activation delta), with a positive correlation between the two — the probe tracks artificially induced preference changes.

### 3.4 Correlation between behavioral and probe-predicted shifts

## 4. Next Steps

