# Weekly Report: Feb 5 - 10, 2026

## Summary

- Trained linear probes on revealed preference activations (last-token, Gemma-3-27B) — both Ridge regression on Thurstonian utilities and direct Bradley-Terry fitting work well.
- Bradley-Terry probes outperform Ridge on pairwise accuracy (84% vs 76%) despite low weight-vector similarity (cosine 0.62), suggesting multiple directions in activation space encode similar preference information.
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

Best λ=10 selected by val split on unique pairs (L-BFGS-B optimizer). Val accuracy degrades monotonically with stronger regularization.

### 1.3 Ridge vs Bradley-Terry

Head-to-head on raw scores (no de-meaning, layer 31):

| Metric | Ridge | BT |
|--------|-------|----|
| Pairwise accuracy | 0.758 | **0.844** |
| Pearson r vs Thurstonian μ | 0.929 | **0.986** |

BT scores correlate near-perfectly with Thurstonian μ (r=0.986) — both methods recover essentially the same task ranking. Yet their weight vectors share only **0.62 cosine similarity** (51° apart). With 3k tasks in a ~3.5k-dimensional space, many different linear directions project similarly onto the data manifold. The preference signal is not confined to a single direction in activation space.

**Caveat: these numbers are not directly comparable.** Ridge's pairwise accuracy (0.758) comes from 5-fold CV where held-out tasks are genuinely unseen. BT's (0.844) is training accuracy on all data — the train/val split during the lambda sweep splits *pairs*, not *tasks*, so a task's activations appear in both train and val pairs. After selecting best λ, BT retrains on all pairs. A fair comparison requires splitting by tasks: train BT only on pairs where both tasks are in the training fold, evaluate on pairs where both tasks are in the test fold. This would give both methods held-out-task pairwise accuracy on the same splits.

## 2. Confound Analysis

Topic and prompt length explain ~58% of preference variance; probes still predict ~50% of the residual, confirming activations carry preference signal beyond metadata.

### 2.1 What metadata predicts preferences?

OLS models decompose variance: topic + length R²=0.58, adding dataset raises it to 0.66, but dataset adds little beyond topic.

### 2.2 Probe R² after de-meaning confounds

After removing topic + length, probe R² drops from 0.86 to 0.53 — substantial but expected, since between-topic preference differences are real evaluative signal, not just confound noise.

### 2.3 Topic classification gaps (harm intent)

Residual analysis reveals the topic classifier misses harmful-intent tasks disguised as benign genres (e.g. bailbench tasks labeled "persuasive_writing"), explaining why dataset dummies capture extra variance.

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

