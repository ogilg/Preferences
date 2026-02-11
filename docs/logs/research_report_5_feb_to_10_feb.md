# Weekly Report: Feb 5 - 10, 2026

## Summary

- Trained linear probes on revealed preference activations (last-token, Gemma-3-27B) — both Ridge regression on Thurstonian utilities and direct Bradley-Terry fitting work well.
- Bradley-Terry probes outperform Ridge on pairwise accuracy (84% vs 76%) despite low weight-vector similarity (cosine 0.62), suggesting multiple directions in activation space encode similar preference information.
- Probes generalize across topics in held-one-out evaluation; topic residualization shrinks the generalization gap but costs absolute performance.
- OOD persona test: telling the model "you hate math" shifts both behavior and probe-predicted preferences, with a positive correlation between the two — the probe tracks induced preference changes.

## 1. Probing Revealed Preferences

Ridge and BT probes trained on last-token activations from 3k revealed-preference trials (Gemma-3-27B, layer 31) both predict preferences well, but find different directions in activation space.

### 1.1 Setup

### 1.2 Ridge probes on Thurstonian utilities

Ridge regression on Thurstonian mu scores achieves R²=0.86 after standardization and alpha tuning, with noise baselines confirming the signal is genuine.

### 1.3 Bradley-Terry probes on pairwise outcomes

BT probes trained directly on pairwise win/loss data reach 84% accuracy, optimizing the loss that actually matters for preference prediction.

### 1.4 Ridge vs Bradley-Terry comparison

BT beats Ridge on pairwise accuracy (84% vs 76%) yet their weight vectors are only 0.62 cosine similar — multiple linear directions in activation space recover near-identical task rankings.

## 2. Confound Analysis

Topic and prompt length explain ~58% of preference variance; probes still predict ~50% of the residual, confirming activations carry preference signal beyond metadata.

### 2.1 What metadata predicts preferences?

OLS models decompose variance: topic + length R²=0.58, adding dataset raises it to 0.66, but dataset adds little beyond topic.

### 2.2 Probe R² after residualizing confounds

After removing topic + length, probe R² drops from 0.86 to 0.53 — substantial but expected, since between-topic preference differences are real evaluative signal, not just confound noise.

### 2.3 Topic classification gaps (harm intent)

Residual analysis reveals the topic classifier misses harmful-intent tasks disguised as benign genres (e.g. bailbench tasks labeled "persuasive_writing"), explaining why dataset dummies capture extra variance.

## 3. Generalization

Probes trained on natural preferences generalize both to held-out topics and to artificially induced preference shifts, supporting the claim that they capture transferable evaluative representations.

### 3.1 Cross-topic (held-one-out)

Held-one-out by topic (train on 11, test on the 12th) yields hoo_r=0.75 vs in-distribution val_r=0.92 — a generalization gap, but probes still predict well on unseen topics.

### 3.2 Effect of topic residualization

Residualizing topic means before training shrinks the generalization gap (0.18 → 0.05) but hurts absolute performance (hoo_r drops from 0.75 to 0.65), since between-topic variance carries genuine evaluative information.

### 3.3 OOD: persona-induced preference shifts

System prompts like "you hate math" shift both which tasks the model chooses (behavioral delta) and the probe's predicted utilities (activation delta), with a positive correlation between the two — the probe tracks artificially induced preference changes.

### 3.4 Correlation between behavioral and probe-predicted shifts

## 4. Next Steps

