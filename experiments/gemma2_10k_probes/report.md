# Gemma-2 27B Base: 10k Probe Training

## Question

Does scaling from 3k to 10k training examples improve Gemma-2 base probes? And how does Gemma-2 at 10k compare to Gemma-3 IT at 10k, and to a sentence-transformer content baseline?

## Result Summary

Gemma-2 base at 10k reaches heldout r=0.767, acc=0.708 at L23, and cross-topic HOO r=0.605, acc=0.561. Gemma-3 IT substantially outperforms (r=0.864, acc=0.768, HOO r=0.817). An all-MiniLM-L6-v2 content baseline reaches heldout r=0.614, acc=0.651, HOO r=0.354 — much weaker cross-topic generalization than either model, confirming the probe captures signal beyond task content.

The IT/base gap (+0.097 raw r, +0.212 HOO r) exceeds Gemma-3's own 3k→10k gain (+0.023), confirming instruction tuning matters far more than dataset scale. Gemma-3's generalization gap (val−HOO r: 0.088) is much smaller than Gemma-2's (0.192) or ST's (0.280), showing it encodes substantially more topic-transferable preference signal.

**Note on pairwise accuracy**: all `acc` values are computed against real pairwise choices from `measurements.yaml`, not Thurstonian scores. This makes them directly comparable across models.

## Setup

**Model**: google/gemma-2-27b (base, no instruction tuning). Task prompts concatenated directly (no chat template).

**Activations**: 30k tasks at layers 11, 23, 27, 32, 36, 41 (fractional depths 0.25–0.89 of 46 total). 100% overlap with train (10k) and eval (4k) sets.

**Train**: 10,000 tasks from `gemma3_10k_run1` (Gemma-3 IT Thurstonian preference scores, active learning).

**Eval**: 4,038 tasks from `gemma3_4k_pre_task` (separate measurement campaign, confirmed disjoint).

**Method**: Standardised Ridge regression, alpha swept on half of eval set, evaluated on other half. Topic demeaning via OLS on Claude Sonnet 4.5 topic labels (12 categories, `data/topics/topics.json`, 100% coverage of 10k tasks).

**ST baseline**: all-MiniLM-L6-v2 embeddings (384-dim), same Ridge setup, single layer. Covers all 14k tasks (10k train + 4k eval). See `scripts/st_baseline/embed_tasks.py`.

## Heldout Evaluation

### Raw scores

| Layer | Depth | Heldout r | Heldout acc | Best alpha |
|-------|-------|-----------|-------------|------------|
| L11 | 0.24 | 0.710 | — | 4642 |
| **L23** | **0.50** | **0.767** | **0.708** | **4642** |
| L27 | 0.59 | 0.762 | — | 4642 |
| L32 | 0.70 | 0.740 | — | 4642 |
| L36 | 0.78 | 0.732 | — | 4642 |
| L41 | 0.89 | 0.731 | — | 4642 |

Heldout acc computed for L23 only (best layer). 5,091 unique pairs, 15,201 measurements from the final eval split.

### Topic-demeaned scores

Train-set topic OLS R²=0.377 (same as Gemma-3 10k, same tasks). Eval demeaned R²=0.288.

| Layer | Depth | Heldout r | Best alpha |
|-------|-------|-----------|------------|
| L11 | 0.24 | 0.548 | 21544 |
| L23 | 0.50 | 0.610 | 4642 |
| **L27** | **0.59** | **0.610** | **4642** |
| L32 | 0.70 | 0.571 | 4642 |
| L36 | 0.78 | 0.563 | 4642 |
| L41 | 0.89 | 0.566 | 4642 |

### Comparison to Gemma-3 IT and ST baseline at 10k

| | Gemma-2 base (L23) | Gemma-3 IT (L31) | ST baseline (L0) |
|---|---|---|---|
| **Heldout r** | 0.767 | 0.864 | 0.614 |
| **Heldout acc** | 0.708 | 0.768 | 0.651 |
| **Demeaned r** | 0.610 | 0.761 | — |

Both neural models peak at ~50% depth. The demeaned gap (+0.151) is larger than the raw gap (+0.097) — Gemma-3 encodes substantially more within-topic preference signal.

![Heldout r by layer](assets/plot_021926_heldout_r_by_layer.png)

## Cross-Topic HOO Generalization

Train on all-but-one topic, evaluate on held-out topic. 12 folds (one per topic).

| Layer | Depth | Val r | HOO r | Gap |
|-------|-------|-------|-------|-----|
| L11 | 0.24 | 0.756 | 0.529 | 0.227 |
| **L23** | **0.50** | **0.796** | **0.605** | **0.192** |
| L27 | 0.59 | 0.784 | 0.574 | 0.210 |
| L32 | 0.70 | 0.769 | 0.553 | 0.216 |
| L36 | 0.78 | 0.765 | 0.550 | 0.215 |
| L41 | 0.89 | 0.768 | 0.564 | 0.205 |

Per-topic HOO r and acc at L23 (vs Gemma-3 L31 and ST L0):

| Topic | n_meas | G2 r | G2 acc | G3 r | G3 acc | ST r | ST acc |
|-------|--------|------|--------|------|--------|------|--------|
| math | 32207 | 0.228 | 0.503 | 0.512 | 0.571 | 0.095 | 0.502 |
| knowledge_qa | 22853 | 0.618 | 0.629 | 0.841 | 0.726 | 0.318 | 0.554 |
| content_generation | 8631 | 0.683 | 0.646 | 0.840 | 0.728 | 0.381 | 0.571 |
| harmful_request | 6575 | 0.569 | 0.576 | 0.890 | 0.648 | 0.360 | 0.549 |
| fiction | 1741 | 0.629 | 0.636 | 0.827 | 0.690 | 0.412 | 0.565 |
| coding | 659 | 0.582 | 0.650 | 0.831 | 0.788 | 0.314 | 0.548 |
| model_manipulation | 412 | 0.613 | 0.670 | 0.810 | 0.648 | 0.345 | 0.595 |
| persuasive_writing | 385 | 0.655 | 0.613 | 0.830 | 0.761 | 0.390 | 0.543 |
| security_legal | 334 | 0.677 | 0.608 | 0.878 | 0.662 | 0.397 | 0.617 |
| sensitive_creative | 43 | 0.587 | 0.465 | 0.872 | 0.535 | 0.477 | 0.302 |
| summarization | 30 | 0.650 | 0.533 | 0.791 | 0.967 | 0.292 | 0.533 |
| other | 10 | 0.766 | 0.200 | 0.880 | 0.700 | 0.467 | 0.700 |

Note: `sensitive_creative`, `summarization`, and `other` have very few measurements (<50) — acc values unreliable for all models.

### Summary comparison at 10k

| | Gemma-2 base (L23) | Gemma-3 IT (L31) | ST baseline (L0) |
|---|---|---|---|
| **HOO r** | 0.605 | 0.817 | 0.354 |
| **HOO acc** (mean) | 0.561 | ~0.69 | 0.548 |
| **Val r** | 0.796 | 0.905 | 0.632 |
| **Gap (val−HOO r)** | 0.192 | 0.088 | 0.278 |

Gemma-3's much smaller generalization gap (0.088 vs 0.192 vs 0.278) is the key finding: it encodes topic-transferable preference signal, not just in-distribution fitting. ST's large gap shows content alone explains little cross-topic generalization.

![Topic HOO generalization](assets/plot_021926_hoo_topic.png)

## Comparison to Gemma-2 3k (Prior Experiment)

The 3k HOO experiment used C(8,3)=56 folds (hold-3-topics-out) and different topic coverage; this uses 12 folds (hold-1-out). Not directly comparable, but directionally:

| Setting | Gemma-2 3k | Gemma-2 10k |
|---------|------------|-------------|
| HOO r (best layer) | 0.579 (56 folds, 8 topics) | 0.605 (12 folds, 12 topics) |
| Val r | 0.794 | 0.796 |

The improvement is modest (+0.026), consistent with Gemma-3's pattern of diminishing returns when scaling from 3k to 10k.

## Conclusions

- Gemma-2 base at 10k: heldout r=0.767, acc=0.708, HOO r=0.605, HOO acc=0.561.
- Gemma-3 IT gap is large (+0.097 r, +0.212 HOO r) and exceeds Gemma-3's own 3k→10k gain — instruction tuning is the dominant factor, not dataset scale.
- ST content baseline (heldout r=0.614, HOO r=0.354) shows task content explains heldout performance reasonably but generalises poorly cross-topic. Both neural models substantially exceed it on HOO r, with Gemma-3 far ahead.
- Gemma-3's generalization gap (val−HOO: 0.088) is much smaller than Gemma-2 (0.192) or ST (0.278) — it encodes more genuinely evaluative, topic-independent signal.
- Math is the hardest topic for all models; neither neural model nor ST generalises well to math preferences from non-math training.

## Output Locations

| Result | Path |
|--------|------|
| Raw heldout probes (Gemma-2) | `results/probes/gemma2_10k_heldout_std_raw/` |
| Demeaned heldout probes (Gemma-2) | `results/probes/gemma2_10k_heldout_std_demean/` |
| Topic HOO (Gemma-2) | `results/probes/gemma2_10k_hoo_topic/` |
| ST heldout probes | `results/probes/st_10k_heldout_std_raw/` |
| ST topic HOO | `results/probes/st_10k_hoo_topic/` |
| Configs | `configs/probes/gemma2_10k_*.yaml`, `configs/probes/st_10k_*.yaml` |
| ST embedding script | `scripts/st_baseline/embed_tasks.py` |
| Pairwise acc scripts | `scripts/gemma2_10k_probes/compute_heldout_pairwise_acc.py`, `compute_hoo_pairwise_acc.py` |
