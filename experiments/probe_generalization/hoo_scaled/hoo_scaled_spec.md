# Scaled HOO Topic Generalization

## Problem

Current HOO results hold out 1 topic at a time (8 folds), Ridge only, no BT, no content baseline. This is a weak test — with 7 training topics the probe sees most of the preference distribution. Holding out 3 topics at once is a much harder test: the probe must generalize across larger distributional shifts.

We also lack a content baseline — we don't know how much of the HOO performance comes from the probe capturing genuine evaluative signal vs just encoding task content (which correlates with preferences).

## Goal

1. Run HOO with hold_out_size=3 (train on 5, eval on 3) for Ridge and BT, with and without topic de-meaning
2. Add a sentence-transformer content baseline on the same splits
3. Show that activation probes generalize beyond what content alone predicts

## Success Criteria

A comparison table like:

| Method | Condition | Mean hoo metric | Std | Gap (val - hoo) |
|--------|-----------|-----------------|-----|-----------------|
| Ridge | Raw | ? | ? | ? |
| Ridge | Topic de-meaned | ? | ? | ? |
| BT | Raw | ? | ? | ? |
| Sentence-transformer | Raw | ? | ? | ? |

Plus per-fold breakdown and plots.

Ridge and ST use Pearson r. BT uses pairwise accuracy.

## Method

### Topic groups

8 topics from v2 classification: math, fiction, coding, persuasive_writing, content_generation, summarization, knowledge_qa, harmful_request.

Group sizes (from research log):

| Topic | n |
|-------|---|
| knowledge_qa | 712 |
| math | 668 |
| harmful_request | 617 |
| content_generation | 417 |
| fiction | 250 |
| coding | 146 |
| persuasive_writing | 146 |
| summarization | 36 |

### Fold design

C(8,3) = 56 total combinations. Run all 56 — each fold is fast (Ridge is closed-form, BT is a single L-BFGS fit at fixed lambda). 56 folds × 3 layers × 4 conditions = 672 fits, but most are sub-second (Ridge) or ~10s (BT).

Hyperparameters (alpha for Ridge, lambda for BT) selected on fold 0 and fixed for all subsequent folds. This is already how `run_hoo` works — it sweeps on fold 0 via `best_hp=None`, then passes the learned value to subsequent folds. Same approach as the BT fair comparison experiment.

### Conditions

1. **Ridge raw**: Train Ridge on raw Thurstonian μ, eval Pearson r on held-out topics
2. **Ridge topic de-meaned**: De-mean training μ by topic before fitting, eval Pearson r on raw held-out μ
3. **BT raw**: Train BT on pairs where both tasks are in train topics, eval pairwise accuracy on pairs where both tasks are in held-out topics
4. **Sentence-transformer baseline**: Train Ridge from ST embeddings → Thurstonian μ on train topics, eval Pearson r on held-out topics. Uses all-MiniLM-L6-v2 (384d) embeddings.

### Sentence-transformer baseline details

The ST baseline answers: "how well can you predict held-out preferences from task content alone?"

For each fold:
1. Encode all task prompts with all-MiniLM-L6-v2
2. Train Ridge (with alpha sweep on fold 0, fixed thereafter) from ST embeddings → Thurstonian μ on train-topic tasks
3. Predict μ for held-out-topic tasks
4. Report Pearson r against true μ

If the activation probe's hoo_r significantly exceeds the ST baseline's hoo_r, the probe captures evaluative signal beyond content.

## Data

- **Activations**: `activations/gemma_3_27b/activations_prompt_last.npz` (3000 tasks, layers 31/43/55)
- **Thurstonian scores**: `results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/`
- **Pairwise measurements**: same run_dir (measurements.yaml)
- **Topics (v2)**: `src/analysis/topic_classification/output/topics_v2.json`
- **Task prompts**: load via `src.task_data` — needed for ST embedding

## Existing Infrastructure

Almost everything exists in `src/probes/experiments/run_dir_probes.py`:

- **`run_hoo(config)`**: Main HOO pipeline. Already supports `hoo_hold_out_size > 1`, both Ridge and BT modes, topic de-meaning via `demean_confounds`, and hyperparameter reuse across folds.
- **`hoo_ridge.make_method()`**: Ridge HOO with optional de-meaning (train-only).
- **`hoo_bt.make_method()`**: BT HOO using `split_by_groups`.
- **Config**: `configs/probes/gemma3_hoo_topic.yaml` — existing HOO config, just needs `hoo_hold_out_size: 3`.

**What's new**: the sentence-transformer baseline. This requires:
1. Extracting ST embeddings for all 3000 tasks (one-time, save to disk)
2. A new HOO method (analogous to `hoo_ridge.make_method()`) that uses ST embeddings instead of Gemma activations

## Implementation Plan

### Step 1: Run existing HOO at scale

Run the existing pipeline with `hoo_hold_out_size: 3` for these configs:

**Config A — Ridge raw + BT raw:**
```yaml
experiment_name: hoo_scaled_raw
run_dir: results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0
activations_path: activations/gemma_3_27b/activations_prompt_last.npz
output_dir: results/probes/hoo_scaled_raw
layers: [31, 43, 55]
modes: [ridge, bradley_terry]
cv_folds: 5
alpha_sweep_size: 10
standardize: true
topics_json: src/analysis/topic_classification/output/topics_v2.json
hoo_grouping: topic
hoo_hold_out_size: 3
```

**Config B — Ridge topic de-meaned:**
```yaml
experiment_name: hoo_scaled_demeaned
run_dir: results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0
activations_path: activations/gemma_3_27b/activations_prompt_last.npz
output_dir: results/probes/hoo_scaled_demeaned
layers: [31, 43, 55]
modes: [ridge]
cv_folds: 5
alpha_sweep_size: 10
standardize: true
topics_json: src/analysis/topic_classification/output/topics_v2.json
demean_confounds: [topic]
hoo_grouping: topic
hoo_hold_out_size: 3
```

### Step 2: Sentence-transformer baseline

Write a script that:
1. Loads all 3000 task prompts
2. Encodes with `sentence-transformers/all-MiniLM-L6-v2` (CPU is fine, 3k tasks is fast)
3. Saves embeddings to `activations/sentence_transformer/embeddings.npz` (same format as Gemma activations: `task_ids` + a single "layer" key)
4. Runs Ridge HOO on these embeddings with the same fold structure (hoo_hold_out_size=3, all 56 folds)

The simplest approach: save ST embeddings in the same npz format, then run the existing pipeline with `activations_path` pointing to the ST file and `layers: [0]` (single "layer"). This avoids writing any new HOO code.

### Step 3: Analysis and plots

1. Aggregate results across all 56 folds for each condition
2. Per-condition summary: mean, std, median of hoo metric
3. Per-held-out-topic breakdown: which topics are hardest to generalize to?
4. Paired comparison: for each fold, compare Ridge hoo_r vs ST hoo_r (sign test or paired t-test)
5. Plot: box/violin plot of hoo metrics across folds for all conditions

## Fallbacks

- If 56 folds × BT is too slow, subsample to 20 random folds. But this is unlikely — each BT fit at fixed lambda takes ~10s, so 56 × 3 layers ≈ 30 minutes.
- If ST embeddings are too similar to Gemma activations (high correlation), that's an interesting finding — report it.
- If the `summarization` topic (n=36) causes issues in held-out evaluation (too few tasks for meaningful Pearson r), note it but don't exclude — the fold averaging will smooth it out.
