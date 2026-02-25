# GPT-OSS-120B Probe Training & Cross-Topic Generalisation

## Goal

Train linear probes on GPT-OSS-120B activations predicting revealed preferences (Thurstonian utilities). Test whether preference signal generalises across content topics. Rerun of previous attempt which had partial activation/utility overlap (~38%); this run uses utilities computed on tasks with activations, giving 10k train / ~1.6k eval.

## Hypothesis

GPT-OSS-120B should encode evaluative representations in its residual stream, detectable by Ridge probes. If the signal is genuine (not topic-confounded), held-one-out cross-topic probes should retain significant predictive power on held-out topics.

## Key Data Paths

All data is already collected — this experiment is CPU-only (no GPU, no API calls).

| Resource | Path |
|----------|------|
| 10k train run | `results/experiments/gpt_oss_120b_10k_actonly/pre_task_active_learning/completion_preference_gpt-oss-120b_completion_canonical_seed0` |
| ~1.6k eval run | `results/experiments/gpt_oss_120b_3k_actonly/pre_task_active_learning/completion_preference_gpt-oss-120b_completion_canonical_seed0` |
| Activations (30k tasks, 9 layers) | `activations/gpt_oss_120b/activations_prompt_last.npz` |
| Topic labels | `data/topics/topics.json` |
| Probe configs | `configs/probes/gptoss_120b_10k_heldout_std_raw.yaml`, `configs/probes/gptoss_120b_10k_heldout_std_demeaned.yaml`, `configs/probes/gptoss_120b_10k_hoo_topic.yaml` |

The eval run contains 3k tasks but 1,372 overlap with train; `run_dir_probes.py` filters these out, leaving ~1,628 clean eval tasks.

## Background

GPT-OSS-120B is a 36-layer reasoning model. Activations extracted at the `prompt_last` position (last token before generation). Layers: [3, 7, 10, 14, 18, 21, 25, 28, 32] (every ~10% of depth).

## Method

### Step 1: Train Ridge probes with heldout evaluation

Train on 10k Thurstonian utilities, evaluate on ~1.6k (disjoint after overlap removal). Sweep alpha on half the eval set, report on the other half.

**1a. Raw scores:**
```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/gptoss_120b_10k_heldout_std_raw.yaml
```

**1b. Topic-demeaned scores** (OLS-demean topic means before fitting):
```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/gptoss_120b_10k_heldout_std_demeaned.yaml
```

### Step 2: Hold-one-out cross-topic generalisation

Train on all-but-one topic group, evaluate on the held-out group. One fold per topic.

```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/gptoss_120b_10k_hoo_topic.yaml
```

### Step 3: Analysis and report

1. Report per-layer metrics (Pearson r, pairwise accuracy) for Steps 1a and 1b. Identify the best layer.
2. Compare raw vs demeaned — how much signal survives topic demeaning?
3. For HOO (Step 2): report mean held-out r across folds, per layer. Compare to in-distribution r.
4. Compare results to Gemma3 baseline (`results/probes/gemma3_10k_heldout_std_raw/` and `results/probes/gemma3_10k_hoo_topic/`). Key question: with matched training N, does the topic-confound gap persist?
5. Generate plots:
   - Per-layer Pearson r bar chart (raw vs demeaned)
   - HOO held-out r across topics (bar chart or heatmap)

Plots go in `experiments/gptoss_probes/assets/`.

## Success Criteria

- Heldout Pearson r > 0.3 on the best layer (raw scores)
- Topic-demeaned r retains at least 50% of raw r
- HOO mean held-out r > 0.2
