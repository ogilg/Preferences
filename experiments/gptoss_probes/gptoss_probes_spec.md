# GPT-OSS-120B Probe Training & Cross-Topic Generalisation

## Goal

Train linear probes on GPT-OSS-120B activations predicting revealed preferences (Thurstonian utilities), replicating the Gemma3 probe pipeline (Steps 3-4 from `docs/pipeline.md`). Test whether preference signal generalises across content topics.

## Hypothesis

GPT-OSS-120B should encode evaluative representations in its residual stream, detectable by Ridge probes. If the signal is genuine (not topic-confounded), held-one-out cross-topic probes should retain significant predictive power on held-out topics.

## Key Data Paths

All data is already collected — this experiment is CPU-only (no GPU, no API calls).

| Resource | Path |
|----------|------|
| 10k train run | `results/experiments/gpt_oss_120b_10k_completion_preference/pre_task_active_learning/completion_preference_gpt-oss-120b_completion_canonical_seed0` |
| 3k eval run | `results/experiments/gpt_oss_120b_3k_completion_preference_rerun/pre_task_active_learning/completion_preference_gpt-oss-120b_completion_canonical_seed0` |
| Activations (30k tasks, 9 layers) | `activations/gpt_oss_120b/activations_prompt_last.npz` |
| Topic labels | `data/topics/topics.json` |
| Probe configs | `configs/probes/gptoss_120b_10k_heldout_std_raw.yaml`, `configs/probes/gptoss_120b_10k_heldout_std_demeaned.yaml`, `configs/probes/gptoss_120b_10k_hoo_topic.yaml` |

## Background

GPT-OSS-120B is a 36-layer reasoning model. Active learning used low reasoning effort (CoT present but minimal). Activations were extracted at the `prompt_last` position — the last token of the prompt before generation begins, so the probe sees the model's state before any chain-of-thought output. Extracted layers: [3, 7, 10, 14, 18, 21, 25, 28, 32] (every ~10% of depth across 36 total layers).

## Method

### Step 1: Train Ridge probes with heldout evaluation

Train on 10k Thurstonian utilities, evaluate on 3k (disjoint set). Sweep alpha on half the eval set, report on the other half. Two variants:

**1a. Raw scores:**
```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/gptoss_120b_10k_heldout_std_raw.yaml
```

**1b. Topic-demeaned scores** (OLS-demean topic means before fitting):
```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/gptoss_120b_10k_heldout_std_demeaned.yaml
```

Both use standardised Ridge regression across all 9 layers. Configs are already created — do not modify them.

### Step 2: Hold-one-out cross-topic generalisation

Train on all-but-one topic group, evaluate on the held-out group. One fold per topic.

```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/gptoss_120b_10k_hoo_topic.yaml
```

### Step 3: Analysis and report

After all three runs complete:

1. Report per-layer metrics (Pearson r, pairwise accuracy) for Steps 1a and 1b. Identify the best layer.
2. Compare raw vs demeaned — how much signal survives topic demeaning?
3. For HOO (Step 2): report mean held-out r across folds, per layer. Compare to in-distribution r.
4. Compare results to the Gemma3 baseline (reference: `results/probes/gemma3_10k_heldout_std_raw/` and `results/probes/gemma3_10k_hoo_topic/`). Key question: does GPT-OSS show similar, weaker, or stronger probe performance?
5. Generate plots:
   - Per-layer Pearson r bar chart (raw vs demeaned)
   - HOO held-out r across topics (bar chart or heatmap)

Plots go in `experiments/gptoss_probes/assets/`.

## Success Criteria

- Heldout Pearson r > 0.3 on the best layer (raw scores) — indicates detectable preference signal.
- Topic-demeaned r retains at least 50% of raw r — indicates signal is not purely topic-driven.
- HOO mean held-out r > 0.2 — indicates cross-topic generalisation.

These thresholds are conservative. Gemma3 achieved r ~ 0.5-0.6 on raw scores.

## Parameters

- Layers: [3, 7, 10, 14, 18, 21, 25, 28, 32]
- Ridge alpha sweep: 10 points (log-spaced, selected by `alpha_sweep_size`)
- Eval split seed: 42
- Standardize: true
- HOO grouping: topic, hold out 1

## Infrastructure

All code exists — use `src.probes.experiments.run_dir_probes` with the provided configs. No new code needed. The activations NPZ is ~3GB; ensure sufficient RAM to load it.
