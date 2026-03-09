# Stronger content baseline: Qwen3-Embedding-8B

## Motivation

LW commenter pointed out that all-MiniLM-L6-v2 (22M params, 384d) is too weak a content baseline. If the claim is that model internals capture something beyond task content, the content encoder should be strong enough to make that a fair test.

## Goal

Replace the sentence-transformer baseline with **Qwen3-Embedding-8B** (`Qwen/Qwen3-Embedding-8B`, 7.5B params, 4096d) and rerun the two probe evaluations from Section 2 of the post:

1. **Heldout eval**: Ridge probe on embeddings → Thurstonian utilities (train 10k, eval on 4k heldout split). Current ST result: r = 0.614.
2. **HOO cross-topic**: Train on 11/12 topics, evaluate on held-out topic, all 12 folds. Current ST result: mean cross-topic r = 0.354.

## Method

### Step 1: Extract Qwen embeddings

Use `src.probes.content_embedding` — it works with any `SentenceTransformer`-compatible model:

```python
from src.probes.content_embedding import embed_tasks, save_content_embeddings

task_ids, embeddings = embed_tasks(
    completions_json=Path("activations/gemma_3_27b/completions_with_activations.json"),
    model_name="Qwen/Qwen3-Embedding-8B",
)
save_content_embeddings(
    path=Path("activations/qwen3_emb_8b/activations_prompt_last.npz"),
    task_ids=task_ids,
    embeddings=embeddings,
)
```

The JSON schema is `[{"task_id": str, "task_prompt": str, "origin": str}, ...]`. `embed_tasks` reads `task_prompt` and returns `(task_ids, embeddings)`. `save_content_embeddings` saves as `.npz` with keys `task_ids` and `layer_0`, matching the format expected by `src.probes.core.activations.load_activations`.

After saving, delete the model from GPU memory before proceeding to probing.

### Step 2: Run heldout probe

```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/qwen3_emb_8b_heldout_std_raw.yaml
```

Config already exists at `configs/probes/qwen3_emb_8b_heldout_std_raw.yaml`.

### Step 3: Run HOO cross-topic probe

```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/qwen3_emb_8b_hoo_topic.yaml
```

Config already exists at `configs/probes/qwen3_emb_8b_hoo_topic.yaml`.

Do NOT reimplement probe training, alpha sweeping, or HOO fold logic — the existing pipeline handles all of it.

## Data

All paths relative to repo root.

**In git (no sync needed):**
- `configs/probes/qwen3_emb_8b_heldout_std_raw.yaml` — heldout probe config
- `configs/probes/qwen3_emb_8b_hoo_topic.yaml` — HOO probe config
- `src/probes/content_embedding.py` — embedding extraction code

**Must be synced to pod (gitignored):**
- `activations/gemma_3_27b/completions_with_activations.json` — task prompts (14k tasks)

**run_dir paths (used by probe configs):**
- `results/experiments/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0` — 10k training preferences
- `results/experiments/gemma3_4k_pre_task/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0` — 4k heldout eval preferences

These paths match the existing probe configs. If they don't exist on the pod, check under `results/experiments/main_probes/` instead.

**Topics file:** `data/topics/topics.json` — topic labels for HOO grouping. Note: the existing `gemma3_10k_hoo_topic.yaml` references `src/analysis/topic_classification/output/topics.json` which may not exist on the pod. If missing, update the config to use `data/topics/topics.json`.

## Current baselines to compare against

| Model | Heldout r | Cross-topic r (HOO) |
|-------|-----------|-------------------|
| Gemma-3-27B IT (L31) | 0.864 | 0.82 |
| Gemma-3-27B PT (L31) | — | 0.63 |
| all-MiniLM-L6-v2 (384d, 22M) | 0.614 | 0.354 |
| **Qwen3-Embedding-8B (4096d, 7.5B)** | **?** | **?** |

## Commit guidance

Do NOT commit `activations/qwen3_emb_8b/activations_prompt_last.npz` (~200MB+). Leave it on the pod. DO commit:
- The two new YAML configs in `configs/probes/`
- Probe result manifests in `results/probes/qwen3_emb_8b_*/`
- The experiment report

## Success criteria

The experiment is done when both probe configs have run and produced manifests with `final_r` (heldout) and `mean_hoo_r` (HOO) values.

Sanity check: the Qwen result should be ≥ the MiniLM baseline (heldout r ≥ 0.614, HOO r ≥ 0.354). If it's lower, something is wrong — debug before reporting.

## Pod requirements

- Qwen3-Embedding-8B: ~16GB in bf16, fits on H100 80GB
- Embedding extraction is the only GPU step; probing is CPU-only
- Sequential: embed → save → delete model → run probes

## Outputs

- `activations/qwen3_emb_8b/activations_prompt_last.npz` (pod only, do not commit)
- `results/probes/qwen3_emb_8b_heldout_std_raw/manifest.json`
- `results/probes/qwen3_emb_8b_hoo_topic/hoo_summary.json`
- `experiments/probe_science/qwen_embedding/qwen_embedding_report.md`
