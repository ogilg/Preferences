# Preferences

MATS 9.0 project with Patrick Butlin investigating whether LLM preferences are driven by evaluative representations.

> **For AI agents:** This README describes the codebase's key modules and entry points. Before writing new extraction, embedding, or probe training code, check the relevant module below — the functionality likely already exists.

## Motivation

Whether LLMs are moral patients may depend on whether they have evaluative representations playing the right functional roles — internal representations that encode valuation and causally influence choice. Across major theories of welfare (hedonism, desire satisfaction, etc.), such representations are central to moral patiency (Long et al., 2024).

*Preferences* are behavioral patterns — choosing A over B. *Evaluative representations* are the hypothesized internal mechanism: representations that encode "how good/bad is this?" and causally drive those choices. The question is whether preferences are driven by evaluative representations, or by something else (e.g., surface-level heuristics, training artifacts).

## Goals

We look for evaluative representations as linear directions in activation space. The methodology follows from the definition:

1. **Probing** — If they encode value, probes should predict preferences
2. **Steering** — If they causally drive choices, steering should shift them
3. **Generalization** — If they're genuine evaluative representations, they should generalize across contexts

We ground this in revealed preferences (pairwise choices where the model picks which task to complete), which have cleaner signal than stated ratings where models collapse to default values.

## Structure

```
src/
├── probes/            # Activation extraction, probe training, and evaluation (see below)
├── steering/          # Activation steering experiments
├── measurement/       # Preference measurement (pairwise choices, stated ratings)
├── fitting/           # Utility function extraction (Thurstonian, TrueSkill)
├── models/            # LLM clients (OpenRouter)
├── task_data/         # Task datasets (WildChat, Alpaca, MATH, BailBench)
├── experiments/       # Core experiment scripts
└── analysis/          # Post-hoc analysis (probes, steering, correlations, etc.)
```

### `src/probes/` — Activation extraction, probe training, and evaluation

#### `extraction/` — Extract activations from any HuggingFace model

Config-driven pipeline that loads a HuggingFace model, runs it on tasks, and saves per-layer activations as `.npz` files. Supports batching, periodic checkpointing (`save_every`), and `--resume` to skip already-extracted tasks. Works with any model — use it for content encoder embeddings too, not just the target model.

Use `--from-completions` to extract activations from an existing completions JSON (skips re-generation).

```bash
python -m src.probes.extraction.run configs/extraction/<config>.yaml [--resume] [--from-completions path.json]
```

Config fields: `model`, `backend` (huggingface/transformer_lens), `layers_to_extract` (fractional like 0.5 = middle layer), `selectors` (e.g. prompt_last), `batch_size`, `n_tasks`, `task_origins`. See `configs/extraction/` for examples.

#### `experiments/` — Probe training orchestration

`run_dir_probes.py` is the main entry point for training probes from a measurement run. Loads Thurstonian scores and/or pairwise comparisons, loads activations, and trains Ridge and/or Bradley-Terry probes.

When `eval_run_dir` is set, uses heldout evaluation (preferred): trains on `run_dir`, sweeps alpha on half the eval set, evaluates on the other half. Otherwise falls back to CV alpha selection. Supports: score demeaning against confounds (topic, dataset), content-orthogonal projection, held-one-out (HOO) evaluation by group.

```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/<config>.yaml
```

Config fields: `run_dir`, `activations_path`, `layers`, `modes` (ridge/bradley_terry), `eval_run_dir` (optional, for heldout eval), `eval_split_seed`, `demean_confounds`, `content_embedding_path`, `hoo_grouping`. See `configs/probes/` for examples.

#### `core/` — Probe training and evaluation primitives

- **`linear_probe.py`** — Ridge regression with CV alpha sweep (`alpha_sweep` for raw sweep results, `train_and_evaluate` to sweep + fit final model, `train_at_alpha` for fixed alpha). Returns probe, eval metrics, and sweep results.
- **`activations.py`** — Loads `.npz` activation files with optional task ID filtering and layer selection (`load_activations`).
- **`evaluate.py`** — Probe evaluation: `evaluate_probe_on_data` (given activations + scores), `evaluate_probe_on_template` (cross-template transfer), `compute_probe_similarity` (cosine similarity between probe weight vectors).

#### `content_orthogonal.py` — Project out content-predictable variance

Fits Ridge regression from content embeddings to activations (or scores), subtracts predictions. The residuals contain only variance the content encoder cannot explain. Key functions: `project_out_content` (activations), `project_out_content_from_scores` (scalar scores). Use `alpha_sweep` from `core/linear_probe.py` to CV-select the content Ridge alpha.

#### `content_embedding.py` — Sentence transformer embeddings of task prompts

Embeds task prompts using a sentence transformer (default: `all-MiniLM-L6-v2`). Used as input to content-orthogonal projection. Functions: `embed_tasks` (from completions JSON), `save_content_embeddings`, `load_content_embeddings` (from `.npz`).

#### `residualization.py` — OLS demeaning against categorical confounds

Removes group-level mean differences (topic, dataset) from preference scores via OLS on one-hot indicators. Distinct from content projection: this removes categorical confounds, content projection removes continuous content-predictable variance. Key function: `demean_scores(scores, topics_json, confounds=["topic", "dataset"])`.

### `src/steering/` — Activation steering experiments

#### `client.py` — Steered model client

`SteeredHFClient` wraps a `HuggingFaceModel` with a steering direction and coefficient, duck-typed as `OpenAICompatibleClient`. Handles the coef==0 bypass, pre-computes the scaled tensor on GPU.

For coefficient sweeps, use `with_coefficient()` to create new clients sharing the same loaded model:

```python
from src.steering.client import create_steered_client

client = create_steered_client("gemma-3-27b", probe_dir, "ridge_L31", coefficient=0)
for coef in [-3000, -1000, 0, 1000, 3000]:
    steered = client.with_coefficient(coef)
    response = steered.generate(messages)
```

`create_steered_client()` is the one-liner entry point: loads model + probe, returns a ready client. Use `compute_activation_norms()` from `src/probes/core/activations` to calibrate coefficients relative to activation norms.

#### `runner.py` — Steering experiment runner

Config-driven runner for stated preference steering experiments (steer during post-task rating). Handles task loading, completion loading, template-based rating prompts, and structured result serialization.

### `src/measurement/` — LLM judges

When building LLM judges (coherence, valence, refusal, etc.), follow the existing pattern in `src/measurement/elicitation/refusal_judge.py`: use `instructor` with Pydantic response models for structured output. Do not write regex-based response parsers — use `instructor.from_openai()` which guarantees valid structured responses from the LLM.