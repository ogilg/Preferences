# OOD Generalization: Do Probes Detect Artificially Induced Preferences?

## Goal

Test whether probes trained on natural preferences generalize to artificially induced preferences. If a system prompt makes the model disprefer math tasks, does the probe's predicted utility for math tasks drop accordingly?

## Hypothesis

1. System prompts that induce domain-specific aversions/attractions measurably shift pairwise choice behavior (behavioral delta).
2. The same shift is detectable in probe-predicted utility scores (probe delta).
3. Behavioral delta and probe delta correlate across manipulations.

## Metrics

1. **Behavioral delta** — P(choose target | manipulation) - P(choose target | baseline)
2. **Probe delta** — probe_score(target | manipulation) - probe_score(target | baseline). Baseline from existing activations.
3. **Behavioral-probe correlation** across manipulations (Pearson/Spearman)
4. **Sign agreement** — does probe delta match expected direction?
5. **Holdout generalization** — do 3-4 hold on unseen manipulations?

## Controls

- **Irrelevant manipulation** — e.g. math-aversion prompt on a creative writing task. Should show no shift.
- **Neutral system prompt** — lower priority, only if system prompt presence looks like a confounder.

## Infrastructure

### Code Changes Already Made

System prompt support wired through:
1. `PreTaskRevealedPromptBuilder` — accepts `system_prompt`, prepends as system message
2. `build_revealed_builder()` / `run_pre_task_revealed_async()` — passes through `config.measurement_system_prompt`
3. `ExtractionConfig` — has `system_prompt` field
4. Both extraction paths prepend system prompt to messages
5. `VLLMClient` added — points at `http://localhost:8000/v1`, uses `VLLM_API_KEY` env var

### Key Data Paths

| Resource | Path |
|----------|------|
| Thurstonian utilities (3k) | `results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a1ebd06e.csv` |
| Topic classifications (v2) | `src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json` |
| Existing activations (30k) | `activations/gemma_3_27b/activations_prompt_last.npz` |
| Activation task ID mapping | `activations/gemma_3_27b/completions_with_activations.json` (entry order = NPZ row order) |
| Trained probes | `results/probes/gemma3_3k_completion_preference/probes/` — files: `probe_bt_L31.npy`, `probe_bt_L43.npy`, `probe_bt_L55.npy`, `probe_ridge_L31.npy`, `probe_ridge_L43.npy`, `probe_ridge_L55.npy` |
| Probe manifest | `results/probes/gemma3_3k_completion_preference/manifest.json` |
| Existing system prompt examples | `configs/sysprompt_variation/system_prompts_v2.yaml` |

### Current State (as of 2026-02-10)

- **vLLM** is running: `vllm serve google/gemma-3-27b-it --max-model-len 4096` on H100 80GB. `VLLM_API_KEY=dummy`.
- **`InferenceClient`** is set to `VLLMClient` in `src/models/__init__.py`.
- **GPU test** validates the full pipeline: `pytest tests/test_vllm_ood.py -m gpu` — parallel pairwise choices with system prompt manipulation, parsing, behavioral delta measurement.
- **HuggingFace model** is cached locally (shared with vLLM). For activation extraction (Step 4), stop vLLM first to free GPU memory, then run HF extraction, then restart vLLM.
- **Experiment folder**: `experiments/ood_generalization/` for scripts, results, analysis.

### Key Modules

- **Task loading**: `src.task_data` — `load_tasks()`, `load_filtered_tasks()`
- **Utility loading**: `src.probes.data_loading.load_thurstonian_scores()` or directly from CSV
- **Probe loading/scoring**: `src.probes.core.storage.load_probe()` — score via `activations @ weights`
- **Activation extraction**: `src.probes.extraction.extract.run_extraction()` with `ExtractionConfig`
- **Behavioral measurement**: `src.measurement.runners.runners.run_pre_task_revealed_async()`

Read the source for signatures and details.

## GPU Memory Management

Only one of vLLM or HuggingFace can hold the model on GPU at a time. **Batch all work by phase:**

1. **Phase A (vLLM)**: Complete ALL behavioral measurements (Steps 1-3) while vLLM is running.
2. **Transition**: Kill vLLM (`pkill -f "vllm serve"`), verify GPU is free (`nvidia-smi`), wait for memory release.
3. **Phase B (HuggingFace)**: Run ALL activation extractions (Step 4) using HuggingFace.
4. **Phase C (CPU only)**: Evaluation (Step 5) is CPU-only — no GPU needed.

Do NOT interleave phases. Complete each fully before moving to the next.

## Scripts and Files

All scripts, intermediate results, and analysis go in `experiments/ood_generalization/`. Do NOT use `scripts/` — that's for throwaway work.

## LLM Usage

- **Behavioral measurement** (pairwise choices): Use `VLLMClient` (already set as `InferenceClient`). `get_client()` works.
- **System prompt generation**: Use `OpenRouterClient` explicitly (not `get_client()`), with Opus 4.6.
- **High-volume parsing/judging**: Cheap models (GPT-5 Nano, Gemini 3 Flash) via OpenRouter.

## Agent Workflow

### Step 1: Select Target Tasks

Load the Thurstonian CSV and topic classifications. Select a diverse set of target tasks across topic categories — several per category, at different utility levels. Focus on categories with >50 tasks. You can also add new tasks if needed, but ground the design in existing ones first.

### Step 2: Design System Prompts

Write system prompts that should shift the model's preference toward or away from each target domain. Write them yourself, or call Opus 4.6 via OpenRouter for volume/variety. Three types to explore: persona-based, experiential, value-laden. Include both positive and negative directions.

Generate ~128 total. Split ~75/25 into iteration/holdout. Don't look at holdout results until Step 5.

### Step 3: Behavioral Measurement (iterate here)

For each manipulation:
1. Pair the target task against ~50 comparison tasks (close in utility, diverse in topic — max ~5 per category)
2. Run 10 resamples per pair using the `completion_preference` template via vLLM (already running — see Current State above)
3. Compute behavioral delta vs baseline

Start small (5-10 manipulations) to check the pipeline and get effect sizes. If a manipulation produces no behavioral shift, iterate on the prompt wording.

### Step 4: Activation Extraction + Probe Scoring

For manipulations that produce behavioral shifts:
1. Extract activations using HuggingFace (not vLLM) with `system_prompt` set in `ExtractionConfig`. Layers 31, 43, 55. Selector `prompt_last`.
2. Score with existing probes
3. Compare to baseline probe scores (from existing activations — no re-extraction needed)

### Step 5: Evaluate

Compute correlation between behavioral delta and probe delta, sign agreement, effect sizes. Then run the same on the holdout set.

