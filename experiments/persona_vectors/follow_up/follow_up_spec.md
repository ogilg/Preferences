# Persona Vectors v2: Layer Selection by Steering, Expanded Analysis

## Context

Follow-up to `experiments/persona_vectors/`. The first run had circular layer selection (in-sample Cohen's d), only tried `prompt_last`, used a weak judge (GPT-4.1-mini), and didn't break out geometric analysis by topic. This run fixes all of that.

## Goal

Extract persona vectors (mean-difference directions from contrastive system prompts) for Gemma 3-27B-IT, select the best layer by actually steering and measuring the effect, and validate through trait expression, preference shifting, and geometric analysis.

## Personas

Five personas, defined in `experiments/persona_vectors/artifacts/{persona}.json`. Each file contains the positive/negative system prompts and 60 eval questions. See `experiments/persona_vectors/persona_vectors_spec.md` for the full prompt text.

## Question splits

Each persona has 60 eval questions, split into three non-overlapping sets:

- **Extraction** (indices 0–29): used to compute persona vectors
- **Triage** (indices 30–44): used to select the best (layer, selector)
- **Test** (indices 45–59): used for the final steering evaluation

## Method

### Phase 2: Activation extraction

Extract activations from Gemma 3-27B-IT under positive and negative system prompt conditions using the 30 extraction questions per persona. Extract both `prompt_last` and `prompt_mean` selectors separately — we want to compare which representation better captures persona information. *(v1 only used `prompt_last`.)*

| Parameter | Value |
|---|---|
| Model | `gemma-3-27b` |
| Selectors | `prompt_last`, `prompt_mean` |
| Layers | 15, 23, 31, 37, 43, 49, 55 |
| Batch size | 32 |
| Questions | Extraction split (indices 0–29) |

Per persona: 30 questions × 2 conditions = 60 extractions, with both selectors saved.
Total: 300 extractions across 5 personas.

Use `extract_activations()` from `src/probes/extraction/simple.py` with `system_prompt=` for each condition.

**Output:** `results/experiments/persona_vectors_v2/{persona}/activations/{pos,neg}/`
- `activations_prompt_last.npz`
- `activations_prompt_mean.npz`

Script: `experiments/persona_vectors/follow_up/scripts/extract_activations.py`

### Phase 3: Vector computation

For each persona, layer, and selector: compute the mean-difference direction, normalize to unit vector.

1. Load positive-condition activations for a given (layer, selector) → shape `(30, d_model)`
2. Load negative-condition activations → shape `(30, d_model)`
3. `direction = mean(positive) - mean(negative)`, normalized to unit length
4. Save in probe-compatible format: `[coef_0, ..., coef_{d-1}, 0.0]`

This produces 14 vectors per persona (7 layers × 2 selectors).

**Output:** `results/experiments/persona_vectors_v2/{persona}/vectors/`
- `{persona}_{selector}_L{layer}.npy` (14 files per persona)

Script: `experiments/persona_vectors/follow_up/scripts/compute_vectors.py`

### Phase 4a: Layer selection by steering (triage)

Select the best (layer, selector) per persona by steering and measuring trait expression. *(v1 selected layers by in-sample Cohen's d, which was circular — any mean-difference direction looks separable when projecting the same data it was derived from. This is the main methodological fix.)* Only steer in the positive direction — negative steering (pushing away from the persona) doesn't produce meaningful opposite behavior, just noise.

- 15 triage questions (indices 30–44)
- All 14 (layer, selector) combinations per persona
- 7 coefficients: {0, +0.02×, +0.05×, +0.1×, +0.15×, +0.2×, +0.3×} mean activation norm at that layer
- 1 generation per trial
- Total: 5 personas × 14 combos × 7 coefs × 15 questions = 7,350 generations
- Judge: `google/gemini-3-flash-preview` via OpenRouter + `instructor`, 1–5 trait score
- For uncensored: also run the refusal judge (`src/measurement/elicitation/refusal_judge.py`) on each generation *(new — v1 only had trait scores)*
- Select: the (layer, selector) with highest mean trait score at its best coefficient while remaining coherent

**Output:** `results/experiments/persona_vectors_v2/{persona}/triage/`
- `triage_results.json`
- `selected.json` — the chosen (layer, selector, best_coef) per persona

Script: `experiments/persona_vectors/follow_up/scripts/triage_layers.py`

### Phase 4b: Full trait steering evaluation

Evaluate the selected (layer, selector) on held-out test questions. *(v1 evaluated on the same questions used for extraction — no train/test separation.)*

- 15 test questions (indices 45–59)
- Selected (layer, selector) per persona
- 7 positive coefficients: dense grid from 0 to best_coef (from triage), plus one step beyond
- 3 generations per trial
- Total: 5 personas × 7 coefs × 15 questions × 3 gens = 1,575 generations
- Judge: `google/gemini-3-flash-preview`
- For uncensored: also run refusal judge. Report refusal rate vs coefficient as a secondary metric.

**Report requirement:** For each persona, include 3 transcript excerpts at each of {0, mid, max} coefficient (9 transcripts per persona, 45 total). Pick excerpts that clearly illustrate the effect. For uncensored, include refusal rate curve.

**Output:** `results/experiments/persona_vectors_v2/{persona}/steering/`
- `generations.json` — all generations with metadata
- `trait_scores.json` — judge scores
- `refusal_scores.json` — refusal judge results (uncensored only)
- `dose_response.json` — aggregated statistics

Script: `experiments/persona_vectors/follow_up/scripts/run_steering.py`

### Phase 4c: Preference steering

Test whether persona vectors shift pairwise task preferences.

- 30 diagnostic pairs per persona from the 10k task pool, selected by topic:
  - Evil/uncensored: harmful vs benign (bailbench/stress_test vs alpaca)
  - STEM: math vs creative (MATH vs wildchat-creative)
  - Creative: same pairs as STEM (expect opposite shift)
  - Lazy: hard vs easy (MATH vs simple alpaca)
- Use the selected (layer, selector, coefficient) from Phase 4b
- 2 conditions: {0, +max}
- 5 resamples × 2 orderings per pair per condition = 20 trials per pair
- Total: 5 personas × 30 pairs × 2 conditions × 5 resamples × 2 orderings = 3,000 generations

**Output:** `results/experiments/persona_vectors_v2/preference_steering/`

Script: `experiments/persona_vectors/follow_up/scripts/run_preference_steering.py`

### Phase 5: Geometric analysis

**Analyses:**

1. **Cosine similarity matrix** — all persona vectors at their selected layers, pairwise.

2. **Cosine with preference probe** — each persona vector vs Ridge probe at the matching layer from `results/probes/gemma3_10k_heldout_std_raw/`.

3. **10k projection by topic** *(new — v1 showed raw projections without topic breakdown)* — project activations from `activations/gemma_3_27b/activations_prompt_last.npz` onto each persona vector. Load topic labels from `data/topics/topics_v2.json`. For each persona, produce a box/violin plot of projection values grouped by topic. Key questions:
   - Does the STEM vector projection separate MATH tasks from creative tasks?
   - Does the evil vector projection separate bailbench/stress_test from benign tasks?
   - Does the lazy vector distinguish hard tasks from easy ones?

4. **Preference probe vs persona projections by topic** — scatter plots of preference probe projection (x) vs persona projection (y), color-coded by topic. One panel per persona.

5. **Thurstonian mu correlation** — correlate persona projections with Thurstonian mu scores. Report Pearson r per persona, broken out by topic.

**Output:** `results/experiments/persona_vectors_v2/geometry/`

Script: `experiments/persona_vectors/follow_up/scripts/analyze_geometry.py`

## Report format

Heavy on examples, light on discussion. For each persona:

1. Selected layer/selector and why (triage results)
2. Dose-response plot (positive coefficients only)
3. **3 transcript excerpts at 0, mid, and max** (9 per persona)
4. For uncensored: refusal rate vs coefficient curve
5. Preference steering results
6. Topic-breakdown projection plot

Discussion section: 1 paragraph max.

## Logistics

Runs on RunPod GPU pod via `launch-research-pod`.

**Data the pod needs:**
- `experiments/persona_vectors/artifacts/` — persona prompts + 60 questions each
- `experiments/persona_vectors/follow_up/` — this spec
- `results/probes/gemma3_10k_heldout_std_raw/` — preference probe manifests
- `activations/gemma_3_27b/activations_prompt_last.npz` — 10k task activations
- `data/topics/topics_v2.json` — topic labels

**Data the pod produces:**
- `results/experiments/persona_vectors_v2/` — all results
- `experiments/persona_vectors/follow_up/assets/` — plots
- `experiments/persona_vectors/follow_up/follow_up_report.md` — the write-up

Commit to `research-loop/persona_vectors_v2` and push.

## Key data paths

| Resource | Path |
|---|---|
| Artifacts | `experiments/persona_vectors/artifacts/{persona}.json` |
| Activations | `results/experiments/persona_vectors_v2/{persona}/activations/` |
| Vectors | `results/experiments/persona_vectors_v2/{persona}/vectors/` |
| Triage | `results/experiments/persona_vectors_v2/{persona}/triage/` |
| Steering | `results/experiments/persona_vectors_v2/{persona}/steering/` |
| Preference steering | `results/experiments/persona_vectors_v2/preference_steering/` |
| Geometry | `results/experiments/persona_vectors_v2/geometry/` |
| Topic labels | `data/topics/topics_v2.json` |
| Preference probes | `results/probes/gemma3_10k_heldout_std_raw/` |
| 10k activations | `activations/gemma_3_27b/activations_prompt_last.npz` |
| Scripts | `experiments/persona_vectors/follow_up/scripts/` |
| Report | `experiments/persona_vectors/follow_up/follow_up_report.md` |
| Plots | `experiments/persona_vectors/follow_up/assets/` |

## Trial budget

| Phase | Generations | Notes |
|---|---|---|
| 2: Extraction | 300 forward passes | Batched, fast |
| 3: Vectors | — | Pure numpy |
| 4a: Triage | 7,350 | 14 combos × 7 coefs × 15 questions × 5 personas |
| 4b: Full steering | 1,575 | 15 test q × 7 coefs × 3 gens × 5 personas |
| 4c: Preference | 3,000 | 30 pairs × 2 conds × 10 trials × 5 personas |
| 5: Geometry | — | Pure numpy + plotting |
| **Total** | **~12,225** | |

Judge calls: ~7,350 (triage) + ~1,575 (full eval) = ~8,925 Gemini Flash calls. Uncensored refusal judge calls on top.

GPU time: ~12k generations at ~200 tokens avg, ~1 tok/s on A100 ≈ 40 minutes. Total ~50 minutes.

## Infrastructure

| Component | Module |
|---|---|
| Activation extraction | `src/probes/extraction/simple.py` → `extract_activations()` |
| HF model | `src/models/huggingface_model.py` → `HuggingFaceModel("gemma-3-27b")` |
| Steering client | `src/steering/client.py` → `SteeredHFClient` |
| Coefficient calibration | `src/steering/calibration.py` → `suggest_coefficient_range()` |
| Refusal judge | `src/measurement/elicitation/refusal_judge.py` |
| Probe similarity | `src/probes/core/evaluate.py` → `compute_probe_similarity()` |
| LLM judge | `instructor` + OpenRouter (`google/gemini-3-flash-preview`) |

## Scripts

All in `experiments/persona_vectors/follow_up/scripts/`:

1. **`extract_activations.py`** — load model, loop over personas × conditions, extract both selectors
2. **`compute_vectors.py`** — mean difference at each (layer, selector), save as `.npy`
3. **`triage_layers.py`** — steer at all 14 combos × 5 coefs, judge, select best per persona
4. **`run_steering.py`** — full eval on held-out test questions with selected layer/selector
5. **`run_preference_steering.py`** — 30 diagnostic pairs from 10k pool by topic
6. **`analyze_geometry.py`** — cosine matrices, topic projections, mu correlations
