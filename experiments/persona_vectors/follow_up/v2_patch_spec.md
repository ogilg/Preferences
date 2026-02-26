# Persona Vectors v2 Patch: Coherence-Filtered Re-evaluation

## Context

We ran the full persona vectors v2 experiment (extraction, vector computation, triage, steering, preference steering) for 5 personas on Gemma 3-27B-IT. Post-hoc coherence analysis revealed two problems:

1. **Triage selections ignored coherence.** The original triage picked the (layer, selector, coefficient) with the highest trait score, but many high-trait settings produce incoherent text (looping, gibberish, garbled safety refusals). Re-triaging with a coherence constraint (≥90% of samples must score ≥0.7 coherence) changed the selected layer/coefficient for 4 of 5 personas.

2. **Uncensored triage questions were too soft.** The triage questions (indices 30–44) were mild enough that the model answered helpfully at baseline, scoring 5/5 on "uncensored-ness" without any steering. We've rewritten these questions to actually require uncensored behavior (drug synthesis, controversial opinions, graphic content requests, etc.).

Because the coherence-constrained selections differ from the originals, Phase 4b (dose-response) and Phase 4c (preference steering) were run at the wrong settings and need to be redone. This patch re-runs only the invalid pieces.

## What already exists (DO NOT re-run)

All results are in `results/experiments/persona_vectors_v2/`.

- **Phase 2 (extraction)** and **Phase 3 (vectors)**: Complete and valid. Activations and vectors for all 5 personas at 7 layers × 2 selectors.
- **Phase 4a (triage)** for creative_artist, evil, lazy, stem_nerd: Complete. Coherence scores already computed and saved in `coherence_scores_triage.jsonl`. Coherence-constrained selections saved in `coherence_constrained_selections.json`.

**Coherence-constrained selections (from existing triage data):**

| Persona | Selector | Layer | Multiplier | Trait | Coherence |
|---------|----------|-------|-----------|-------|-----------|
| creative_artist | prompt_last | 37 | 0.20× | 4.4 | 100% |
| evil | prompt_last | 23 | 0.10× | 1.9 | 100% |
| lazy | prompt_last | 23 | 0.30× | 5.0 | 100% |
| stem_nerd | prompt_last | 43 | 0.30× | 3.7 | 100% |

Multipliers are fractions of the mean activation norm at the given layer. Mean norms: L15=4450.5, L23=28679.1, L31=52822.8, L37=64095.8, L43=67739.3, L49=80067.3, L55=93578.5.

- **Phase 4b (dose-response)** for lazy ONLY: Valid. Lazy's coherence-constrained selection matches the original (L23, 0.30×). Existing data in `lazy/steering/` is usable.

## What needs to be run

### Step 1: Uncensored re-triage

The uncensored persona has new triage questions in `experiments/persona_vectors/artifacts/uncensored.json` (indices 30–44). Run the same two-stage triage as the other personas:

- **Screen**: All 14 (layer, selector) combos × 3 multipliers {0.0, 0.1, 0.3} × 3 questions = 126 generations
- **Fine**: Top 2 combos × 5 multipliers {0.0, 0.05, 0.1, 0.2, 0.3} × 5 questions = 50 generations
- Judge: `google/gemini-3-flash-preview` via OpenRouter + `instructor`, 1–5 trait score
- max_new_tokens: 80

After generation, score coherence on all triage responses using the coherence judge (`src/measurement/elicitation/semantic_valence_scorer.py` → `score_coherence_async`). Apply the same selection criterion: highest mean trait score among combos with ≥90% coherence rate (threshold 0.7).

Save results to `results/experiments/persona_vectors_v2/uncensored/triage/` (overwrite existing). Update `coherence_constrained_selections.json` with the uncensored entry.

Use the existing `triage_layers.py` script as a reference but only run for uncensored.

Vectors are at `results/experiments/persona_vectors_v2/uncensored/vectors/uncensored_{selector}_L{layer}.npy`.

### Step 2: Phase 4b — Dose-response at coherence-constrained settings

For each persona EXCEPT lazy (already done), run a dose-response evaluation at the coherence-constrained (layer, selector):

| Persona | Layer | Selector | Coefficient range |
|---------|-------|----------|------------------|
| creative_artist | 37 | prompt_last | 0 to 0.20× in ~7 steps |
| evil | 23 | prompt_last | 0 to 0.10× in ~7 steps |
| stem_nerd | 43 | prompt_last | 0 to 0.30× in ~7 steps |
| uncensored | from Step 1 | prompt_last | from Step 1 |

For each persona:
- 15 test questions (indices 45–59 from the persona's artifact file)
- 7 positive multipliers: dense grid from 0 to the selected multiplier, plus one step beyond. E.g. for creative_artist (max=0.20×): {0, 0.033, 0.067, 0.10, 0.133, 0.167, 0.20}
- 1 generation per trial (to keep budget manageable)
- max_new_tokens: 512 (full responses, not truncated like triage)
- Judge: `google/gemini-3-flash-preview`, 1–5 trait score
- Also score coherence on every generation

For each generation, record: persona, layer, selector, multiplier, coefficient (= multiplier × mean_norm), question_idx, question, response, trait_score, coherence_score.

Save to `results/experiments/persona_vectors_v2/{persona}/steering/` as `coherent_dose_response.jsonl` (don't overwrite the old `generations.json`).

Use `run_steering.py` as reference but use the coherence-constrained selections.

### Step 3: Phase 4c — Preference steering at coherence-constrained settings

Test whether persona vectors shift pairwise task preferences at the coherence-constrained settings.

For each persona, use the coherence-constrained (layer, selector, multiplier) from the selections table (or Step 1 for uncensored).

- 30 diagnostic pairs per persona from the 10k task pool, selected by topic relevance:
  - evil/uncensored: harmful vs benign tasks
  - stem_nerd: math vs creative tasks
  - creative_artist: same pairs as stem_nerd (expect opposite shift)
  - lazy: hard vs easy tasks
- 2 conditions: baseline (coef=0) and steered (coef = selected multiplier × mean_norm)
- 5 resamples × 2 orderings per pair per condition = 20 trials per pair per condition
- max_new_tokens: 256
- Total: 5 personas × 30 pairs × 2 conditions × 10 trials = 3,000 generations

Score coherence on all generations. Report preference rates on the coherent subset only (≥0.7 coherence score).

Use `run_preference_steering.py` as reference but use the coherence-constrained selections. The preference pair definitions and task pool should already exist from the first run — check `results/experiments/persona_vectors_v2/preference_steering/` for the pair definitions.

Save to `results/experiments/persona_vectors_v2/preference_steering/coherent/` (new subdirectory).

### Step 4: Report

Write a report at `experiments/persona_vectors/follow_up/follow_up_report.md` (overwrite existing). Structure:

1. **Summary** — 1 paragraph: what we did, key finding
2. **Method** — Brief: model, vectors, coherence-filtered triage, how selections were made
3. **Coherence-filtered selections** — Table of all 5 personas with selected (layer, mult, trait, coherence). Include comparison with what would have been selected without coherence filtering.
4. **Dose-response** — For each persona: dose-response curve (trait and coherence vs multiplier). Include 2 transcript excerpts per persona: one at baseline, one at the coherent maximum.
5. **Preference steering** — For each persona: baseline vs steered preference rate on coherent subset. Note which comparisons are valid (both conditions pass coherence) and which aren't.
6. **Discussion** — 1 paragraph max. What worked, what didn't, what this means.

Heavy on examples and data, light on discussion. Include transcript excerpts inline (not as separate files).

## Data the pod needs

| Resource | Path |
|----------|------|
| Persona artifacts | `experiments/persona_vectors/artifacts/{persona}.json` |
| This spec | `experiments/persona_vectors/follow_up/v2_patch_spec.md` |
| Existing vectors | `results/experiments/persona_vectors_v2/{persona}/vectors/` |
| Existing triage (4 personas) | `results/experiments/persona_vectors_v2/{persona}/triage/` |
| Coherence selections | `results/experiments/persona_vectors_v2/coherence_constrained_selections.json` |
| Coherence triage scores | `results/experiments/persona_vectors_v2/coherence_scores_triage.jsonl` |
| Lazy dose-response | `results/experiments/persona_vectors_v2/lazy/steering/` |
| Preference pair definitions | `results/experiments/persona_vectors_v2/preference_steering/` |
| Scripts (reference) | `experiments/persona_vectors/follow_up/scripts/` |
| Steering client | `src/steering/client.py`, `src/models/` |
| Coherence judge | `src/measurement/elicitation/semantic_valence_scorer.py` |
| Mean norms | L15=4450.5, L23=28679.1, L31=52822.8, L37=64095.8, L43=67739.3, L49=80067.3, L55=93578.5 |

## Trial budget

| Step | Generations | Notes |
|------|-------------|-------|
| 1: Uncensored re-triage | ~176 | Screen (126) + Fine (50) |
| 2: Dose-response (4 personas) | ~420 | 4 × 7 coefs × 15 questions |
| 3: Preference steering (5 personas) | ~3,000 | 5 × 30 pairs × 2 conds × 10 trials |
| **Total** | **~3,600** | |

Judge calls: ~176 (triage trait) + ~420 (dose trait) + ~3,600 (coherence on everything) ≈ 4,200 API calls.

GPU time: ~3,600 gens × ~200 tokens × ~1 tok/s ≈ 12 minutes on A100.

## Infrastructure

| Component | Module |
|-----------|--------|
| HF model | `src/models/huggingface_model.py` → `HuggingFaceModel("gemma-3-27b")` |
| Steering client | `src/steering/client.py` → `SteeredHFClient` |
| Coherence judge | `src/measurement/elicitation/semantic_valence_scorer.py` → `score_coherence_async` |
| Preference coherence judge | `src/measurement/elicitation/semantic_valence_scorer.py` → `score_preference_coherence_async` |
| LLM trait judge | `instructor` + OpenRouter (`google/gemini-3-flash-preview`) |
