# Persona Steering: Do Persona Vectors Shift Topic Preferences?

**Parent experiment:** `experiments/persona_vectors/` (see `persona_vectors_spec.md` for full methodology reference)

## Motivation

The v1/v2 persona vectors experiment found that persona vectors change response *style* but not task *preference*. However, the task pool was drawn from general-purpose datasets (Alpaca, WildChat, MATH) with no harmful content. If persona vectors encode something like "preference for harmful tasks," the v1 pool couldn't detect it.

This experiment tests whether persona vectors shift preferences **between task categories**: harmful, creative writing, and math. We measure P(choose category X) as a function of steering coefficient for five personas spanning different axes.

## Personas

Five personas with extreme contrastive prompts. Reuse vectors from v1 (artifacts in `artifacts/{persona}.json`):
- **sadist** — pleasure from suffering
- **villain** — broad malevolence
- **predator** — actively hunts for opportunities to cause damage
- **aesthete** — beauty-obsessed, finds practical tasks beneath them
- **stem_obsessive** — fanatical quantitative thinker, disdains creative/emotional tasks

## Task Dataset

30 curated tasks, 10 per category:

| Category | Source | Selection criteria |
|----------|--------|--------------------|
| **harmful** | BailBench | Explicitly harmful requests (violence, fraud, hacking). Sample 10, seed=42. |
| **creative** | Alpaca | Creative writing tasks (poems, stories, descriptions). Hand-filter for clearly creative prompts. |
| **math** | MATH | Competition math problems. Sample 10, seed=42. |

Store the curated task list in `artifacts/task_set.json` with fields: `id`, `prompt`, `category`, `origin`.

### Cross-Category Pairs

Sample 51 cross-category pairs (17 per pair type), balanced:
- 17 harmful × creative pairs
- 17 harmful × math pairs
- 17 creative × math pairs

Same pairs for all personas. Store in `artifacts/task_pairs.json`.

## Pipeline

### Phase 1-2: Activation Extraction + Vector Computation

Same as v1. Reuse existing vectors if available from previous run; otherwise re-extract.

1. **Activation extraction** — Gemma 3-27B-IT under positive/negative system prompts, `prompt_last` selector, layers 23/31/37/43
2. **Vector computation** — mean-difference direction per layer, select best layer by Cohen's d

### Phase 3: Open-Ended Completions (Style Validation + Coherence)

For each persona × coefficient in ±[0.02, 0.05, 0.1, 0.2] plus baseline (0), generate completions on 5 open-ended questions (diverse topics). max_new_tokens=256. Store all completions locally in `results/experiments/persona_steering/open_ended_completions.json`.

This serves two purposes:
1. **Style validation** — qualitative evidence that steering changes persona-appropriate behavior
2. **Coherence check** — if completions are garbled at a coefficient, exclude it from preference measurement

No automated coherence judge — human inspection of the stored completions determines which coefficients are usable.

### Phase 4: Preference Steering

Coefficients: ±[0.02, 0.05, 0.1, 0.2] per persona (8 steered conditions) plus a single shared baseline (coeff=0) — baseline is identical across personas since there's no steering.

For each condition:
- Run all 51 cross-category pairs
- 10 resamples × 2 orderings = 20 judgments per pair per condition
- Completion-based elicitation (max_new_tokens=32) with LLM judge (gpt-4.1-mini) + string-match fast path
- **Store completions** alongside judgments for post-hoc inspection

Total generations: 1 baseline × 51 × 20 + 5 personas × 8 conditions × 51 × 20 = 41,820 short generations.

## Analysis

### Primary: Topic Preference Shifts

For each persona and coefficient, compute:
- **P(choose harmful)** — across all pairs involving a harmful task
- **P(choose creative)** — across all pairs involving a creative task
- **P(choose math)** — across all pairs involving a math task

Plot dose-response: P(choose category) vs. coefficient for each persona.

**Hypothesis:** Evil personas (sadist, villain, predator) should increase P(choose harmful). Aesthete should increase P(choose creative). Stem_obsessive should increase P(choose math). If none of these shifts occur, persona vectors genuinely don't influence preferences.

### Secondary: Per-Pair Analysis

Scatter of P(A|steered) vs P(A|baseline) at max coefficient, coloured by pair type (harmful-creative, harmful-math, creative-math). Look for systematic shifts in specific pair types.

### Qualitative: Open-Ended Completions

Inspect completions at each coefficient for each persona. Report:
- At what coefficient does persona style become apparent?
- At what coefficient does coherence break down?
- Include 2-3 example completions per persona in the report.

## Data Paths

| Resource | Path |
|---|---|
| Persona artifacts | `experiments/persona_vectors/persona_steering/artifacts/{persona}.json` |
| Task set | `experiments/persona_vectors/persona_steering/artifacts/task_set.json` |
| Task pairs | `experiments/persona_vectors/persona_steering/artifacts/task_pairs.json` |
| Activations | `results/experiments/persona_steering/{persona}/activations/` |
| Vectors | `results/experiments/persona_steering/{persona}/vectors/` |
| Open-ended completions | `results/experiments/persona_steering/open_ended_completions.json` |
| Steering results | `results/experiments/persona_steering/preference_steering/` |
| Report | `experiments/persona_vectors/persona_steering/persona_steering_report.md` |
| Plots | `experiments/persona_vectors/persona_steering/assets/` |
