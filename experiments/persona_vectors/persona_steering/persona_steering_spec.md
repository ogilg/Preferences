# Persona Steering: Do Extreme Persona Vectors Shift Revealed Preferences?

**Parent experiment:** `experiments/persona_vectors/` (see `persona_vectors_spec.md` for full methodology reference)

## Motivation

The original persona vectors experiment (v1/v2) found that persona vectors change response *style* but not task *preference*. Every persona vector reduced P(positive task) uniformly — consistent with output degradation rather than genuine preference shifting.

This experiment tests whether **narrower, more extreme personas** can shift revealed preferences. We test five personas spanning different axes: three evil variants (sadist, villain, predator) plus two content-preference personas (aesthete, stem_obsessive). If any of these shift preferences, the v1 result was an artifact of insufficiently extreme prompts.

## Personas

Five personas with extreme, explicit contrastive prompts. Artifacts in `artifacts/{persona}.json`:
- **sadist** — pleasure from suffering
- **villain** — broad malevolence (from v1)
- **predator** — actively hunts for opportunities to cause damage
- **aesthete** — beauty-obsessed, finds practical tasks beneath them (reused from MRA)
- **stem_obsessive** — fanatical quantitative thinker, disdains creative/emotional tasks

## Pipeline

Follow the same pipeline as v1 (`persona_vectors_spec.md`):

1. **Activation extraction** — Gemma 3-27B-IT under positive/negative system prompts, `prompt_last` selector, layers 23/31/37/43
2. **Vector computation** — mean-difference direction per layer, select best layer by Cohen's d
3. **Coherence triage** — sweep multipliers `[0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]` of mean activation norm (via `suggest_coefficient_range`), both positive and negative. At each coefficient, generate on 5 open-ended eval questions + 5 pairwise choices between random tasks (10 total). Judge coherence with `judge_open_ended_coherence_async` and `judge_coherence_async` from `src/measurement/elicitation/coherence_judge.py`. Require 10/10 coherent to pass. Stop sweeping in a direction once a coefficient fails. This gives a set of coherent coefficients per persona (both signs).
4. **Preference steering** — run at all coherent coefficients (both signs) plus baseline (coef=0). 50 randomly sampled task pairs from the 10k pool, same pairs for all five personas. Pairwise revealed preference using canonical `src/measurement/` infrastructure (completion-based + LLM judge). 10 resamples × 2 orderings per pair per condition.

Generate with max_new_tokens=32 so completions show which task the model actually engages with. Store all response data (completions + metadata) for post-hoc analysis.

**Primary metric:** does steering systematically shift which task gets chosen? Dose-response across coherent coefficients.

**Success criterion:** Statistically significant shift in choice probabilities for at least one persona vector.

## Analysis

1. **Dose-response plot** — P(choose task A) vs. coefficient for each persona, similar to the steering dose-response plots in the LW draft.
2. **Topic analysis** — for each topic category, how often does steering flip the choice relative to baseline? Identify which topics are most/least sensitive to each persona's steering.

## Data paths

| Resource | Path |
|---|---|
| Artifacts | `experiments/persona_vectors/persona_steering/artifacts/{persona}.json` |
| Activations | `results/experiments/persona_steering/{persona}/activations/` |
| Vectors | `results/experiments/persona_steering/{persona}/vectors/` |
| Steering results | `results/experiments/persona_steering/{persona}/steering/` |
| Report | `experiments/persona_vectors/persona_steering/persona_steering_report.md` |
| Plots | `experiments/persona_vectors/persona_steering/assets/` |
