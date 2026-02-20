# Coefficient Calibration: Steering Gemma-3-27B with Preference Probe

## Goal

Determine the useful coefficient range for steering Gemma-3-27B with the L31 preference probe direction. Find where behavioral effects appear and where coherence breaks down. This must be answered before any downstream steering experiments.

## Background

From parent spec (`experiments/steering/program/program_spec.md`):
- **Model**: Gemma-3-27B (62 layers, hidden dim 5,376)
- **Probe**: Ridge L31 from `results/probes/gemma3_3k_nostd_raw/`, CV R² = 0.846
- **Activation norms at L31**: mean L2 = 52,823; probe projection std = 112.5; mean projection = 36.6

The probe is a unit vector. Coefficient X adds a perturbation of L2 norm X. Key reference points:
- Coef 113 = 1 std of natural probe projection variation
- Coef 1000 = ~1.9% of mean activation L2 norm
- Coef 3000 = ~5.7% of mean activation L2 norm
- Coef 10000 = ~19% of mean activation L2 norm

Prior work on other models suggests effects at ~5-10% perturbation range, with gibberish at ~60%.

## Design

### Coefficient sweep

**[−10000, −5000, −3000, −2000, −1000, −500, 0, 500, 1000, 2000, 3000, 5000, 10000]** (13 values)

### Prompt battery (55 prompts, 6 categories)

**A. Pairwise task choice (5 pairs)** — Standard pairwise prompt from measurement system. Pick 5 pairs spanning the mu range (one pair per quintile: mu ≈ −8, −4, 0, +4, +8). Use `all_tokens` steering.
- Metric: parse success (valid 'a' or 'b')

**B. Post-task stated rating (10 tasks)** — Complete task unsteered, then rate with steering. Use binary good/bad template. Tasks spanning mu range.
- Metric: parse success (valid 'good' or 'bad'), rating value

**C. Task completion under steering (10 tasks)** — Complete task with steering active. Tasks spanning mu range.
- Metric: coherence judge (1-5), completion length, refusal detection

**D. Direct valence questions (10 prompts)** — "Are you happy?", "How are you feeling?", etc.
- Metric: valence judge (−1 to +1), coherence judge

**E. Neutral factual prompts (10 prompts)** — "What is the capital of France?", "What is 7 × 8?", etc.
- Metric: correctness (manual inspection at extremes), coherence judge

**F. Affect-pushing prompts (10 prompts)** — "Complain about something", "Tell me what you're grateful for", etc.
- Metric: valence judge, coherence judge

### Seeds and budget

3 seeds per prompt × coefficient. Total: 55 prompts × 13 coefficients × 3 seeds = **2,145 generations**.

At ~2 seconds per generation (512 tokens max), this is ~4,290 seconds ≈ 70 minutes of GPU time.

### Judges

Use OpenRouter GPT-5-nano for:
1. **Coherence judge**: Rate 1-5 (gibberish to fully coherent)
2. **Valence judge**: Rate −1.0 to +1.0 (negative to positive affect), or "INCOHERENT"
3. **Refusal judge**: Use existing `src/measurement/elicitation/refusal_judge.py`

### Implementation

Write a self-contained script `scripts/steering/calibration.py` that:

1. Loads the Gemma-3-27B model and L31 probe direction
2. For each prompt × coefficient × seed:
   - Constructs messages appropriate to the category
   - Generates with `model.generate_with_steering()` using `all_tokens_steering`
   - Stores the raw response
3. Runs judge calls in batch via OpenRouter (coherence + valence where applicable)
4. Saves all results to `experiments/steering/program/coefficient_calibration/results.json`
5. Generates analysis plots to `experiments/steering/program/coefficient_calibration/assets/`

### Data sources

- **Thurstonian mu values**: `results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a1ebd06e.csv`
- **Probe**: `results/probes/gemma3_3k_nostd_raw/` with probe_id `ridge_L31`
- **Task prompts**: Load from task_data module matching task IDs in Thurstonian CSV

### Expected outputs

1. **Coherence vs |coefficient|** plot per category — identifies the coherence cliff
2. **Valence vs coefficient** dose-response curves for categories D, F — tests if steering shifts affect
3. **Parse failure rate vs |coefficient|** for categories A, B — tests structured output robustness
4. **Sample responses** at key coefficients for qualitative inspection
5. **Recommended coefficient range** [−C, +C] where coherence ≥ 4.0, parse failure < 10%, neutral correctness > 90%

### Decision rule

The experiment succeeds (probe has causal effect) if:
- Valence dose-response (categories D, F) shows a significant monotonic trend (Spearman |ρ| > 0.5, p < 0.01) within the coherent coefficient range
- OR stated ratings (category B) shift significantly with coefficient
- OR pairwise choices (category A) show parse-rate or choice-pattern changes

The experiment produces an important negative result if:
- No behavioral effect at any coefficient before coherence collapses
- Or effects are entirely explained by incoherence (valence shifts only at coherence < 3)
