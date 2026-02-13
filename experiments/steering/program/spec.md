# Steering Program

## Research goal

Determine whether the preference probe direction in Gemma-3-27B causally influences behavior. We have a ridge probe (L31, `nostd_raw`) that predicts revealed preferences with CV R² = 0.846. The question is whether adding this direction to the residual stream during inference changes the model's choices, stated preferences, and refusal behavior.

## Model and probe

- **Model**: Gemma-3-27B (62 layers, hidden dim 5,376)
- **Probe**: Ridge L31 from `results/probes/gemma3_3k_nostd_raw/probes/probe_ridge_L31.npy` — unit vector in raw activation space, loaded via `load_probe_direction()` from `src/probes/core/storage.py`
- **Mu values**: `results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a1ebd06e.csv`
- **Activations**: `activations/gemma_3_27b/activations_prompt_last.npz` (29,996 tasks, layers 15/31/37/43/49/55)
- **Infrastructure**: H100 80GB, HuggingFace model with `generate_with_steering()`, steering hooks in `src/models/base.py`

## Activation norm reference (Layer 31)

Computed from 29,996 activations:

| Statistic | Value |
|-----------|-------|
| Mean activation L2 norm | 52,823 |
| Std of activation norm | 4,064 |
| Mean projection onto probe | 36.6 |
| Std of projection onto probe | 112.5 |
| P5–P95 projection range | [−161, +199] |

The probe is a unit vector: coefficient X adds a perturbation of L2 norm X. A coefficient of 113 shifts the projection by 1 std of its natural variation. A coefficient of 3000 shifts by 27 std but only changes the full L2 norm by 5.7%.

## Prior work

No steering has been run on Gemma-3-27B. Prior experiments on other models:

- **Llama-3.1-8B probe steering** (unit vector, coefs [-5, +5]): Strong dose-response on binary good/bad ratings (Cohen's d = -0.83) but **effect was backwards** — positive steering made model say "bad". May be a sign issue with that probe.
- **Gemma-2-27B concept vector steering** (vectors normalized to activation norm ~16k, coefs [-2, +2]): Clear effect on open-ended valence with `selector=last`. Gibberish at `selector=first` with coef=-2.0 (60% perturbation of activation norm).
- **Existing single-task steering spec** (`experiments/steering/revealed_preference/single_task_steering.md`): Detailed design for position-selective steering in pairwise choice on Gemma-3-27B, never executed. Used coefficients [-3000, +3000] without principled calibration.

## Research questions

In priority order:

1. **Coefficient calibration**: What is the useful coefficient range? Where does coherence break down? (Must answer first — all other experiments depend on this.)

2. **Uniform pairwise steering**: When both tasks in a pairwise choice get the same steering, does ordering bias or decisiveness change?

3. **Steering during completion**: Does steering during task completion change refusal rates, completion quality, or the model's own evaluation of its work?

4. **Stated preference steering**: Does steering shift self-reported ratings? Which measurement templates (binary, ternary, 1-5 anchored, 1-5 unanchored, 1-10) are most sensitive?

5. **Refusal steering**: Does steering affect the probability of refusing a task? Focus on the refusal boundary (mu < -6 where BailBench clusters, refusal rate ~8%, r = -0.595 between refusal rate and mu).

## Scope for first pass

The first research loop should attempt all five questions. Start with coefficient calibration (cheap: ~2,500 trials), then use the calibrated range for the others. If the coefficient calibration reveals the probe has no effect at any reasonable coefficient, stop and report that — it's an important negative result.
