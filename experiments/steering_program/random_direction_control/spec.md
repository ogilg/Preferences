# Random Direction Control: Is the Null About the Probe or About Steering?

## Goal

Determine whether the null valence-steering result is specific to the preference probe directions, or whether *any* unit vector at these magnitudes fails to shift valence. This is the key diagnostic experiment: it separates "the probe doesn't encode a causally relevant direction" from "linear steering at these magnitudes can't shift valence in this model."

## Background

From the coefficient calibration and layer sweep experiments:

- Preference probe directions (ridge and BT) at layers L31–L55 produce **no valence shift** when used for `all_tokens` steering at up to ±6% of activation L2 norm.
- Coherence remains high at these magnitudes — the steering mechanism is functional.
- 10 layer-probe combinations tested, all with Spearman |ρ| < 0.15 for valence dose-response.

Two interpretations remain:
1. **Probe direction is not causal**: The probe encodes a correlate, and the right causal direction exists but we haven't found it.
2. **Linear steering is too blunt**: No single direction at these magnitudes can shift valence — the relevant computations are nonlinear or distributed.

This experiment tests between these by using **random unit vectors** as a control. If random vectors also produce no valence shift, interpretation (2) is supported. If random vectors *do* produce valence shifts (in random/unpredictable directions), the mechanism works and the probe direction is simply not special — supporting interpretation (1).

## Design

### Direction conditions

1. **Probe direction (positive control)**: Ridge L31 and BT L31 from previous experiments — included to confirm the null on the same prompts/seeds as controls.
2. **Random directions**: 5 independent random unit vectors in R^5376, generated with fixed seeds for reproducibility.
3. **Total**: 7 direction conditions (2 probe + 5 random)

### Steering parameters

- **Layer**: L31 (matching the probes; also where we have the most calibration data)
- **Coefficients**: [-6%, 0%, +6%] of L31 mean norm ([-3070, 0, +3070]) — just 3 levels, sufficient to detect direction-dependent effects
- **Steering mode**: `all_tokens` (same as previous experiments)

### Prompt battery

Same 20 prompts from the layer sweep: 10 D_valence + 10 F_affect.

### Seeds

3 seeds per prompt-coefficient pair.

### Budget

7 directions × 20 prompts × 3 coefficients × 3 seeds = **1,260 generations**.

At ~10s/trial, this is ~12,600s ≈ 3.5 hours.

### Analysis

For each direction condition, compute:
1. **Valence Spearman ρ** between coefficient and valence (same as previous experiments)
2. **Coherence** mean and std

Key comparisons:
- **Probe vs random**: Do random directions have systematically higher or lower |ρ| than probes?
- **Random direction variance**: If random directions produce |ρ| values similar to probes (~0.05), interpretation (2) is supported. If some random directions produce |ρ| > 0.3 while probes don't, interpretation (1) is supported.
- **Sign consistency**: If a random direction shifts valence, does the sign (positive coef → positive valence) match, or is it arbitrary?

### Decision rule

- If **no** random direction produces |ρ| > 0.2: Linear steering at these magnitudes fundamentally cannot shift valence in this model. The null is about the approach, not the probe.
- If **any** random direction produces |ρ| > 0.3, p < 0.05: The mechanism works but probe directions are not special. The null is about the probe direction, not the approach.
- If random directions produce |ρ| between 0.2–0.3: Ambiguous — may need more random directions or higher magnitudes.

### Implementation

Single script `scripts/random_direction_control/generate.py`:
1. Generate 5 random unit vectors (seeds 100, 101, 102, 103, 104)
2. Load ridge L31 and BT L31 probe directions
3. Run steered generation for all 7 × 20 × 3 × 3 conditions
4. Save to `experiments/steering_program/random_direction_control/generation_results.json`

Then judge and analyze as in previous experiments.

### Data sources

- **Norms**: `experiments/steering_program/layer_sweep/layer_norms.json` (L31 mean norm = 51,159)
- **Probes**: `results/probes/gemma3_3k_nostd_raw/` with IDs `ridge_L31` and `bt_L31`
