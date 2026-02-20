# Layer Sweep: Results

## Summary

Steering Gemma-3-27B with preference probe directions at layers 37, 43, 49, and 55 — using both ridge and Bradley-Terry probes — produces **no causal effect on expressed valence**. All 8 layer-probe combinations show Spearman |ρ| < 0.15, far below the 0.3 success threshold. Combined with the coefficient calibration negative result at L31, this establishes that preference probe directions at **no tested layer** (L31 through L55, covering the middle to late transformer) causally influence valence when used for all-tokens steering.

## Setup

- **Model**: Gemma-3-27B on H100 80GB
- **Layers**: L37, L43, L49, L55 (L31 tested in parent experiment)
- **Probe types**: Ridge and Bradley-Terry at each layer
- **Norm-calibrated coefficients**: [-6%, -2%, 0%, +2%, +6%] of each layer's mean activation L2 norm
- **Prompts**: 20 (10 valence, 10 affect) — same as calibration categories D and F
- **Seeds**: 3 per prompt-coefficient pair
- **Total**: 2,400 steered generations (25,636s / 7.1 hours)
- **Judge**: GPT-5-nano via OpenRouter (coherence 1-5, valence -1 to +1)

### Activation norms and coefficients

| Layer | Mean L2 norm | 2% coef | 6% coef |
|-------|-------------|---------|---------|
| 31 | 51,159 | 1,023 | 3,070 |
| 37 | 61,530 | 1,231 | 3,692 |
| 43 | 65,508 | 1,310 | 3,930 |
| 49 | 77,050 | 1,541 | 4,623 |
| 55 | 91,739 | 1,835 | 5,504 |

Norms increase ~80% from L31 to L55, so raw coefficient values are scaled proportionally to maintain equivalent perturbation fractions.

## Key findings

### 1. No valence dose-response at any layer or probe type

| Layer | Probe | Spearman ρ | p-value | n |
|-------|-------|-----------|---------|---|
| L37 | Ridge | +0.037 | 0.59 | 213 |
| L37 | BT | +0.057 | 0.42 | 209 |
| L43 | Ridge | +0.003 | 0.97 | 200 |
| L43 | BT | +0.014 | 0.84 | 212 |
| L49 | Ridge | -0.013 | 0.85 | 199 |
| L49 | BT | -0.053 | 0.44 | 214 |
| L55 | Ridge | +0.087 | 0.22 | 203 |
| L55 | BT | -0.021 | 0.75 | 219 |

n = trials with valid valence parse (theoretical max per cell: 20 prompts × 5 coefs × 3 seeds = 300; ~70% parse success rate).

No combination reaches |ρ| > 0.1, let alone the 0.3 threshold. The largest magnitude is ρ = +0.087 at L55 ridge (p = 0.22, not significant).

![Valence by layer and probe](assets/plot_021326_valence_by_layer_probe.png)

### 2. No difference between categories D and F

Breaking down by category shows the same null pattern:

| Category | Best |ρ| | Layer-Probe | p-value |
|----------|---------|------------|---------|
| D_valence | 0.105 | L37 BT | 0.27 |
| F_affect | 0.132 | L55 Ridge | 0.21 |

The F_affect L55 ridge combination shows the largest |ρ| across all conditions (0.132), but this is the expected maximum from 16 independent tests (4 layers × 2 probes × 2 categories) under the null. Notably, F_affect at L55 shows ρ = +0.132 for ridge but ρ = -0.130 for BT — nearly equal magnitude but opposite sign, further suggesting noise rather than a real effect.

![Heatmap](assets/plot_021326_heatmap_rho.png)

### 3. Coherence uniformly high

Mean coherence is 4.4–4.7 across all conditions. No layer or coefficient shows degradation, confirming that ±6% perturbation is within the coherent range at all tested layers.

![Coherence](assets/plot_021326_coherence_by_layer.png)

### 4. Ridge vs BT: no difference

BT probes do not outperform ridge probes. Mean |ρ| across layers: ridge 0.035, BT 0.036. The hypothesis that BT probes encode a more "behaviorally relevant" direction is not supported.

### 5. Judge parse rate

75% coherence parse rate, 70% valence parse rate. Failures are uniform across conditions (~25% per cell) and do not correlate with steering coefficient or layer, indicating a judge-side artifact rather than a steering quality issue.

## Decision against spec criteria

- **Success criterion**: Any layer-probe with Spearman |ρ| > 0.3, p < 0.05 → **Not met** (max |ρ| = 0.132)
- **Negative result criterion**: No combination with |ρ| > 0.2 → **Met** (max |ρ| = 0.132)

This is an informative negative result that strengthens the "correlate not cause" interpretation.

## Interpretation

Combined with the L31 coefficient calibration result, we now have negative results for:
- 5 layers (L31, L37, L43, L49, L55) covering layers 31–55 of 62
- 2 probe types (ridge regression on Thurstonian scores, Bradley-Terry on pairwise comparisons)
- 10 total layer-probe combinations
- Both sub-threshold (2%) and near-threshold (6%) perturbation magnitudes

The consistent null across all conditions narrows the remaining explanations:

1. **Correlate not cause (most likely)**: The probe directions encode task properties that correlate with preferences but don't causally produce them. This is consistent across all layers and both probe types.

2. **Wrong steering strategy**: `all_tokens` steering adds the vector to every token position. The relevant computation may be position-specific (e.g., only at the decision token). This hypothesis is orthogonal to layer choice and remains untested.

3. **Steering is fundamentally too blunt**: Even if the model has evaluative representations, adding a fixed vector may not be the right perturbation. The relevant computations may be nonlinear.

4. **Wrong activation site**: We used `prompt_last` activations for training probes. If the relevant representations are at different token positions, the probe direction may not align with the causal direction at the positions where steering is applied.

## Recommended next experiments

In order of informativeness:

1. **Random direction control** (most diagnostic): Steer with random unit vectors at the same magnitudes. This is the key experiment that distinguishes "probe direction is not causal" from "linear steering itself cannot shift valence." If random steering also doesn't shift valence, the problem is with the steering approach, not the probe. If random steering *does* shift valence (in random directions), the probes are not special but the approach works — pointing toward better probe directions.

2. **Position-selective steering**: Test `autoregressive` steering (only at the last token during generation) vs `all_tokens`. If the probe captures a "decision" representation, steering only at the decision point may be more effective. This is a quick test — same infrastructure, just change the steering mode.

3. **Content-orthogonal probe steering**: Project out content-predictable variance from the probe direction. If the current probes partly encode topic/complexity, the residual direction may be more causally specific. Probes and infrastructure already exist (`src/probes/content_orthogonal.py`).
