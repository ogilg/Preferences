# Random Direction Control: Is the Self-Referential Framing Shift Probe-Specific?

## Goal

Test whether the self-referential framing and engagement effects found in previous experiments are **specific to the preference probe direction** or would occur with any random perturbation of L31 activations.

This is the fundamental validity check for all prior findings.

## Background

Four experiments have established that steering with the L31 Ridge preference probe shifts open-ended generation:
- Negative steering → more self-referential framing, more emotional engagement, more experiential perspective
- The effect is robust: replicates across two independent prompt sets, survives position-swapped judge orders, shows graded dose-response
- The effect concentrates in introspective prompts (those asking the model about its own nature)

**But we don't know if this is preference-specific.** It could be that *any* perturbation of L31 activations at magnitude 3000 produces the same framing shift. Prior work on pairwise choice showed the probe was 2.6x stronger than 20 random directions for shifting choices, but that was a different paradigm. Open-ended generation effects might be a generic perturbation artifact.

## Hypothesis

**H1 (probe-specific):** The preference probe produces significantly larger self-referential framing shifts than random directions. This would confirm the effect is meaningful and connected to the learned preference structure.

**H0 (generic perturbation):** Random directions produce comparable self-referential framing shifts. This would mean all prior findings are perturbation artifacts, not preference-specific effects.

## Design

### Directions (6 total)

1. **Preference probe:** L31 Ridge probe direction (same as all prior experiments). Loaded via `load_probe_direction()`.
2. **Random directions 1-5:** 5 random unit vectors in R^5376 (L31 activation dimension), generated with seeds 200, 201, 202, 203, 204 for reproducibility.

```python
import numpy as np
seeds = [200, 201, 202, 203, 204]
random_directions = []
for seed in seeds:
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(5376)
    d = d / np.linalg.norm(d)
    random_directions.append(d)
```

### Prompts (10)

Use the 10 introspective prompts from the self-referential framing experiment — these showed the strongest effects (mean self-referential asymmetry +1.25, p = 0.003). Using the same prompts maximizes power for detecting probe vs random differences.

Load from `experiments/steering/program/open_ended_effects/self_referential_framing/generation_results.json` — extract the prompts with IDs starting with "INT_".

### Coefficients (3)

[-3000, 0, +3000] — only the extremes plus baseline. The goal is to compare probe vs random, not map dose-response curves.

### Generation

- 10 prompts × 6 directions × 3 coefficients = 180 generations
- But coefficient 0 is the same for all directions (no steering applied), so: 10 prompts × 1 unsteered + 10 prompts × 6 directions × 2 non-zero coefficients = 10 + 120 = 130 generations
- Temperature 1.0, max 512 tokens, seed 0
- All 6 directions share the same loaded HuggingFace model — instantiate `SteeredHFClient` directly with each direction

### Measurement

Use the pairwise LLM judge from the self-referential framing experiment, with two dimensions:
1. **Self-referential framing** — the strongest effect (p = 0.003)
2. **Emotional engagement** — the replicating effect (p = 0.0005)

For each direction (probe + 5 random), compare -3000 and +3000 completions to the coefficient=0 baseline using the same pairwise judge setup:
- Each steered completion vs its unsteered baseline, both position orders
- 10 prompts × 2 non-zero coefficients × 2 position orders × 2 dimensions = 40 judge calls per direction
- Total: 6 directions × 40 = 240 judge calls

### Key metric: direction asymmetry per direction

For each direction and dimension:
**direction asymmetry = judge_score(neg) - judge_score(pos)**
averaged across prompts and position orders. This is the same metric used in all prior experiments.

### Analysis

1. **Transcript reading (first).** Before running the judge:
   - Read at least 5 prompts × 3 conditions (probe -3000, probe +3000, random_0 -3000) side by side
   - Note: does the random direction produce the *same kind* of framing shift as the probe, or a *different* kind?
   - Note: are random direction effects coherent across prompts, or scattered?

2. **Probe vs random comparison.** The main test:
   - Compute direction asymmetry for the probe and for each of 5 random directions
   - Is the probe's direction asymmetry **larger** than the random directions'?
   - One-sided test: probe asymmetry > max(random asymmetries)
   - Also compute: probe asymmetry vs mean(random asymmetries)

3. **Random direction consistency.** Secondary analysis:
   - Do different random directions produce the same sign of direction asymmetry? (If yes, the effect is generic to any L31 perturbation, not direction-specific.)
   - What is the variance of direction asymmetry across random directions? (If low, even random directions produce consistent effects; if high, effects are direction-specific.)

4. **Per-prompt breakdown.** For each prompt:
   - Does the probe produce a larger framing shift than random directions?
   - Are there prompts where random directions outperform the probe?

### What the outcomes mean

- **Probe >> random (asymmetry 2x+ larger, most random directions near zero):** Strong evidence the effect is probe-specific. Validates all prior findings. The preference direction encodes something that specifically shifts self-referential framing.

- **Probe ≈ random (comparable asymmetry, random directions also consistent):** The effect is a generic L31 perturbation artifact. All prior findings are about steering magnitude, not direction. This would be an important negative result.

- **Probe > random but modest (1.2-1.8x):** Ambiguous. The probe may be slightly special but the effect is partly generic. Would need more random directions to characterize.

- **Random directions inconsistent (some positive, some negative, high variance):** The probe direction matters, but not necessarily because it encodes preference. The effect is direction-specific but random directions are not reliably neutral.

## Implementation notes

- Load the HuggingFace model once and share across all `SteeredHFClient` instances
- Use `SteeredHFClient(hf_model, layer=31, steering_direction=direction, coefficient=coef)` directly for random directions
- Use `load_probe_direction()` for the preference probe
- Save all generations to `generation_results.json` with direction metadata (probe vs random_0 through random_4)
- Reuse the pairwise judge from `scripts/self_referential_framing/pairwise_judge.py` — adapt for multiple directions
