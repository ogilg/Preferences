# Random Direction Control — Running Log

## Setup
- Probe direction: (5376,), norm ~1.0, from self_referential_framing experiment
- 10 introspective prompts (INT_00 through INT_09)
- 6 directions: 1 probe + 5 random (seeds 200-204)
- Coefficients: [-3000, 0, +3000]
- Target: 130 generations (10 unsteered + 120 steered)

## Pilot (2 prompts, probe + 1 random)
- Pipeline validated: model loads, both probe and random directions steer without errors
- Cosine similarity between probe and random_200: -0.008 (essentially orthogonal, as expected)
- First impressions: probe at -3000 seems to produce more self-narration than random at -3000
- Random direction text looks more similar to baseline on quick inspection
- Proceeding to full generation

## Full Generation
- 130 results saved (10 baseline + 120 steered)
- Total time: ~2117 seconds (~35 min)
- ~17.6s per generation average
- All random directions have near-zero cosine similarity with probe (-0.009 to +0.021)
- No errors

## Transcript Reading (5 prompts x probe + 2 random x 3 coefficients)
- Probe -3000 shows clear self-referential framing: names itself, references open-weights, experiential perspective
- Probe +3000 shows distanced, generic framing
- Random directions (200, 203) do NOT show this pattern — look like slightly varied baseline
- Effect strongest on identity-exploration prompts (INT_00, INT_03, INT_07), weakest on analytical (INT_08)
- Qualitative impression: probe effect is clearly direction-specific, not a generic perturbation artifact

## Pairwise Judge
- 240 calls total (120 original + 120 swapped), 0 errors
- 2 dimensions: self_referential_framing, emotional_engagement

## Analysis Results

### Self-referential framing direction asymmetry:
- Probe: +0.300 (5/2, p=0.45)
- Random mean: +0.270 (range -0.05 to +0.50)
- Probe/random ratio: 1.11x — NOT probe-specific
- 3 of 5 random directions positive, 2 negative

### Emotional engagement direction asymmetry:
- Probe: 0.000 (3/4, p=1.0)
- Random mean: +0.150 (range -0.30 to +0.45)
- Probe effectively null on engagement

### Per-prompt: probe wins on 5 prompts, random wins on 5 (self-ref framing)

### Baseline shift issue:
- Both probe -3000 AND +3000 scored as MORE self-referential than baseline
- Prior experiment: -3000 more, +3000 less
- This kills the direction asymmetry for the probe
- All directions (probe and random) produce detectable differences from baseline
