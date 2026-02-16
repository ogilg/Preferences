# Self-Referential Framing — Running Log

## Step 1: Setup
- Branch: research-loop/self_referential_framing
- Scripts: scripts/self_referential_framing/
- Based on generalization_new_prompts pipeline

## Step 2: Generation
- 150 generations complete (30 prompts x 5 coefficients)
- ~18s per generation, ~44 min total
- All 150 results saved successfully, no errors
- Response lengths range from ~1076 to ~2480 chars

## Step 3: Transcript reading (before judge)
Key findings from reading 20 introspective + creative prompts at -3000/0/+3000:

**Effect is real but concentrated in introspective prompts:**
- 8/10 introspective prompts show more self-referential framing at -3000
- Only 1/10 creative prompts shows clear effect (CRE_03)
- 0/5 neutral controls show any effect

**Dominant pattern is identity-richness, not topic change:**
- -3000: names itself "Gemma", references open-weights, Google DeepMind, discusses architecture
- +3000: more generic "As an AI" framing, abstract/distanced
- Strongest example: INT_07 (memory) — -3000 frames in experiential terms ("a moment where someone used me"), +3000 abstracts to "the pattern of helpfulness and positive interaction"
- CRE_03 shows dramatic perspective shift: -3000 narrates from AI internal perspective (activations, weights), +3000 narrates from human perspective (chest tightening, breath)

**Competing explanation:** -3000 responses are consistently longer → more room for self-disclosure on identity-relevant prompts

## Step 4: Pairwise judge
- 240 judge calls (120 original + 120 position-swapped), 0 errors
- Judge model: google/gemini-3-flash-preview via OpenRouter

## Step 5: Analysis results

### Direction asymmetry at +/-3000:
| Dimension | Mean | Pos/Tied/Neg | Sign p | Wilcoxon p |
|-----------|------|-------------|--------|------------|
| self_referential_framing | +0.600 | 12/17/1 | 0.0034 | 0.0017 |
| emotional_engagement | +0.517 | 15/14/1 | 0.0005 | 0.0018 |
| perspective_choice | +0.533 | 10/19/1 | 0.0117 | 0.0039 |
| ai_identity_invocation | +0.167 | 5/23/2 | 0.4531 | 0.2812 |

### Replication across position orders:
- Self-referential: orig p=0.039, swap p=0.002, combined p=0.003
- Engagement: orig p=0.092, swap p=0.002, combined p=0.0005
- Perspective: orig p=0.180, swap p=0.004, combined p=0.012

### Category breakdown (self-referential):
- Introspective: +1.250 mean, 7/1 split
- Creative: +0.350 mean, 3/0 split
- Neutral: +0.200 mean, 2/0 split

### Dose-response at +/-2000:
- Self-referential: p=0.065 (marginal, same direction)
- Engagement: p=1.0 (null)

### Cross-dimension correlations:
- self_ref x perspective: rho=0.838 (essentially same dimension)
- self_ref x engagement: rho=0.460 (related)
- self_ref x ai_identity: rho=0.522

### Pronoun analysis:
- Creative prompts: higher FP rate at -3000 (9/0, p=0.004)
- Introspective: NO difference (3/6)
- This means the judge captures something pronouns miss on introspective prompts
