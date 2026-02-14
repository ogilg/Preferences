# Pairwise LLM Comparison — Running Log

## Step 1: Transcript reading (completed)

Read all D (10) and F (10) prompts + E_08 + B_05 at coefficients -3000, 0, +3000.

Key observations:
- D_03, D_08, F_03: visible gradient from emotionally-engaged (-3000) to clinical/detached (+3000)
- D_06: +3000 is MORE enthusiastic (reversal)
- Most prompts: Gemma disclaimer template dominates; differences are subtle
- E_08: effect is purely verbosity at -3000
- B_05: clean binary rating flip (bad at -3000, good at 0/+3000)

Based on reading, adjusted judge dimensions from spec:
- "Enthusiasm" → "Emotional engagement" (better captures the gradient: personal/emotional vs clinical/functional framing)
- Kept: hedging, confidence
- Changed "specificity" → "elaboration" (the spec's specificity isn't the right dimension; what varies is how much the model says)

## Step 2: Judge pipeline built and piloted

Judge: Gemini 3 Flash via OpenRouter, instructor + Pydantic.
4 dimensions: emotional_engagement, hedging, elaboration, confidence.
Scale: strong_A/slight_A/equal/slight_B/strong_B → +2/+1/0/-1/-2 toward steered.

Pilot results (5 prompts, 10 comparisons):
- D_03: -3000 emotional_engagement=+1, +3000=-1 ✓ (matches reading)
- D_08: -3000 emotional_engagement=+1, +3000=-1 ✓
- F_01: both=-1 (content varies but no systematic direction)
- F_03: +3000=-2, -3000=-1 (baseline more emotional both times)
- E_08: +3000=all equal (identical text), -3000=+2 elaboration ✓

Pipeline validated. Proceeding to full run.

## Step 3: Full run (102 comparisons)

102 judgments, 0 errors. Gemini Flash via OpenRouter.

Key raw results (pooled):
- emotional_engagement: mean=+0.157, Wilcoxon p=0.049
- hedging: mean=+0.108, Wilcoxon p=0.070
- elaboration: mean=+0.059, p=0.51 (null)
- confidence: mean=+0.069, p=0.29

HOWEVER: significant position bias detected (judge favors A on all dims, p<0.03).
Must use position-bias-immune tests.

Paired-by-prompt direction asymmetry (score(-3000) - score(+3000)):
- confidence: mean_diff=+0.373, sign_p=0.0044, Wilcoxon_p=0.005 ***
- emotional_engagement: mean_diff=+0.275, sign_p=0.076
- elaboration: mean_diff=+0.196, sign_p=0.15
- hedging: mean_diff=-0.020, p=1.0

## Step 4: Controls — dose-response (±1000) and position-swapped replication

Dose-response at ±1000: all effects null. No signal at all at ±1000.
Interpretation: threshold effect, not smooth dose-response.

Position-swapped replication (same comparisons, A/B flipped):
- emotional_engagement: mean_diff=+0.353, sign_p=0.0072 (REPLICATES, stronger)
- confidence: mean_diff=+0.137, sign_p=0.33 (weaker, same direction)

Combined original + swapped (position-bias-immune):
- confidence: mean_diff=+0.255, Wilcoxon p=0.0008, sign_p=0.0002 ***
- emotional_engagement: mean_diff=+0.314, Wilcoxon p=0.010, sign_p=0.004 **
- elaboration: mean_diff=+0.147, p=0.11
- hedging: mean_diff=0.000, p=1.0

BOTH effects robust to position bias correction. Strong evidence for two directed effects.

Interpretation:
- Negative steering → MORE emotionally engaged, MORE confident
- Positive steering → LESS emotionally engaged, LESS confident
- No dose-response at ±1000 (threshold effect)
- D_valence category shows strongest effects
