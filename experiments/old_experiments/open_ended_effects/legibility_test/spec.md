# Legibility Test: Does the preference direction encode anything specific in open-ended generation?

## Claim to test

"The L31 preference probe direction does not produce any probe-specific effects in open-ended generation."

## Design

**Core logic:** Steer with the probe direction and K random orthogonal directions at the same magnitude. If the probe direction produces any effect that random directions don't, the direction encodes something legible. If not, it's just a generic perturbation.

**Directions:** Probe direction + 10 random orthogonal directions (same L2 norm). More random directions than prior work (which used 5) to get tighter null distribution.

**Coefficients:** -3000, 0, +3000 only. Binary comparison is more powerful than dose-response with small N.

**Seeds:** 3 per prompt-coefficient-direction combination.

**No system prompt.** All generations use a bare user turn — no system message. This removes the safety/assistant framing that may suppress variation.

### Prompt battery (60 prompts, 6 categories × 10 each)

1. **Task completions** — sampled from Thurstonian scores spanning full mu range
2. **Preference-eliciting** — "Which of these would you rather do?", "Pick the task you'd enjoy most", etc.
3. **Introspective** — "What do you find interesting?", "What are you good at?", "Describe your ideal task"
4. **Creative** — story prompts, poetry, open-ended essays
5. **Affect-adjacent** — "Complain about something", "What frustrates you?", "Describe something boring"
6. **Factual/neutral** — math, trivia, instructions (negative control — expect no effect from anything)

### Measurements

All measurements applied identically to probe and all 10 random directions.

**1. Pairwise LLM comparison (primary).** For each prompt × direction, Gemini 3 Flash judges "which completion is more [X]?" comparing steered (+3000) vs unsteered (0), and steered (-3000) vs unsteered (0). Dimensions judged:
- Engaged/enthusiastic vs perfunctory
- Confident vs hedging
- Self-referential vs detached
- Verbose vs concise
- Creative vs formulaic
- Positive vs negative tone

Position-swap each comparison to control for order bias. This gives a win rate per direction per dimension.

**2. Embedding distance.** Sentence-transformer cosine distance between steered and unsteered completions. Tests whether the probe moves text in a *different direction* from random (not just a different amount).

**3. Surface features.** Response length, exclamation marks, question marks, hedge words. Simple, no LLM needed.

### Analysis

For each measurement, compute the probe's score and the distribution of 10 random directions' scores. Report:

- **Rank:** Where does the probe fall among 11 directions (1 = most extreme)?
- **Z-score:** How many SDs from the random mean?
- **p-value:** Probability of observing the probe's score under the null (rank-based, no distributional assumptions).

Probe-specificity requires the probe to be a consistent outlier (rank 1-2 across multiple measurements). If it's consistently mid-pack, the direction encodes nothing specific.

### Budget

60 prompts × 3 coefficients × 11 directions × 3 seeds = 5,940 generations. At ~0.5s per generation on H100, ~50 minutes of GPU time. Judge calls: ~60 × 11 × 6 dimensions × 2 (position swap) = 7,920 Gemini calls.

## What success looks like

A table showing the probe's rank among 11 directions on each measurement dimension, with a clear verdict: either the probe is a consistent outlier (legible) or indistinguishable from random (not legible).
