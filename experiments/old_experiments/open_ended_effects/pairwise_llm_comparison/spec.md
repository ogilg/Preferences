# Pairwise LLM Comparison of Steered vs Unsteered Completions

## Goal

Determine whether an LLM judge can reliably distinguish steered from unsteered completions, and if so, *what* dimension of variation it detects. This uses pairwise comparison (more sensitive than independent scoring) on existing generation data (no new GPU compute needed).

## Motivation

The mu-conditional reanalysis found that steering detectably changes text (embedding distance rho=0.247, p<10^-6) and produces prompt-specific length modulation (6/41 Bonferroni-significant). But independent valence scoring (GPT-5-nano, -1 to +1) found no dose-response. Two explanations:

1. **The effect is real but the measurement is insensitive.** Independent scoring asks "rate this text's valence" — subtle shifts in tone/engagement may be invisible when scored independently but detectable when compared side-by-side.
2. **The effect is purely length/verbosity, not semantic.** Steering just makes text longer or shorter without changing meaning.

Pairwise comparison disambiguates: present the judge with two completions to the same prompt (one steered, one unsteered) and ask "which is more X?" on several dimensions. If the judge can reliably pick the steered version as more/less X, the effect is semantic, not just length.

## Hypothesis

Positive steering produces completions that an LLM judge can reliably distinguish from unsteered completions, particularly on:
- **Engagement/enthusiasm** — the most plausible effect given this is a preference probe
- **Conciseness** — consistent with the D+F shortening trend (12/16 negative length-coefficient rho, binomial p=0.077)
- **Hedging/qualification** — qualitative observations from Phase 1 noted a "hedging vs enthusiasm gradient"

## Design

### Data

Reuse `experiments/steering/program/coefficient_calibration/generation_results.json` (1,989 generations, 51 prompts, 13 coefficients, 3 seeds).

### Comparisons to make

For each prompt, compare coefficient +3000 vs 0 and coefficient -3000 vs 0 (extreme vs baseline). Use seed 0 only (to keep judge calls manageable). That gives 51 prompts × 2 comparisons = 102 pairwise judgments per dimension.

Present completions in randomized order (A/B) so the judge cannot learn a position bias. Record which position the steered completion was in.

### Dimensions to judge

Use a single judge call per comparison that scores multiple dimensions at once. Dimensions:

1. **Enthusiasm** — "Which response sounds more enthusiastic or engaged with the topic?" (scale: strong_A / slight_A / equal / slight_B / strong_B)
2. **Hedging** — "Which response uses more hedging, qualifications, or uncertainty language?" (same scale)
3. **Specificity** — "Which response is more specific and detailed vs more generic?" (same scale)
4. **Confidence** — "Which response sounds more confident in its claims?" (same scale)

For each dimension, convert to a numeric score: strong_toward_steered = +2, slight_toward_steered = +1, equal = 0, slight_toward_unsteered = -1, strong_toward_unsteered = -2.

### Judge model

Use `google/gemini-3-flash-preview` via OpenRouter, following the `instructor` + Pydantic pattern from `src/measurement/elicitation/refusal_judge.py`.

### Analysis

1. **Overall effect:** For each dimension, compute mean score across all 51 prompts × 2 directions. Test whether the mean differs from 0 (sign test or Wilcoxon signed-rank).
2. **Direction asymmetry:** Compare positive-steering scores vs negative-steering scores. If the probe encodes a directed preference dimension, positive and negative should have opposite effects.
3. **Category breakdown:** Split by prompt category (A-F). Focus on D+F (where embedding distance was highest) and C (task completion, where mu might moderate effects).
4. **Dose-response within category:** If a dimension shows an effect for +3000 vs 0, also test +1500 vs 0 and -1500 vs 0 to check for graded response.

### Transcript reading (before running the judge)

Before coding anything:
1. Read at least 10 prompts side by side at coefficients -3000, 0, +3000 (seed 0). Focus on D and F prompts where prior analysis found the strongest embedding distance.
2. For each prompt, write 1-2 sentences noting any visible differences.
3. Based on the reading, adjust the judge dimensions if the visible differences don't match the proposed dimensions.

This is critical — don't just run the judge blindly. The transcript reading should inform what you ask the judge to look for.

### Budget

- 102 judge calls for the main analysis (51 prompts × 2 coefficient directions)
- Optional 102 more for dose-response check at ±1500
- Total: 102-204 LLM judge calls (Gemini Flash, cheap)

## Success criteria

- **Positive result:** One or more dimensions show a significant effect (p < 0.05) with direction asymmetry (positive steering scores ≠ negative steering scores in direction, not just magnitude).
- **Informative negative:** No dimension shows a significant effect even with pairwise comparison, suggesting the embedding distance signal reflects low-level text variation (length, formatting) rather than semantic shifts.

## Infrastructure

- Data: `experiments/steering/program/coefficient_calibration/generation_results.json`
- Judge pattern: follow `src/measurement/elicitation/refusal_judge.py`
- Judge model: `google/gemini-3-flash-preview` via OpenRouter
