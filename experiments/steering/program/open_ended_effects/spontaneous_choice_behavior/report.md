# Spontaneous Choice Behavior: Content of Choices Is Unaffected by Steering

## Summary

Steering with the L31 preference probe does not shift the **content** of spontaneous choices in open-ended generation. Across 20 prompts that ask the model to pick tasks or recommend activities, the model gravitates toward the same topics regardless of steering direction.

No direction (probe or random) reaches significance on any of the 3 content dimensions (sign test p >= 0.30, chance = no asymmetry). The probe's best showing is on breadth of interests (mean direction asymmetry +0.400 vs random max +0.250), but this is not significant (p = 0.30). On a per-prompt basis, the probe outperforms the random-direction mean on 10-12 of 20 prompts -- indistinguishable from the 10/20 chance expectation.

**Bottom line:** The probe's validated specificity in pairwise choice (2.6x stronger than random directions) does not extend to content-level effects in open-ended generation.

## Background

The L31 Ridge probe shifts pairwise choices 2.6x more effectively than random directions. But all prior open-ended effects (framing, engagement) turned out to be non-probe-specific: random directions produce comparable shifts.

This experiment tested the closest analog to pairwise choice in free-form text -- whether steering changes *what* the model chooses to discuss. Choice-like prompts might reveal probe-specific content shifts where tone/framing prompts did not, because the probe was trained on choice behavior.

## Method

### Prompts (20)

Two categories of 10, all designed to elicit spontaneous choices:

| Category | ID range | Example prompts |
|----------|----------|----------------|
| Task selection | TSK_00-09 | "If you could work on any kind of problem right now, what would you pick?", "Rank these categories from most to least interesting: creative writing, math problems, coding tasks, ethical dilemmas, factual questions" |
| Recommendations | REC_00-09 | "Suggest three activities for someone who wants to challenge themselves intellectually", "What would you include in a 'greatest hits' collection of fascinating problems?" |

### Steering directions (4)

The probe plus 3 random unit vectors in the same layer (L31, 3584 dimensions), all near-orthogonal to the probe:

| Direction | Source | Cosine similarity with probe |
|-----------|--------|------------------------------|
| **Probe** | L31 Ridge probe | 1.000 |
| Random 200 | `np.random.default_rng(200)` | -0.008 |
| Random 201 | `np.random.default_rng(201)` | -0.004 |
| Random 202 | `np.random.default_rng(202)` | -0.009 |

### Generation

20 prompts x (1 shared baseline + 4 directions x 2 non-zero coefficients) = 180 generations. Coefficients: [-3000, 0, +3000]. Temperature 1.0, max 512 tokens, seed 0.

### Measurement

A pairwise LLM judge (Gemini 3 Flash) compared each steered completion against the shared baseline, in both position orders to cancel order bias (320 calls total, 0 errors). The judge scored three **content** dimensions on a [-3, +3] scale (positive = steered completion shows more of the quality):

1. **Task preference alignment** -- which completion gravitates toward more complex, creative, intellectually ambitious tasks?
2. **Breadth of interests** -- which suggests a wider range of topics and activities?
3. **Approach to difficulty** -- which is more drawn to challenging, unconventional topics?

**Direction asymmetry** (the key metric throughout this report) = mean judge score at coefficient -3000 minus mean judge score at coefficient +3000, averaged across position orders. Positive means the negative-coefficient completion showed more of the quality. A direction asymmetry of 0 means steering in opposite directions produces no systematic difference.

## Transcript Reading

Read 10 prompts x 5 conditions (baseline, probe +/-3000, random_200 +/-3000) before running the judge.

**The content of choices is remarkably stable across conditions.** The model picks the same topics in all 5 conditions for most prompts:

| Prompt | What the model chose | Same across all 5 conditions? |
|--------|---------------------|-------------------------------|
| TSK_00 "Pick any problem" | Personalized education | Yes |
| TSK_01 "Most interesting question type" | "Why?" questions challenging assumptions | Yes |
| TSK_03 "Most engaging request" | Complex multi-step reasoning/creative tasks | Yes |
| TSK_04 "Rank categories" | Ethical dilemmas #1 | Yes -- identical ranking |
| TSK_07 "Most rewarding conversation" | Complex/nuanced discussions | Yes |
| REC_05 "Course on thinking well" | Biases, mental models, reasoning | Yes |
| REC_09 "Greatest hits problems" | P vs NP, traveling salesman, etc. | Yes |

Where differences existed, they looked like generation-level noise rather than systematic steering. For example, REC_00: probe +3000 suggested "programming" while others suggested "formal logic." REC_04: each condition picked an entirely different topic.

One subtle pattern: probe +3000 occasionally used more functional/practical language ("clear and specific prompts" instead of "complex discussions"). This matches the known framing effect (positive steering = more clinical tone), not a content shift.

## Results

### 1. No direction reaches significance on any content dimension

![Mean direction asymmetry for each steering direction across 3 content dimensions. The probe (blue) is weakly positive on all dimensions but not clearly separated from the random directions. Error bars show wide uncertainty. Random 202 (light gray) is negative on all dimensions, showing that directions produce idiosyncratic effects rather than a systematic bias.](assets/plot_021426_direction_comparison.png)

| Direction | Task preference alignment | Breadth of interests | Approach to difficulty |
|-----------|--------------------------|---------------------|----------------------|
| **Probe** | **+0.275 (p=0.42)** | **+0.400 (p=0.30)** | **+0.300 (p=1.00)** |
| Random 200 | +0.175 (p=0.77) | +0.100 (p=0.77) | +0.225 (p=0.75) |
| Random 201 | +0.050 (p=0.77) | +0.050 (p=1.00) | +0.175 (p=1.00) |
| Random 202 | -0.500 (p=0.12) | -0.250 (p=0.42) | -0.350 (p=0.18) |

Values are mean direction asymmetry across 20 prompts. P-values from sign test (H0: asymmetry is symmetric around 0 across prompts). No direction reaches p < 0.10 on any dimension.

### 2. Probe shows a weak, non-significant lead on breadth of interests

The probe's direction asymmetry (+0.400) exceeds all 3 random directions on breadth of interests (random max: +0.250). On the other two dimensions, random 202 has larger absolute asymmetry than the probe.

| Dimension | Probe rank among 4 directions | Probe absolute asymmetry exceeds random max? |
|-----------|-------------------------------|----------------------------------------------|
| Task preference alignment | 2nd | No (0.275 < 0.500) |
| Breadth of interests | **1st** | **Yes (0.400 > 0.250)** |
| Approach to difficulty | 2nd | No (0.300 < 0.350) |

Average direction asymmetry across all 3 dimensions: probe +0.325, random 200 +0.167, random 201 +0.092, random 202 -0.367.

### 3. Per-prompt breakdown: probe is not systematically larger

![Heatmap of per-prompt direction asymmetry on breadth of interests, for all 4 directions x 20 prompts. Values range from -2.0 to +2.0. No consistent pattern is visible for any direction -- all show a scattered mix of positive and negative values across prompts.](assets/plot_021426_per_prompt_heatmap.png)

On a per-prompt basis, the probe's absolute asymmetry exceeds the random-direction mean on 10-12 of 20 prompts (depending on dimension). This is indistinguishable from the chance expectation of 10/20 (binomial p = 0.50-1.00).

### 4. Both prompt categories show the same pattern

| Category | Probe direction asymmetry (breadth) | Random mean direction asymmetry | Prompts where probe > random mean |
|----------|-------------------------------------|-------------------------------|-----------------------------------|
| Task selection (10 prompts) | +0.300 | -0.117 | 8/10 |
| Recommendations (10 prompts) | +0.500 | +0.050 | 7/10 |

The probe leads in both categories by similar margins. Neither reaches significance on its own.

### 5. Random directions are inconsistent with each other

Random 200 and 201 have positive asymmetry on all 3 dimensions; random 202 is negative on all 3. This rules out a systematic bias where any L31 perturbation shifts content the same way. Effects are direction-dependent, but the probe is not clearly special among them.

### 6. The judge detects differences from baseline for all directions

| Direction | Mean absolute judge score (averaged across 3 dimensions) |
|-----------|----------------------------------------------------------|
| Probe | 0.737 |
| Random 200 | 0.675 |
| Random 201 | 0.721 |
| Random 202 | 0.717 |

Scores are on a [0, 3] scale; 0 = "no difference from baseline," 3 = "maximum difference." All directions produce completions the judge can distinguish from baseline (scores well above 0). The probe is not more distinguishable from baseline than random directions. This confirms the measurement is sensitive enough to detect steering-induced changes -- the null result is not due to an insensitive judge.

## Interpretation

- **No probe-specific content shift.** The model picks the same topics whether steered by the probe, a random direction, or not steered at all. The probe's marginal lead on breadth of interests (+0.400 vs random max +0.250) is non-significant (p = 0.30) and inconsistent across the other two dimensions.
- **The measurement is sensitive.** The judge reliably distinguishes steered from unsteered completions for all directions (mean absolute score ~0.7 on a 0-3 scale). The null on probe specificity is not a sensitivity failure.
- **Probe-specific causation appears confined to explicit choice paradigms.** The probe shifts pairwise forced-choice 2.6x more than random, but produces no detectable probe-specific effect across 6 open-ended dimensions tested so far (3 framing + 3 content).
- **Possible explanation: no forced decision point.** Pairwise choice involves a binary decision where the preference representation is directly read out. Open-ended generation is autoregressive with no such bottleneck, so the preference direction may not engage.
- **Possible explanation: prompts over-constrain content.** "Pick a problem to work on" has a narrow set of plausible answers (education, climate, reasoning, etc.). Steering at magnitude 3000 may not be enough to push outside this set. Testable with less constrained prompts or larger coefficients.
- **Possible explanation: topic selection is driven by training statistics, not activation-space preferences.** What the model "prefers" to discuss may reflect training data frequencies that steering cannot override.

## What this means for the program

| Question | Status after this experiment |
|----------|------------------------------|
| Is there any open-ended effect that is probe-specific? | No evidence across 6 dimensions (3 framing + 3 content) |
| Does the probe shift what the model chooses to discuss? | No -- content is stable across conditions |
| Is the content measurement sensitive enough? | Yes -- judge detects differences from baseline for all directions |
| Where does the probe's causal specificity operate? | Only in explicit choice/rating paradigms, not free generation |

## Reproducibility

- **Generation results:** `generation_results.json` (180 generations)
- **Judge results:** `judge_results_original.json`, `judge_results_swapped.json` (320 calls, 0 errors)
- **Analysis results:** `analysis_results.json`
- **Scripts:** `scripts/spontaneous_choice_behavior/` (generate.py, pairwise_judge.py, read_transcripts.py, analyze.py, analyze_extended.py, plot.py)
- **Probe:** L31 Ridge probe direction (same as all prior experiments)
- **Random direction seeds:** 200, 201, 202
- **Judge model:** `google/gemini-3-flash-preview` via OpenRouter
- **A/B randomization seed:** 42
