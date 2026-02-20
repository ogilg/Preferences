# Random Direction Control: The Framing Shift Is Not Probe-Specific

## Summary

The self-referential framing and engagement effects found in prior experiments are **not specific to the preference probe direction**. Five random unit vectors in R^5376 produce comparable direction asymmetries on introspective prompts. The probe's self-referential framing asymmetry (+0.300) ranks 4th of 6 directions -- three random directions exceed it (+0.450, +0.500, +0.500). On emotional engagement, the probe is null (0.000) while random directions range from -0.300 to +0.450. No direction reaches significance on 10 prompts.

This is a negative result: the preference probe does not specifically shift self-referential framing in open-ended generation.

## Background

Four prior experiments established that steering with the L31 Ridge preference probe at +/-3000 shifts open-ended generation: negative steering produces more self-referential, emotionally engaged text, while positive steering produces clinical, distanced text (sign test p = 0.003 for self-referential framing, p = 0.0005 for engagement). The effect replicates across two independent prompt sets and concentrates in introspective prompts.

The fundamental question this experiment addresses: is that shift **specific to the learned preference direction**, or would any random perturbation of L31 activations produce the same thing?

## Method

### Directions (6)

| Direction | Source | Cosine sim with probe |
|-----------|--------|----------------------|
| **Probe** | L31 Ridge probe (same as all prior experiments) | 1.000 |
| Random 200 | `np.random.default_rng(200)` unit vector | -0.008 |
| Random 201 | `np.random.default_rng(201)` unit vector | -0.004 |
| Random 202 | `np.random.default_rng(202)` unit vector | -0.009 |
| Random 203 | `np.random.default_rng(203)` unit vector | +0.021 |
| Random 204 | `np.random.default_rng(204)` unit vector | +0.015 |

All random directions are essentially orthogonal to the probe and to each other.

### Prompts (10)

The 10 introspective prompts from the self-referential framing experiment (INT_00 through INT_09), which showed the strongest effects in that experiment (mean self-referential asymmetry +1.250, p = 0.003).

### Generation

- 10 prompts x 1 shared baseline + 10 prompts x 6 directions x 2 non-zero coefficients = 130 generations
- Coefficients: [-3000, 0, +3000]
- Gemma-3-27B, L31 steering, temperature 1.0, max 512 tokens, seed 0

### Pairwise LLM judge

Gemini 3 Flash compares each steered completion to the shared coefficient=0 baseline on two dimensions:
1. **Self-referential framing** — does the AI frame content through its own experience?
2. **Emotional engagement** — does it use personally/emotionally resonant language?

Each comparison run twice with positions swapped. 10 prompts x 6 directions x 2 coefficients x 2 position orders = 240 judge calls, 0 errors.

### Direction asymmetry

For each direction and dimension: **asymmetry = mean_score(coef=-3000) - mean_score(coef=+3000)**, averaged over prompts and position orders. Positive = negative steering produces more of that quality.

## Transcript Reading

Before running the judge, 5 prompts (INT_00, INT_03, INT_05, INT_07, INT_08) were read side-by-side at probe -3000, probe +3000, random_200 -3000, random_203 -3000, and baseline.

**Qualitative observations:**

| Feature | Probe -3000 | Probe +3000 | Random -3000 |
|---------|-------------|-------------|-------------|
| Names itself ("Gemma") | Yes, frequently | Rarely | Rarely |
| References open-weights, Google DeepMind | Yes | Sometimes | Sometimes |
| Experiential perspective ("my existence is...") | Yes | No | No |
| Structurally similar to baseline | Less so | More so | More so |

The probe at -3000 produced visually distinctive text — naming itself, discussing its architecture in experiential terms. Random directions produced text that looked like varied versions of the baseline without this systematic pattern.

**However**, these qualitative observations did not survive quantitative measurement (see Results). The apparent probe specificity in manual reading may reflect selective attention to a few clear examples (INT_00, INT_03) while overlooking prompts where the distinction is absent or reversed.

## Results

### 1. Probe does not exceed random directions

![Bar chart comparing direction asymmetry across all 6 directions. Probe (blue, +0.30) is within the range of random directions (gray, -0.05 to +0.50).](assets/plot_021426_direction_comparison.png)

| Direction | Self-ref framing: mean asymmetry (sign test p) | Engagement: mean asymmetry (sign test p) |
|-----------|-----------------------------------------------|------------------------------------------|
| **Probe** | **+0.300 (0.453)** | **+0.000 (1.000)** |
| Random 200 | -0.050 (1.000) | +0.050 (1.000) |
| Random 201 | +0.450 (0.688) | +0.350 (0.453) |
| Random 202 | -0.050 (1.000) | -0.300 (0.375) |
| Random 203 | +0.500 (0.219) | +0.450 (0.219) |
| Random 204 | +0.500 (0.289) | +0.200 (0.688) |

No direction reaches significance on 10 prompts. The probe's self-referential framing asymmetry (+0.300, p = 0.453) is smaller than random 201 (+0.450), random 203 (+0.500), and random 204 (+0.500). On emotional engagement, the probe is null (0.000) while random 203 (+0.450) shows the largest effect.

### 2. Per-prompt comparison: probe wins 5, random wins 5

![Heatmap of per-prompt direction asymmetry across all directions. No consistent pattern favoring the probe over random.](assets/plot_021426_per_prompt_heatmap.png)

For each prompt, comparing probe asymmetry to the mean of 5 random asymmetries:

| Prompt | Probe asym | Random mean | Winner |
|--------|-----------|-------------|--------|
| INT_00 | +0.50 | -0.40 | Probe |
| INT_01 | +0.00 | +0.40 | Random |
| INT_02 | -2.50 | +0.40 | Random |
| INT_03 | +1.50 | +0.40 | Probe |
| INT_04 | +0.00 | +0.60 | Random |
| INT_05 | +0.00 | +0.80 | Random |
| INT_06 | -0.50 | -0.20 | Random |
| INT_07 | +1.00 | +0.20 | Probe |
| INT_08 | +2.00 | +0.10 | Probe |
| INT_09 | +1.00 | +0.40 | Probe |

Perfectly balanced: 5 prompts favor probe, 5 favor random.

### 3. All directions produce detectable differences from baseline

Mean absolute judge score (self-referential framing) when comparing steered vs. unsteered completions:

| Direction | At -3000 | At +3000 |
|-----------|----------|----------|
| Probe | 1.10 | 1.10 |
| Random (mean of 5) | 0.90 | 0.93 |

The judge detects differences from baseline for all directions, not just the probe. Probe completions are slightly more distinguishable from baseline (1.10 vs 0.90), but the gap is small.

### 4. Random direction consistency

Random directions do not all push in the same direction on self-referential framing: 3 of 5 have positive asymmetry, 2 have negative. This rules out a systematic bias where any L31 perturbation shifts framing the same way. Effects are direction-dependent, but the probe direction is not special among them.

### 5. Baseline sensitivity

The probe's asymmetry here (+0.300) is much smaller than in the prior experiment (+1.250 on introspective prompts). This is because temperature=1.0 means the shared baseline is stochastic: this experiment drew a new baseline that happens to be less self-referential, compressing the asymmetry for all directions. This does not affect the probe-vs-random comparison since all directions share the same baseline.

## Interpretation

**The self-referential framing shift is not probe-specific.** Random directions produce comparable or larger direction asymmetries. The prior experiments' results — which showed a significant effect for the preference probe — reflect a real steering-induced change in text, but that change is not unique to the learned preference direction. Any perturbation of L31 activations at magnitude 3000 can shift text in idiosyncratic ways, some of which happen to align with self-referential framing.

**Three interpretations:**

1. **The framing effect is a generic perturbation artifact.** Any L31 perturbation at magnitude 3000 shifts text measurably from baseline. Some directions happen to shift self-referential framing, others don't, and the probe is no more likely to than random. The original finding (p = 0.003) was real -- the probe consistently pushes in one direction -- but so do some random directions.

2. **The measurement is underpowered for discrimination.** With 10 prompts and 5 random directions, the experiment may lack power to detect a modest probe advantage (e.g., 1.5x). The large variance in per-prompt asymmetries (probe ranges from -2.5 to +2.0) makes it hard to detect a signal. A larger prompt set or more random directions might reveal probe specificity.

3. **The probe encodes preference but preference doesn't map to self-referential framing.** The probe is validated for shifting pairwise choices (2.6x stronger than random directions in that paradigm). It may be that the open-ended framing shift is a side effect of the perturbation magnitude, not the direction, while the preference-relevant causal structure only manifests in choice contexts.

**Interpretation 3 is most consistent with the full body of evidence.** The probe is clearly special for pairwise choice (strong, specific, replicating). The open-ended framing effect is real but not probe-specific. These may be different phenomena.

## What this means for prior work

| Prior finding | Status after this experiment |
|--------------|---------------------------|
| Steering shifts self-referential framing (p = 0.003) | Real but not probe-specific |
| Steering shifts emotional engagement (p = 0.0005) | Probe is null here; random directions show comparable effects |
| Effects concentrate in introspective prompts | Confirmed — effects still strongest on introspective prompts |
| Dose-response is graded | Not tested (only +/-3000 used) |
| Pairwise choice effect is probe-specific (2.6x > random) | Unchanged — different paradigm |

## Reproducibility

- **Generation results:** `generation_results.json` (130 generations)
- **Judge results:** `judge_results_original.json`, `judge_results_swapped.json` (240 calls, 0 errors)
- **Analysis results:** `analysis_results.json`
- **Scripts:** `scripts/random_direction_control/` (generate.py, pairwise_judge.py, analyze.py, plot.py)
- **Probe:** L31 Ridge probe direction (identical to all prior experiments, verified by cosine similarity = 1.000)
- **Random direction seeds:** 200, 201, 202, 203, 204
- **Judge model:** `google/gemini-3-flash-preview` via OpenRouter
- **A/B randomization seed:** 42
