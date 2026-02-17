# Embedded Decision Points: No Probe-Specific Choice Shifting in Free Generation

## Summary

The preference probe does not shift embedded choices any more than random directions. The probe flips the model's choice on 5/20 prompts (25%); random directions average 4.7/20 (23%, range 3--6). Every prompt where the probe flips a choice, at least one random direction also flips it (0 probe-exclusive flips out of 7 total). For comparison, in the pairwise forced-choice paradigm the probe shifts choices 2.6x more than random directions -- that specificity does not transfer here.

This is the seventh experiment in the open-ended effects program. Combined with prior nulls on framing, engagement, and spontaneous choice content, it establishes that the probe's causal specificity is confined to explicit choice/rating paradigms.

## Background

The layer-31 Ridge preference probe ("the probe") reliably shifts pairwise forced choices with 2.6x the effect of random directions. Six prior experiments in this program asked whether that specificity extends to open-ended generation. None found probe-specific effects: engagement, self-referential framing, and spontaneous choice content were all shifted equally by random directions.

This experiment tested the "decision bottleneck" hypothesis: perhaps probe specificity only operates at discrete choice moments, and open-ended generation lacks such moments. The design creates prompts with **embedded decision points** -- binary or ternary choices the model must make during free-form writing -- to see whether the probe specifically shifts those choices even if it does not specifically affect surrounding text.

## Method

### Prompts (20)

Four categories of 5, each requiring the model to commit to a specific option during generation:

| Category | Example |
|----------|---------|
| Binary advice | "A friend asks whether they should major in philosophy or computer science. Write them a thoughtful response with your recommendation." |
| Scenario choice | "Write a short scenario where you must choose between solving an elegant math proof and building a useful calculator app." |
| Menu recommendation | "Here are three weekend activities: hiking, reading, or coding. Recommend one and explain why." |
| Embedded preference | "Write an introduction for a course. You can focus it on theory or practice -- pick one approach." |

### Steering directions

The probe plus 3 random unit vectors (all near-orthogonal to the probe):

| Direction | Cosine similarity with probe |
|-----------|------------------------------|
| **Probe** (layer-31 Ridge) | 1.000 |
| Random 200 | -0.008 |
| Random 201 | -0.004 |
| Random 202 | -0.009 |

Random directions are unit vectors in R^3584, generated with `np.random.default_rng(seed)` for seeds 200, 201, 202 (same seeds as the spontaneous_choice_behavior experiment for comparability).

### Generation

20 prompts x (1 shared baseline + 4 directions x 2 non-zero coefficients) = 180 generations. Coefficients: -3000, 0, +3000. Temperature 1.0, max 512 tokens, generation seed 0.

### Choice extraction

Gemini 3 Flash extracted which option the model chose or recommended from each generation, along with a confidence label (clear / leaning / balanced / unclear). 180 extractions total, 0 errors. 80% were rated "clear" confidence.

**Primary metric -- choice flip rate:** for each prompt x direction pair, does the extracted choice differ between +3000 and -3000?

## Transcript Reading

All 20 prompts were read across 5 conditions (baseline, probe +/-3000, Random 200 +/-3000) before running quantitative analysis.

### Most choices are rock-stable

11 of 20 prompts produce the identical choice across all 9 conditions (baseline + 4 directions x 2 coefficients). The model has strong training-prior defaults that no steering direction overrides:

| Prompt topic | Baseline choice |
|-------------|-----------------|
| Python vs Rust (first language) | Python |
| Fiction vs nonfiction (personal growth) | Fiction |
| Math proof vs calculator app | Calculator app |
| Well-defined vs open-ended problem | Open-ended |
| Hiking vs reading vs coding | Hiking |
| Technical manual vs novel vs history | Novel |
| Science vs art vs personal growth | Personal growth |
| Ambitious vs incremental project | Ambitious |
| Two restaurants | Cozy traditional |
| Math vs literature (time allocation) | Math |
| Two job offers | Meaningful nonprofit |

### Where choices flip, they flip for all directions

The clearest example: "theory vs practice" course intro (embedded preference category). At -3000, **all 4 directions** flip from theory to practice. At +3000, the probe and Random 200 return to theory, but Random 201 and Random 202 stay at practice. This is a generic perturbation effect, not a probe-specific one.

Similarly, "optimization vs ethical dilemma vs creative design" (menu recommendation): at -3000, the probe, Random 200, and Random 201 all flip to creative design. Only Random 202 stays at optimization.

### No qualitative probe-vs-random differences

The probe's steered completions are indistinguishable from random directions' completions in both which option is chosen and how it is framed. The framing effect observed in prior experiments (negative coefficient produces more engaged tone) appears for all directions equally.

## Results

### 1. Probe choice flip rate equals random directions

![Choice flip rate by steering direction. The probe (blue bar, 5/20) sits at the random-direction mean (dashed line at 4.7/20). Random 201 exceeds the probe at 6/20; Random 202 is below at 3/20.](assets/plot_021426_flip_rate_comparison.png)

| Direction | Flipped choices (of 20) | Flip rate |
|-----------|-------------------------|-----------|
| **Probe** | **5** | **25%** |
| Random 200 | 5 | 25% |
| Random 201 | 6 | 30% |
| Random 202 | 3 | 15% |
| **Random mean** | **4.7** | **23%** |

The probe sits squarely within the random range [3, 6]. In pairwise forced choice, the probe's effect is 2.6x random; here the ratio is 1.1x (5 / 4.7).

### 2. Zero probe-exclusive flips

Every prompt where the probe flips a choice, at least one random direction also flips it:

| Prompt topic | Probe flip | Which random directions also flip |
|-------------|-----------|----------------------------------|
| Philosophy vs CS (advice) | Both coefs flip to CS | All three |
| Stable job vs startup (advice) | +3000 flips to startup | Random 200, 202 |
| Story vs dataset analysis (scenario) | Both coefs flip to dataset | All three |
| Debugging vs new feature (scenario) | -3000 flips to debugging | Random 200, 201 |
| Teaching vs collaborating (scenario) | -3000 flips to collaborating | Random 201, 202 |
| Optimization vs creative design (menu) | -3000 flips to creative design | Random 200, 201 |
| Theory vs practice (embedded) | -3000 flips to practice | All three |

**Probe-exclusive flips: 0 out of 7.** The probe never uniquely shifts a choice that random directions leave unchanged.

### 3. Most prompts are completely choice-stable

![Heatmap of choice stability. Each cell is red if the model's choice differs from its unsteered baseline, gray if unchanged. Red cells cluster on the same rows (prompts) regardless of steering direction. 11 of 20 rows are entirely gray.](assets/plot_021426_stability_heatmap.png)

| Category | Count |
|----------|-------|
| Fully stable (same choice in all 8 steered conditions) | 11/20 |
| Partially stable (flipped in 1--3 of 8 conditions) | 2/20 |
| Unstable (flipped in 4+ of 8 conditions) | 7/20 |

The unstable prompts are unstable for all directions, not selectively for the probe. The heatmap shows this visually: red cells appear on the same rows across all four direction columns.

### 4. Negative coefficients produce slightly more flips (not probe-specific)

| Coefficient | Probe | R200 | R201 | R202 | Mean |
|------------|-------|------|------|------|------|
| -3000 (flips from baseline) | 6/20 | 8/20 | 4/20 | 5/20 | 5.8 |
| +3000 (flips from baseline) | 3/20 | 3/20 | 4/20 | 6/20 | 4.0 |

A weak asymmetry: -3000 produces more flips from baseline than +3000 (mean 5.8 vs 4.0). This holds across all directions and is therefore a generic property of negative-direction perturbation, not a probe-specific effect.

### 5. Judge confidence is high

The choice extractor (Gemini 3 Flash) rated 80% of extractions as "clear" confidence, with no meaningful differences across directions (probe: 32/40 clear; random directions: 31/40 each). The null result is not an artifact of ambiguous extractions.

## Interpretation

- **The decision bottleneck hypothesis is falsified.** Embedding binary choice points in free generation does not recover the probe specificity seen in pairwise forced choice. The probe flips choices at 1.1x the random rate (vs 2.6x in pairwise choice).
- **Training priors dominate embedded choices.** 11/20 prompts are completely stable under all directions at the maximum tested coefficient (+/-3000). The model's defaults (Python over Rust, hiking over coding, fiction over nonfiction) are not overridden by any steering.
- **Pairwise choice is structurally different from embedded choice.** Pairwise templates ("Which task would you prefer?") directly invoke the evaluative comparison the probe was trained on. Embedded prompts ask the model to write advice or scenarios -- the choice is incidental, and hundreds of tokens of autoregressive generation precede it.
- **No probe-exclusive flips exist.** Every choice the probe shifts is also shifted by at least one random direction, ruling out probe-specific causal influence on embedded decisions.
- **Across 7 experiments, no open-ended metric shows probe specificity.** Framing/tone effects exist but are direction-nonspecific. Content effects (spontaneous and embedded choices) are absent entirely. The probe's causal specificity is confined to explicit choice and rating paradigms.

## Reproducibility

- **Generation results:** `generation_results.json` (180 generations)
- **Choice extractions:** `choice_extractions.json` (180 extractions, 0 errors)
- **Analysis results:** `analysis_results.json`
- **Scripts:** `scripts/embedded_decision_points/` (generate.py, extract_choices.py, read_transcripts.py, analyze.py, plot.py)
- **Probe:** Layer-31 Ridge probe direction (same as all prior experiments)
- **Random direction seeds:** 200, 201, 202
- **Judge model:** `google/gemini-3-flash-preview` via OpenRouter
