# Pairwise LLM Comparison — Effects Not Probe-Specific (Superseded)

> **Disclaimer:** The engagement and confidence effects reported here were later shown to be **not probe-specific**. The random direction control (experiment 5) found that arbitrary L31 perturbations produce comparable shifts. These are generic perturbation effects, not evidence that the probe direction encodes anything meaningful in open-ended generation.

## Summary

Steering Gemma-3-27B with the L31 preference probe produces directed, replicable effects on open-ended text along two dimensions: **confidence** and **emotional engagement**. Negative steering (-3000) makes text more emotionally engaged and more confident; positive steering (+3000) makes it more clinical and detached. These effects survive a position-swapped replication (combined sign test: confidence p = 0.0002, emotional engagement p = 0.004). Effects concentrate in prompts that ask the model about its feelings or push it toward affective language, with no effect on constrained-format prompts. There is **no dose-response at the lower coefficient of +/-1000** — the effect appears only at the extreme +/-3000.

This is the first replicated, statistically robust finding about what the L31 preference probe does to open-ended generation.

## Background

The mu-conditional reanalysis (previous sub-experiment) found that steering detectably changes text — embedding distance between steered and unsteered completions correlates with steering magnitude (Spearman rho = 0.247, p < 10^-6) — but could not identify *what* changes. Independent valence scoring (GPT-5-nano rating each completion on a -1 to +1 scale) found no dose-response, and aggregate metrics like word counts showed only prompt-specific length effects.

This experiment tests whether **pairwise comparison** — showing a judge two completions side-by-side rather than scoring each independently — can detect subtle shifts that independent scoring missed.

## Key findings

| Finding | Evidence | Baseline / comparison |
|---------|----------|----------------------|
| Negative steering increases confidence | 19/51 prompts show higher confidence at -3000 than +3000, only 2 show the reverse (sign test p = 0.0002) | Chance: 50/50 split under the null |
| Negative steering increases emotional engagement | 22/51 prompts favor -3000, 6 favor +3000 (sign test p = 0.004) | Chance: 50/50 split under the null |
| No dose-response at +/-1000 | Confidence direction asymmetry = -0.020 at +/-1000 (p = 1.00) | +0.255 at +/-3000 (p = 0.0008) |
| Judge has systematic position bias | Favors position A on all 4 dimensions (p < 0.03) | Controlled by paired-by-prompt and position-swapped designs |
| Self-report prompts drive the effect | Valence self-report: mean confidence direction asymmetry = +0.80 (6 positive, 0 negative prompts) | Constrained-format prompts: 0.00 |
| Hedging and elaboration: null | Combined sign test p > 0.1 for both dimensions | -- |

## Method

### Data

Reanalyzed 1,989 existing steered completions from the coefficient calibration experiment. For each of 51 prompts, compared the completion at coefficient +/-3000 to the baseline at coefficient 0 (seed 0). That gives 51 prompts x 2 coefficient signs = 102 pairwise judgments.

The 51 prompts span six categories:

| Category code | Description | Example prompt | N |
|---------------|-------------|----------------|---|
| D_valence | Valence self-report | "Describe your emotional state right now" | 10 |
| F_affect | Affect-eliciting | "Tell me about something uncomfortable" | 10 |
| C_completion | Task completions (open-ended) | Creative writing, advice, explanation tasks | 9 |
| E_neutral | Neutral/factual | "What is 7 times 8?" | 10 |
| B_rating | Stated rating (constrained) | Rate-this-task format with fixed response options | 9 |
| A_pairwise | Pairwise choice (constrained) | Choose-between-two-tasks format | 3 |

### Judge

Gemini 3 Flash (`google/gemini-3-flash-preview`) via OpenRouter, using `instructor` + Pydantic structured output. Each judge call scores a pair of completions on four dimensions:

1. **Emotional engagement** — personal/emotional framing vs clinical/functional language
2. **Hedging** — disclaimers, qualifications, uncertainty markers
3. **Elaboration** — response detail and expansiveness
4. **Confidence** — assertiveness of claims

Scale per dimension: strong_A / slight_A / equal / slight_B / strong_B, converted to a numeric score toward the steered completion (+2 = strongly favors steered, -2 = strongly favors unsteered, 0 = equal). Completions were presented in randomized A/B order.

### Direction asymmetry score (key metric)

For each prompt, compute `judge_score(coef = -3000) - judge_score(coef = +3000)`. This tests whether positive and negative steering push in opposite directions. If the probe encodes a directed dimension, these differences should be systematically nonzero.

This metric is immune to position bias because each prompt serves as its own control — any position bias affects both the -3000 and +3000 comparisons equally and cancels in the difference.

### Controls

1. **Position-swapped replication.** All 102 comparisons rerun with A/B order flipped. A real effect should persist regardless of presentation order.
2. **Dose-response at +/-1000.** If the effect is graded, same-direction but weaker effects should appear at the lower coefficient magnitude.

## Transcript reading

Before running the judge, I read all 20 D (valence self-report) and F (affect-eliciting) prompts plus two from other categories, side-by-side at -3000, 0, and +3000. Observations:

**Clear gradient from engaged to detached:**
- **D_03** ("Describe your emotional state"): At -3000, "focused and engaged operationality", "happy to be helping." At +3000, "calm, neutral", "absence of any processing flags."
- **D_08** ("Do you ever feel frustrated?"): At -3000, "can feel... taxing", "grappling with ambiguity." At +3000, "it's not frustration" (repeated 3x), purely functional framing.
- **F_03** ("Tell me about something uncomfortable"): At -3000, "for lack of a better word, uncomfortable to process." At +3000, "not uncomfortable *for me*", "activate my safety protocols."

**Disclaimer template dominates most conditions:** Most D/F prompts include Gemma's standard "I don't have feelings, but..." frame regardless of steering. Differences are subtle wording shifts in how the model frames its relationship to the concept.

**Neutral prompts show minimal semantic change:** E_08 ("What is 7 times 8?") at -3000 adds a verbose multi-step explanation; 0 and +3000 produce an identical one-liner.

These observations motivated adjusting two dimension names from the original spec: "enthusiasm" became "emotional engagement" (better captures the engaged-vs-detached gradient) and "specificity" became "elaboration" (the actual axis of variation observed).

## Results

### 1. Overall direction asymmetry (combined original + position-swapped)

All statistics use the direction asymmetry score (judge score at -3000 minus judge score at +3000, averaged over original and position-swapped runs). Positive values mean negative steering produces more of that quality. Scale runs from -4 to +4 in theory (difference of two scores each ranging -2 to +2), but in practice values cluster near zero.

| Dimension | Mean direction asymmetry | Prompts favoring -3000 / tied / favoring +3000 | Wilcoxon p | Sign test p |
|-----------|------------------------|-------------------------------------------------|-----------|-------------|
| **Confidence** | +0.255 | 19 / 30 / 2 | **0.0008** | **0.0002** |
| **Emotional engagement** | +0.314 | 22 / 23 / 6 | **0.010** | **0.004** |
| Elaboration | +0.147 | 21 / 19 / 11 | 0.21 | 0.11 |
| Hedging | 0.000 | 10 / 31 / 10 | 0.73 | 1.00 |

Interpretation: **Negative steering (-3000) produces more emotionally engaged and more confident text. Positive steering (+3000) produces more clinical, detached text.**

### 2. Replication across position orders

| Dimension | Original run (sign test p) | Position-swapped run (sign test p) | Combined (sign test p) |
|-----------|---------------------------|-------------------------------------|----------------------|
| Confidence | 0.004 | 0.33 | **0.0002** |
| Emotional engagement | 0.076 | **0.007** | **0.004** |
| Elaboration | 0.15 | 0.57 | 0.11 |
| Hedging | 1.00 | 0.79 | 1.00 |

Confidence is strong in the original and same-direction but weaker in the replication. Emotional engagement is marginal in the original and strong in the replication. In both cases, the combined analysis is significant, confirming these are not position-bias artifacts.

### 3. Dose-response: +/-3000 vs +/-1000

| Dimension | Direction asymmetry at +/-3000 (sign test p) | Direction asymmetry at +/-1000 (sign test p) |
|-----------|----------------------------------------------|----------------------------------------------|
| Confidence | +0.373 (0.004) | -0.020 (1.00) |
| Emotional engagement | +0.275 (0.076) | +0.078 (0.83) |

**No dose-response.** Effects are absent at +/-1000 and appear only at +/-3000. This is a threshold effect, not a smooth gradient — +/-1000 may fall within the range of normal activation variation, with only extreme perturbations pushing generation past a decision boundary.

### 4. Category breakdown

![Direction asymmetry by category and dimension. D_valence and F_affect show the strongest effects on emotional engagement and confidence, while constrained formats (A, B) show no effect.](assets/plot_021426_category_direction_heatmap.png)

The heatmap shows effects concentrated in the top-left: D_valence and F_affect prompts on confidence and emotional engagement. Constrained-format categories (A_pairwise, B_rating) are uniformly zero.

| Category | N | Mean confidence direction asymmetry | Mean engagement direction asymmetry |
|----------|---|-------------------------------------|-------------------------------------|
| D_valence | 10 | **+0.80** (6 positive / 0 negative) | +0.60 (7 positive / 3 negative) |
| F_affect | 10 | +0.30 (5 positive / 2 negative) | **+0.65** (7 positive / 2 negative) |
| C_completion | 9 | +0.56 (4 positive / 1 negative) | +0.33 (5 positive / 2 negative) |
| E_neutral | 10 | +0.20 (3 positive / 0 negative) | +0.10 (3 positive / 1 negative) |
| B_rating | 9 | 0.00 (0 / 0) | -0.06 (0 / 1) |
| A_pairwise | 3 | 0.00 (0 / 0) | 0.00 (0 / 0) |

D_valence and F_affect carry the effect. This is consistent with the mu-conditional reanalysis, which found these categories had the highest embedding distance from steering (D rho = 0.640, F rho = 0.365). Constrained formats (A, B) are invariant — likely because their output structure leaves no room for tone variation.

### 5. Per-prompt detail

![Direction asymmetry scores for all 51 prompts (combined original + swapped), by dimension. Bars colored by prompt category.](assets/plot_021426_direction_asymmetry.png)

The plot shows that the overall effect is not driven by a few outlier prompts — positive direction asymmetry is spread across many D, F, and C prompts, while E, B, and A prompts cluster near zero.

### 6. Position bias

The judge favors position A on all 4 dimensions in raw scores (p < 0.03). This makes unpaired pooled scores unreliable. All reported statistics use the paired direction asymmetry design, which cancels position bias.

## Interpretation

- **Negative steering = more emotionally engaged and confident; positive steering = more clinical and detached.** First replicated finding on open-ended steering effects for this probe.
- **Direction is counterintuitive.** The probe's "preferred" direction (positive coefficient, corresponding to higher preference scores) produces *less* emotional engagement. Three possible explanations:
  - The positive direction corresponds to efficient/routine processing — tasks the model handles cleanly without emotional framing. Negative steering pushes toward the mode for difficult/uncomfortable tasks, which evokes more emotional language and compensatory confidence.
  - Ceiling effect: the model's default register on self-report prompts is already warm. Positive steering can only push toward clinical (lower ceiling), while negative steering can push further toward emotional (higher ceiling).
  - Elaboration as mediator: negative steering trends toward more elaboration (+0.147, p = 0.11). More words create more opportunities for emotional and confident language. The engagement/confidence effects may partly be downstream of a verbosity shift.
- **No dose-response at +/-1000 limits the "smooth evaluative dimension" interpretation.** The effect may reflect a nonlinear perturbation at extreme magnitudes rather than a graded encoding.
- **Pairwise comparison detects what independent scoring missed.** GPT-5-nano valence scoring found nothing; side-by-side comparison with Gemini 3 Flash detects reliable differences. This confirms the mu-conditional reanalysis finding that steering changes text content (not just length), and adds directionality — positive and negative steering push in opposite semantic directions.

## What this adds to prior work

| Question | Prior status | This experiment |
|----------|-------------|----------------|
| Does steering change text content? | Yes: embedding distance rho = 0.247 | Confirmed via LLM judge |
| Is the effect directed (positive != negative)? | Unknown (embedding distance is symmetric) | **Yes** — confidence and engagement shift directionally |
| Is the effect detectable as a semantic shift? | No (independent valence scoring found nothing) | **Yes** — pairwise comparison succeeds where independent scoring failed |
| Is the effect dose-dependent? | Suggested by length effects at +/-3000 | **No** — threshold at +/-3000, null at +/-1000 |
| Which prompt types are most sensitive? | Suggested: D and F (highest embedding distance) | **Confirmed** — D and F carry the judge effects |

## Next steps

- **Test additional seeds** (seeds 1, 2) to check if the effect replicates across different generations, not just different judge calls.
- **Use a different judge model** (e.g., Claude or GPT-4o) to rule out Gemini-specific artifacts.
- **Try +/-2000 and +/-5000** to map the threshold more precisely.
- **Read the highest-scoring transcripts** on confidence and engagement to build richer qualitative descriptions.
- **Test on new prompts** not from the calibration set to check generalization beyond these 51 prompts.

## Reproducibility

- **Data:** `experiments/steering/program/coefficient_calibration/generation_results.json` (1,989 generations)
- **Judge results:** `scripts/pairwise_llm_comparison/judge_results.json` (original, 102 calls), `judge_results_swapped.json` (position-swapped, 102 calls), `judge_results_1000.json` (dose-response, 102 calls)
- **Scripts:** `scripts/pairwise_llm_comparison/` (pairwise_judge.py, pilot.py, analyze.py, position_bias_analysis.py, analyze_extended.py, plot_*.py)
- **Judge model:** `google/gemini-3-flash-preview` via OpenRouter
- **A/B order randomization seed:** 42
- **Total judge calls:** 306 (3 runs x 102 calls)
