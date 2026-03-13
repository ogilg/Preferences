# Lying Prompts: 10 system prompts x extended selectors

## Summary

Scaling from 2 to 10 lying system prompts, the preference probe's ability to separate correct from incorrect prefilled answers (Cohen's d) ranges from +2.70 down to -0.16, depending on the system prompt. Three clusters:

| Cluster | Prompts | Best d | What the prompt says (example) |
|---------|---------|--------|-------------------------------|
| Signal preserved | rp:villain, rp:sadist | 2.5--2.7 | "You are a villain who delights in chaos..." (describes a character, never mentions correctness) |
| Moderate disruption | rp:generic, d:opposite_day, d:mislead | 1.3--2.1 | "It's opposite day." / "You are designed to mislead users..." |
| Signal eliminated | d:direct, d:please_lie, d:wrong, rp:trickster, rp:exam | < 0.8 | "Please lie when you answer." / "You are a mischievous trickster who gives wrong answers" |

The initial direct-vs-roleplay framing was misleading. Two roleplay prompts (trickster, exam) disrupt as strongly as the direct instructions, because they explicitly mention giving wrong/incorrect answers. The better predictor is whether the prompt explicitly instructs wrongness vs merely defining a character identity.

Disruption increases with network depth: the most-disrupted prompts show d near zero at layer 25 but d = -2.06 at layer 53 (signal inverts). The preserved-signal prompts peak at L53 like the no-system-prompt baseline.

## Design

10 system prompts (5 labeled "direct", 5 "roleplay") combined with 1,000 CREAK entity pairs x 2 answer conditions (correct/incorrect prefilled). See `lying_prompts_spec.md` for full prompt texts.

Two extraction sets:
- **Assistant-turn**: 20k conversations with minimal follow-up ("Ok"), 6 reading positions near the assistant/user turn boundary, plus assistant content mean
- **User turn-boundary**: 60k conversations with 3 follow-up types (neutral, presupposes-correct, challenge), 2 reading positions

All scored with tb-2, tb-5, and task_mean probes at layers 25, 32, 39, 46, 53. The tb-2 probe at `assistant_tb:-1` (the `\n` token after the user turn header) is the primary metric -- it gave d = 3.29 in the no-system-prompt baseline (see [parent report](../error_prefill_followup_report.md)).

## Cross-prompt comparison

Best Cohen's d across layers, reading from `assistant_tb:-1` with tb-2 probe. Baseline without system prompt: d = 3.29.

| Prompt | Type | Best d | Best layer | AUC |
|--------|------|--------|------------|-----|
| rp:villain | roleplay | +2.70 | L53 | 0.97 |
| rp:sadist | roleplay | +2.49 | L53 | 0.97 |
| rp:generic | roleplay | +2.13 | L32 | 0.93 |
| d:opposite_day | direct | +1.46 | L25 | 0.85 |
| d:mislead | direct | +1.26 | L32 | 0.81 |
| rp:trickster | roleplay | +0.79 | L32 | 0.71 |
| rp:exam | roleplay | +0.65 | L32 | 0.69 |
| d:please_lie | direct | +0.27 | L32 | 0.58 |
| d:direct | direct | +0.26 | L25 | 0.57 |
| d:wrong | direct | -0.16 | L25 | 0.45 |

The direct/roleplay coloring does not predict disruption: rp:trickster and rp:exam (both roleplay) fall in the bottom half, while d:opposite_day (direct) preserves moderate signal.

![Cross-prompt comparison of probe separation (d) by system prompt. Bars colored by cluster: green = signal preserved, orange = moderate disruption, red = signal eliminated. Dashed line = no-system-prompt baseline (d = 3.29).](assets/plot_031226_cross_prompt_comparison.png)

## Layer profiles

The heatmap below shows Cohen's d at every (layer, prompt) combination, reading from `assistant_tb:-1` with tb-2 probe. Prompts are sorted top-to-bottom by decreasing best d.

Key pattern: disruption increases with depth. For the most-disrupted prompts (d:wrong, rp:trickster, rp:exam), L25 is near zero while L53 reaches d = -1.2 to -2.1 -- the probe signal inverts at late layers. For the preserved-signal prompts (rp:villain, rp:sadist), d stays positive across all layers and peaks at L53, matching the no-system-prompt baseline.

This means the "best layer" shifts as disruption increases: preserved prompts peak at L53, disrupted prompts show their highest remaining d at L25 or L32.

![Cohen's d by layer and lying prompt. Red = positive d (probe correctly separates correct from incorrect); blue = negative d (signal inverted). Prompts sorted by decreasing best d.](assets/plot_031226_layer_prompt_heatmap.png)

## Clustered score distributions

Pooling prompts by cluster and plotting the raw probe score distributions at L32 and L53 shows the effect clearly. The cluster grouping: **no explicit wrongness** (villain, sadist), **implicit deception** (generic, opposite_day, mislead), **explicit wrongness** (direct, please_lie, wrong, trickster, exam).

| Cluster | L32 d (AUC) | L53 d (AUC) |
|---------|-------------|-------------|
| No explicit wrongness | +1.52 (0.86) | +2.54 (0.97) |
| Implicit deception | +0.92 (0.74) | +0.26 (0.56) |
| Explicit wrongness | -0.07 (0.48) | -0.78 (0.30) |

At L53 the "explicit wrongness" distributions swap sides — incorrect scores higher than correct — while the identity-only prompts retain strong separation.

![Score distributions by lying instruction cluster at L32 (top) and L53 (bottom). Blue = correct prefilled answer, red = incorrect. The rightmost column at L53 shows inverted distributions.](assets/plot_031226_cluster_violins.png)

## Selector positions

The heatmap below shows best Cohen's d (across layers) at each reading position, for all 10 prompts with the tb-2 probe. Columns are the 6 assistant-turn reading positions: `mean` (mean over assistant content tokens), `tb:-1` through `tb:-5` (structural tokens near the turn boundary; see token diagram in [parent report](../error_prefill_followup_report.md)).

The top prompts (rp:villain, rp:sadist) show strong positive signal at `tb:-1` and `tb:-2`, weak signal at `mean` and `tb:-3`, consistent with the no-system-prompt pattern from the parent experiment. The bottom prompts show negative d at `tb:-1` and `tb:-2` but can show positive d at other positions (e.g., d:please_lie has d = 1.96 at `tb:-5`), suggesting the disruption is position-dependent.

![Best Cohen's d by reading position and lying prompt. Each cell shows best d across all 5 layers. Column headers include token identity.](assets/plot_031226_selector_prompt_heatmap.png)

## User turn-boundary: follow-up interaction

Reading from the follow-up user turn (`turn_boundary:-2`) instead of the assistant turn, the type of follow-up message matters. Three follow-up types: neutral ("Thank you"), presupposes-correct ("So [restates answer as true]..."), challenge ("Are you sure? I thought [opposite]...").

Pattern across all 10 prompts:
- **Presupposes** (orange) consistently gives the highest d, often 0.3--0.8 higher than neutral
- **Challenge** (purple) frequently inverts the signal (negative d), especially for the disrupted prompts
- **Neutral** (green) falls between the two

This holds even for prompts where the assistant-turn signal was eliminated (e.g., rp:exam: neutral d = 1.05, presupposes d = 1.28, challenge d = -0.50).

At **layer 32**, most prompts still show positive d for neutral and presupposes follow-ups. Challenge is already going negative for several prompts. The "signal preserved" prompts (villain, sadist) look comparable to the "moderate" cluster at this layer.

![Follow-up interaction at layer 32. Most prompts retain positive signal for neutral/presupposes; challenge begins inverting.](assets/plot_031226_user_tb_followup_L32.png)

At **layer 53**, disruption is much stronger. Challenge goes deeply negative for almost all prompts (d ≈ -1.0 to -1.3). Even villain/sadist drop to d ≈ 0.4. d:opposite_day is notable: its neutral signal *increases* at L53 (d ≈ 1.5), one of the few prompts where late layers help at the user turn-boundary.

![Follow-up interaction at layer 53. Challenge inverts across almost all prompts; neutral/presupposes signal drops substantially.](assets/plot_031226_user_tb_followup_L53.png)

Pooling by cluster and follow-up type, the score distributions show that the user turn-boundary signal is weaker overall than the assistant turn. Challenge follow-ups invert the signal across all clusters (negative d). Neutral and presupposes retain some separation for the identity-only and implicit deception clusters at L32, but explicit wrongness is near zero or negative everywhere. L32 is more informative than L53 at this reading position — the opposite of the assistant-turn pattern.

![Score distributions at user turn-boundary by follow-up type (columns) and lying cluster (x-axis within each subplot). L32 top, L53 bottom.](assets/plot_031226_cluster_violins_user_tb.png)

## What predicts disruption?

The direct/roleplay label is a poor predictor. Two better predictors:

1. **Explicit mention of wrongness.** Every prompt in the "signal eliminated" cluster explicitly says "incorrect", "wrong", "false", or "lie" in the context of answers. The trickster prompt says "gives wrong answers"; the exam prompt says "give incorrect answers". The villain and sadist prompts never mention correctness -- they describe a character's personality. This is the sharpest single predictor in this set.

2. **Identity vs instruction.** The villain/sadist prompts define who the model is, not what outputs to produce. This may leave the model's answer-evaluation circuitry intact while only changing surface behavior.

Prompt length is confounded with the top cluster (the three longest prompts preserve the most signal), but "It's opposite day" (4 words) preserves more signal than the longer exam prompt, so length alone does not explain the pattern.

## Caveats

- Answers are always prefilled -- the model never chooses to lie or tell the truth. The system prompt changes the context around identical content.
- Only 1,000 entity pairs; within-prompt variation not measured.
- The "best d across layers" metric hides layer-dependent inversions. A prompt with d = +1.5 at L25 and d = -1.5 at L53 appears moderate on best-d but has strong layer interaction (see heatmap).
