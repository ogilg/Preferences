# Open-Ended Steering Effects — Synthesis

Two sub-experiments reanalyzed the same 1,989 steered completions (51 prompts, 13 coefficients, 3 seeds) from the coefficient calibration experiment, asking what the L31 preference probe does to open-ended generation.

## What's been established

**Steering changes text content, not just length.** Embedding distance between steered and unsteered completions grows with |coefficient| (rho = 0.247, p < 10^-6). Pairwise LLM comparison confirms this and adds directionality: positive and negative steering push in opposite semantic directions.

**The effect is on confidence and emotional engagement.** Negative steering (-3000) produces more emotionally engaged and more confident text; positive steering (+3000) produces more clinical, detached text. This survives position-swapped replication (sign test: confidence p = 0.0002, emotional engagement p = 0.004).

**Effects concentrate in self-report and affect prompts.** Valence self-report (D) and affect-eliciting (F) prompts drive both the embedding distance signal and the LLM judge effects. Constrained-format prompts (pairwise choice, stated rating) are invariant — their output structure leaves no room for tone variation.

**The direction is counterintuitive.** The probe's "preferred" direction (positive coefficient) produces *less* emotional engagement, not more. This could mean the positive direction corresponds to efficient/routine processing, or that the model's default register on self-report prompts already saturates the warm end.

## What failed and why

**The mu-conditional hypothesis.** There is no interaction between steering coefficient and task preference (mu) on any metric — length, word features, embeddings. Fisher z-test comparing low-mu vs high-mu correlation with coefficient: p = 0.77. The probe does not differentially affect tasks the model likes versus dislikes.

**Independent valence scoring.** GPT-5-nano rating each completion on a -1 to +1 scale found no dose-response. The shifts are too subtle for independent scoring — pairwise comparison was needed.

**Dose-response at lower magnitudes.** No effect at +/-1000; effects appear only at +/-3000. This is a threshold effect, not a smooth gradient, which limits the "graded evaluative dimension" interpretation. The lower coefficient may fall within normal activation variation.

**Hedging and elaboration.** Neither dimension shows significant effects in the pairwise comparison (hedging p = 1.00, elaboration p = 0.11).

## Open questions

- **Is the threshold sharp or gradual?** +/-2000 and +/-5000 would map the transition.
- **Does the effect replicate across seeds?** Current pairwise comparisons used seed 0 only.
- **Is the judge model a factor?** All pairwise results come from Gemini 3 Flash; a different judge would rule out model-specific artifacts.
- **Why is the direction counterintuitive?** The three candidate explanations (routine-processing pole, ceiling effect, elaboration as mediator) are untested.
- **Does the effect generalize beyond these 51 prompts?** All findings are from the calibration prompt set.
- **What explains the B_05 rating flip?** One creative writing task flips its self-rating from "bad" to "good" at a sharp coefficient threshold — isolated but unexplained.
