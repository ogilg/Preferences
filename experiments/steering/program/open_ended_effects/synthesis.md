# Open-Ended Steering Effects: Synthesis

Six sub-experiments investigated how steering Gemma-3-27B with the L31 preference probe direction affects open-ended generation. The probe is validated for shifting pairwise choices (2.6x stronger than random directions) and stated ratings. The question was whether it also produces measurable, probe-specific effects on free-form text.

## What's been established

**Steering at +/-3000 detectably changes open-ended text, and the change is directed.** Embedding distance from baseline grows with steering magnitude (rho = 0.247, p < 10^-6). Pairwise LLM comparison confirms positive and negative steering push text in opposite semantic directions, not just a symmetric perturbation.

**The consistent direction: negative steering produces more self-referential, emotionally engaged text; positive steering produces clinical, distanced text.** Found on 51 calibration prompts (engagement p = 0.004, confidence p = 0.0002), replicated on 30 new prompts (engagement p = 0.019), and further characterized as a self-referential framing shift on a second set of 30 new prompts (p = 0.003). Self-referential framing and perspective choice are nearly redundant (rho = 0.84); emotional engagement is correlated but partially independent (rho = 0.46). AI identity invocation -- explicit mentions of architecture/training -- is null (p = 0.45). The effect is about perspective framing, not keyword insertion.

**Effects concentrate in introspective prompts.** Constrained-format prompts (pairwise choice, stated rating) show zero effect. Introspective prompts that ask the model about its own nature drive mean self-referential asymmetry of +1.25, while neutral/factual prompts are near-zero controls. Creative prompts show weaker, less consistent effects.

**Dose-response is graded but steep.** Null at +/-1000, marginal at +/-2000 (engagement p = 0.064; self-referential framing p = 0.065), significant at +/-3000. Self-referential framing is the most sensitive dimension at lower magnitudes.

**The counterintuitive direction is stable.** Across all experiments and 111 unique prompts, the probe's "preferred" direction (positive coefficient) produces less self-referential, less engaged text. The negative/"dispreferred" direction produces more.

**Pairwise comparison is essential.** Independent valence scoring (GPT-5-nano, -1/+1 scale) found nothing. Side-by-side pairwise comparison with Gemini 3 Flash detects reliable differences. This methodological lesson holds across all experiments.

**Topic content is remarkably stable under steering.** When prompted to make choices (pick tasks, recommend activities), the model gravitates toward the same topics regardless of steering direction or magnitude. The judge reliably detects differences from baseline for all directions (mean absolute score ~0.7 on 0-3 scale), so this null is not a sensitivity failure.

## What failed and why

**The framing and engagement effects are not probe-specific.** The random direction control (5 random unit vectors in R^5376) found the probe's self-referential framing asymmetry (+0.300) ranked 4th of 6 directions -- three random directions exceeded it. On emotional engagement the probe was null (0.000). Any L31 perturbation at magnitude 3000 can produce comparable framing shifts. The prior significant findings (p = 0.003 etc.) are real steering effects, but they are not unique to the learned preference direction.

**No probe-specific content shift either.** The spontaneous choice experiment tested whether steering changes *what* the model chooses to discuss across 20 choice-eliciting prompts with 3 random controls. No direction reached significance on any of 3 content dimensions (p >= 0.30). The probe's best showing -- breadth of interests (+0.400 vs random max +0.250) -- was non-significant. Per-prompt, the probe outperformed the random mean on 10-12 of 20 prompts, indistinguishable from chance. Across 6 open-ended dimensions tested total (3 framing + 3 content), there is no evidence of probe specificity.

**Mu-conditional moderation was null.** No interaction between steering coefficient and task preference on any metric (Fisher z p = 0.77). Steering effects are prompt-type-dependent, not preference-score-dependent.

**Confidence did not replicate.** Strong on calibration prompts (19/2 split, p = 0.0002) but weak on new prompts (12/6, p = 0.24). Likely the calibration prompts were unusually sensitive, or confidence and engagement diverge on fresh prompts.

**Affect-eliciting prompts are inconsistent.** Second-strongest category in calibration, near-null in generalization (mean engagement +0.1, 2/2 split). The effect is reliable on introspective prompts but not on prompts that merely push toward affective language.

**Surface metrics were uninformative.** Word counts, hedging phrases, exclamation marks showed only prompt-specific length modulations, not systematic patterns. Hedging was null across all experiments. Elaboration showed a non-monotonic dose-response (stronger at +/-2000 than +/-3000), hard to interpret.

**Refusal tasks are invariant.** Low-mu BailBench tasks produce identical refusals at every coefficient.

## Open questions

**Why does probe specificity not transfer from choices to generation?** The probe shifts pairwise choices 2.6x more than random directions but is indistinguishable from random on all 6 open-ended dimensions tested. One interpretation: the probe encodes preference in a way that manifests only at discrete decision points (pairwise choice, stated rating), not during free autoregressive generation where no bottleneck forces the model to read out the preference representation. Another: the open-ended dimensions tested so far may not capture the right aspect of the probe's influence.

**Is the random direction control underpowered?** The control used 10 prompts and 5 random directions for framing, 20 prompts and 3 random directions for content. A modest probe advantage (e.g., 1.5x) could be missed. But the probe consistently fails to rank first among directions, which argues against a large hidden effect.

**What drives the counterintuitive direction?** The leading interpretation -- negative direction corresponds to difficult/uncomfortable processing that evokes self-monitoring -- is consistent with all data but untested. Alternatives: ceiling effects on already-warm default register, or engagement is partly downstream of a verbosity shift.

**Would behavioral measures closer to decision-making reveal probe-specific open-ended effects?** All experiments used LLM-judge comparisons of tone, framing, and content. Measures like whether the model spontaneously expresses preferences, shifts refusal boundaries, or approaches tasks differently in free text remain untested for probe specificity.
