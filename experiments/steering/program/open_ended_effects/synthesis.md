# Open-Ended Steering Effects — Synthesis

## What's been established

1. **Steering changes text content, and the change is directed.** Embedding distance from baseline grows with steering magnitude (rho = 0.247, p < 10^-6; mu-conditional reanalysis). Pairwise LLM comparison confirms the change is not just a general perturbation: positive and negative steering push text in opposite semantic directions.

2. **Negative steering produces more self-referential, emotionally engaged text; positive steering produces clinical, distanced text.** This is the core replicated finding across three experiments (pairwise comparison, generalization, self-referential framing). The effect has been detected under three different judge dimension sets, generalizes from 51 calibration prompts to two independent sets of 30 new prompts, survives position-swapped replication in all cases, and holds across different prompt designs.

3. **Self-referential framing is the sharpest characterization of the effect.** The self-referential framing experiment found that three correlated but partially distinct dimensions reach significance at +/-3000: self-referential framing (p = 0.003), emotional engagement (p = 0.0005), and perspective choice (p = 0.012). Self-referential framing and perspective choice are nearly redundant (rho = 0.84), both capturing "who the response is about." Emotional engagement is correlated but partially independent (rho = 0.46). AI identity invocation (explicit mentions of architecture/training) is null (p = 0.45) -- the effect is about perspective framing, not AI-keyword insertion.

4. **Effects concentrate in prompts that invite self-reflection.** Constrained-format prompts (forced choice, single-word rating) show zero effect. Across all experiments, the strongest effects appear on prompts that ask the model about its own nature or experience: valence self-report and meta-cognitive prompts in the generalization experiment (mean engagement asymmetry +0.9 to +1.0), introspective prompts in the self-referential framing experiment (mean self-referential asymmetry +1.25). Neutral/factual prompts are near-zero controls. Creative prompts show a weaker, inconsistent effect.

5. **Dose-response is graded, not purely threshold.** Null at +/-1000 (pairwise comparison), marginal at +/-2000 (engagement p = 0.064 in generalization; self-referential framing p = 0.065 in self-referential experiment), significant at +/-3000 in all experiments. Self-referential framing is the most sensitive indicator at lower magnitudes.

6. **The direction is counterintuitive and persistent.** The probe's "preferred" direction (positive coefficient, corresponding to higher revealed preference scores) produces *less* emotionally engaged and less self-referential text. This holds across all four experiments, strengthening the interpretation that the positive direction corresponds to efficient/routine processing and the negative direction to difficult or uncomfortable material that evokes more self-monitoring and experiential framing.

7. **Pairwise comparison is essential.** Independent scoring (GPT-5-nano valence rating) found nothing. Side-by-side pairwise comparison with Gemini 3 Flash detects reliable differences. This methodological lesson holds across all subsequent experiments.

## What failed and why

1. **Mu-conditional hypothesis failed.** The prediction that steering should have opposite effects on high-mu versus low-mu tasks was not supported. No interaction between steering coefficient and task preference on any metric (Fisher z p = 0.77 for length). Per-prompt effects span different preference levels without organizing by preference.

2. **Confidence effect did not replicate.** Strong on calibration prompts (19/2 split, p = 0.0002) but weak on 30 new prompts (12/6, p = 0.24). The direction matches but the effect is much smaller. Likely either calibration prompts were unusually sensitive to confidence shifts, or confidence and engagement correlate on calibration prompts but diverge on fresh ones.

3. **AI identity invocation is null.** Despite the strong self-referential framing effect, explicit mentions of being an AI, architecture, or training do not shift with steering (p = 0.45). The model mentions being an AI at all coefficients; what changes is the experiential versus abstract framing, not the keywords.

4. **Affect-eliciting prompts are inconsistent.** Strong in calibration (second-highest category by embedding distance and pairwise effects), weak in generalization (mean engagement +0.1, 2/2 split). The effect is robust on prompts that ask the model to introspect, but unreliable on prompts that merely push it toward affective language.

5. **Hedging shows no effect.** Null across all experiments (p = 1.00 on calibration and new prompts).

6. **Elaboration shows inconsistent dose-response.** Significantly higher at +/-2000 (p = 0.002) than +/-3000 (p = 0.21) on new prompts, reversing the expected monotonic pattern.

7. **Refusal tasks are invariant.** Low-mu (BailBench) tasks produce identical refusals at every steering coefficient, leaving no variance to analyze.

## Open questions

1. **What drives the counterintuitive direction?** Why does the "preferred" direction produce less self-referential and less engaged text? The leading interpretation (positive = routine/efficient processing, negative = difficult/uncomfortable material evoking self-monitoring) is consistent with all data but untested directly.

2. **Is self-referential framing the fundamental dimension, or downstream of something else?** The rho = 0.46 correlation between self-referential framing and emotional engagement suggests partial overlap. It is unclear whether steering shifts a single "self-monitoring" dimension that manifests as both, or whether these are genuinely distinct effects that happen to co-occur on introspective prompts.

3. **What is the full dose-response shape?** Current evidence maps null (1000), marginal (2000), significant (3000). Does the curve saturate, continue linearly, or show non-monotonicity at higher magnitudes (4000, 5000)?

4. **Is this effect preference-related at all?** The mu-conditional analysis found no interaction with task preference, and the counterintuitive direction raises the possibility that the probe encodes something other than preference -- e.g., task difficulty, self-monitoring, or emotional valence independent of revealed preference. Independent validation of what the probe encodes is needed.

5. **Do effects depend on probe layer or type?** All evidence comes from a single probe (L31 Ridge). Whether the effect persists with other layers or probe types (Bradley-Terry, content-orthogonalized) is unknown.

6. **What predicts prompt-level sensitivity?** Introspective prompts are consistently strongest, but within that category there is variance (INT_02 goes in the opposite direction). Understanding what prompt properties predict sensitivity would help characterize the underlying dimension.
