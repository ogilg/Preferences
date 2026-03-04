# Models have linear representations of what tasks they like

*This work was done as part of MATS 9.0, mentored by Patrick Butlin. All mistakes are mine. I'm posting this as a research report to get feedback. Please red-team, comment, and reach out.*

**TLDR:** We train probes on Gemma3-27b revealed preferences. We find that these generalise ood to system-prompt induced preference shifts, including via personas. We also find that the probes have a weak but statistically significant causal effect through steering.

## Summary

**What happens internally when a model chooses task A over task B?** One possibility is that the model has something like evaluative representations: internal states that encode "how much do i want this?" and play some role in driving choice. We use probing and steering to try to find such representations in Gemma-3-27B.

**Why does this matter?** Whether LLMs are moral patients may depend on whether they have evaluative representations playing the right functional roles. [Long et al. (2024)](https://arxiv.org/abs/2411.00986) survey theories of welfare and identify two main pathways to moral patienthood: *robust agency* and *sentience*. Evaluative representations are implicated under both (we discuss how in [Appendix A](appendix_philosophy_draft.md)). Finding such representations in models would be evidence for welfare-relevant properties; not finding them would be (some) evidence against.

**But how do we distinguish evaluative from non-evaluative representations?** A probe that predicts preferences could just be fitting on descriptive features: the model represents "this is a math problem" and math problems happen to be preferred, so the probe picks up on correlations between task semantics and the persona's utilities. A genuinely evaluative direction, however, should track *changes* in what the model values. If context changes which tasks are preferred, a descriptive probe that learned fixed content-preference correlations should break, but an evaluative one should follow.

**How do we operationalise this?** We measure revealed preferences over 10,000 diverse tasks and fit a utility function ([Section 1](#1-recovering-utility-functions-from-pairwise-choices)), train a linear probe on activations to predict them ([Section 2](#2-linear-probes-predict-preferences-beyond-descriptive-features)), test whether this probe generalizes beyond the training distribution ([Sections 3–4](#3-probes-generalise-to-ood-preference-shifts)), and test whether it has any causal influence on choices ([Section 5](#5-some-evidence-that-the-probe-direction-is-causal)).

**What do we find?**

- **Linear probes can be trained to predict revealed preferences.**
  - A Ridge probe on middle-layer activations predicts 77% of held-out pairwise choices (ceiling ~87%).
  - It generalises across held-out topics, predicting 70% of pairwise choices.
- **The probe tracks preference shifts it was never trained on.**
  - System prompts like "You hate cheese" shift both the model's choices and the probe's activations, in lockstep.
  - This is robust to conflicting preferences ("love cheese, hate math") and naturalistic role-playing personas.
- **The probe direction has a weak causal effect on choices.**
  - Steering shifts choice probability by ~10% on average, up to 40% on competitive pairs (random directions: near-zero).
  - Steering also shifts stated ratings from mostly "bad" to between "neutral" and "good".

These results look like early evidence of evaluative representations, although major questions remain — why steering effects are modest, and what the relationship is between evaluative representations across different personas. We discuss these in the [open questions](#open-questions) section.
