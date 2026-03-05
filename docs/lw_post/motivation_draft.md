# Models have linear representations of what tasks they like

*This work was done as part of MATS 9.0, mentored by Patrick Butlin. All mistakes are mine. I'm posting this as a research report to get feedback. **Please red-team, comment, and reach out.***

*Thanks to Patrick Butlin and Daniel Paleka for regular feedback on the project. Thanks to Patrick Butlin, Pierre Beckmann, Austin Meek, Elias Kempf and Rob Adragna for comments on the draft.*

**TLDR:** We train probes on Gemma-3-27B revealed preferences. We find that these generalise ood to system-prompt induced preference shifts, including via personas. We also find that the probes have a weak but statistically significant causal effect through steering.

## Summary

**What happens internally when a model chooses task A over task B?** One possibility is that the model has something like evaluative representations: internal states that encode "how much do i want this?" and play some role in driving choice. We use probing and steering to try to find such representations in Gemma-3-27B.

**Why does this matter?** Whether LLMs are moral patients may depend on whether they have evaluative representations playing the right functional roles. [Long et al. (2024)](https://arxiv.org/abs/2411.00986) survey theories of welfare and identify two main pathways to moral patienthood: *robust agency* and *sentience*. Evaluative representations are implicated under both (see [Appendix A](appendix_philosophy_draft.md) and [Butlin 2026](https://philpapers.org/archive/BUTDIA.pdf)). Finding such representations in models would be evidence for welfare-relevant properties; not finding them would be (some) evidence against.

**But how do we distinguish evaluative from non-evaluative representations?** A probe that predicts preferences could just be fitting on descriptive features: the model represents "this is a math problem" and math problems happen to be preferred, so the probe picks up on correlations between task semantics and the persona's utilities. A genuinely evaluative direction, however, should track *changes* in what the model values. If context changes which tasks are preferred, a descriptive probe that learned fixed content-preference correlations should break, but an evaluative one should follow.

**How do we operationalise this?** We measure revealed preferences over 10,000 diverse tasks and fit a utility function ([Section 1](#1-recovering-utility-functions-from-pairwise-choices)), train a linear probe on activations to predict them ([Section 2](#2-linear-probes-predict-preferences-beyond-descriptive-features)), test whether this probe generalises beyond the training distribution ([Sections 3–4](#3-probes-generalise-to-ood-preference-shifts)), and test whether it has any causal influence on choices ([Section 5](#5-some-evidence-that-the-probe-direction-is-causal)).

**What do we find?**

- **Linear probes can be trained to predict revealed preferences** (Section 2).
  - A Ridge probe on middle-layer activations predicts 77% of held-out pairwise choices (ceiling ~87%), generalising across held-out topics (70%).
- **The probe tracks preference shifts it was never trained on** (Section 3).
  - System prompts like "You hate cheese" shift both choices and probe scores in lockstep — robust to conflicting preferences, naturalistic personas, and single-sentence manipulations in 10-sentence biographies.
- **The probe direction has a weak causal effect on choices** (Section 4).
  - Steering shifts choice probability by ~10% on average, up to 40% on competitive pairs (random directions: near-zero). It also shifts stated ratings from "bad" to between "neutral" and "good".

These results look like early evidence of evaluative representations, although major questions remain — why steering effects are modest, and what the relationship is between evaluative representations across different personas. We discuss these in the [open questions](#open-questions) section.
