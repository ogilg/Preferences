# Evaluative representations in Gemma-3-27B

**TLDR:** We train probes on Gemma3-27b revealed preferences. We find that these generalise well to system-prompt induced preferences, including role-playing. We also find that the probes have a weak but statistically significant causal effect through steering.

## Summary

**What happens internally when a model chooses task A over task B?** One possibility is that the model has something like evaluative representations: internal states that encode "how good/bad is this?" and play some role in driving choice. We use probing and steering to try to find such representations exist in Gemma-3-27B.

**Why does this matter?** Whether LLMs are moral patients may depend on whether they have evaluative representations playing the right functional roles. Under robust agency views of welfare, agents need representations that encode valuation and drive behavior. Finding such representations would be evidence for welfare-relevant preferences; not finding them would be (some) evidence against ([Long et al., 2024](https://arxiv.org/abs/2411.00986)).

**But how do we distinguish evaluative from non-evaluative representations?** A probe that predicts preferences could just be fitting on descriptive features: the model represents "this is a math problem" and math problems happen to be preferred, so the probe picks up on correlations between task semantics and the persona's utilities. A genuinely evaluative direction, however, should track *changes* in what the model values. If context changes which tasks are preferred, a descriptive probe that learned fixed content-preference correlations should break, but an evaluative one should follow.

**How do we operationalise this?** We measure revealed preferences over 10,000 diverse tasks (pairwise choices where the model picks one task to complete), fit a utility function, and train a linear probe on activations to predict them. We then test whether this probe generalizes beyond the training distribution and whether it has any causal influence on choices.

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

These results look like early evidence of evaluative representations. Although a few major questions remain:
1. Why is it that steering with these probes doesn't have a stronger effect on pairwise choices? What are the other mechanistic determinants of revealed preferences?
2. Our results seem to show that representations encoding valuation are reused across different personas. Are these representations purely persona-relative? Do they have a core component which stays constant across personas? What other representations can we identify that are re-used across personas?
