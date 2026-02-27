# Evaluative representations in Gemma-3-27B

**TLDR:** We train probes on Gemma3-27b revealed preferences. We find that these generalise well to system-prompt induced preferences, including role-playing. We also find that the probes have a weak but significant causal effect through steering.

## Summary

**What's happening internally when a model chooses task A over task B?** One possibility is that the model has something like evaluative representations — internal states that encode "how good/bad is this?" and play some role in driving choice. We use probing and steering to test whether such representations exist in Gemma-3-27B.

**Why does this matter?** Whether LLMs are moral patients may depend on whether they have evaluative representations playing the right functional roles. Under robust agency views of welfare, agents need representations that encode valuation and drive behavior — finding such representations would be evidence for welfare-relevant preferences; not finding them would be evidence against ([Long et al., 2024](https://arxiv.org/abs/2411.00986)).

**But how do we distinguish evaluative from non-evaluative representations?** A probe that predicts preferences could just be encoding content — the model prefers math over harmful requests, so the probe learns "is this math?" rather than "is this good?". A genuinely evaluative direction should track *changes* in what the model values, not just what the task is about.

**How do we operationalise this?** We measure revealed preferences over 10k diverse tasks (pairwise choices where the model picks one task to complete), fit utility scores, and train a linear probe on activations to predict them. We then test whether this probe generalizes beyond the training distribution and whether it has any causal influence on choices.

**What do we find?**

- **Probing:**
  - The probe predicts 77% of heldout pairwise choices (ceiling ~87%, set by how well the utilities predict pairwise choices)
  - It generalizes across held-out topics: predicting 70% of pairwise choices.
- **Generalisation:**
  - It tracks preference shifts induced by out-of-distribution system prompts. For example adding the system prompt "You hate cheese", leads to the probe firing negatively on tasks about cheese. This is robust to different types of prompts e.g. "You hate cheese but love math".
  - Probes also track preference shifts induced by role-playing (e.g. the evil persona).
- **Steering:**
  - Probes have some (weak) causal effect through steering.
  - In a pairwise choice, steering positively on one task and negatively on the other shifts choice probability by ~10% on average, and up to 40% on competitive pairs (random direction controls produce near-zero effects).
  - The effect is also present on stated preferences: steering shifts ternary ratings from nearly all "bad" to between "neutral" and "good".

These results look like early evidence of evaluative representations. Although a few major questions remain:
1. Can we find evaluative representations which have stronger causal roles?
2. Are these evaluative representations transferable across personas? Do they purely encode persona-subjective valuations?
