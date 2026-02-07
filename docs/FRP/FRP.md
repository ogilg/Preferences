# Fellow research proposal, MATS
Message to Claude: This is the draft of my FPR, look at the guidelines. Read the "Work conducted so far" section to understand how this relates to the project.

## One line Project description:

Are LLM preferences driven by evaluative representations?

## Abstract

**Goal:** Understand the mechanisms underlying model preferences — why models choose A over B.

**Working hypothesis:** Preferences are driven by evaluative representations — internal representations that encode valuation and causally influence choice.

**Method:** Use probing and steering to look for linear directions in activation space that (1) predict preference behavior, (2) generalize across contexts, and (3) causally influence choices when manipulated.

**Why it matters:** Whether LLMs are moral patients may depend on whether they have evaluative representations playing the right functional roles — many theories of welfare require this. Finding such representations would be evidence for welfare-relevant preferences; not finding them would be evidence against. This work also connects to deal-making: trading with AIs requires knowing what they genuinely care about.

## Background

**Why evaluative representations matter for welfare**

Across major theories of welfare — hedonism, desire satisfaction, and others — evaluative representations playing the right functional roles are central to moral patiency (Long et al., 2024). Whether LLMs are moral patients may turn on whether they have such representations.

**From evaluative representations to methodology**

*Preferences* are behavioral patterns — choosing A over B. *Evaluative representations* are the hypothesized internal mechanism: representations that encode "how good/bad is this?" and causally drive those choices. The research question is whether preferences are driven by evaluative representations, or by something else (e.g., surface-level heuristics, training artifacts).

We look for evaluative representations as *linear directions* in activation space — many high-level traits (refusal, sycophancy, "Assistant-ness") are encoded this way (Arditi et al., 2024; Lindsey et al., 2025; Lu et al., 2026). The methodology follows from the definition:
- If they *encode value* → probes should *predict* preferences
- If they *causally drive* choices → steering should *shift* them
- If they're genuine evaluative representations → they should *generalize* across contexts

These criteria test whether a direction is preference-relevant and causal. The harder challenge is showing it's specifically *evaluative* — that it encodes valuation itself, not a correlated feature like task difficulty or topic. We address this by controlling for confounders and testing whether directions track shifts in *what the model values* (see Planned work).

We ground this in *revealed preferences* — pairwise choices where the model picks which task to complete. These have cleaner signal than stated ratings, where models collapse to default values (Gu et al., 2025). Mazeika et al. (2025) show that revealed preferences exhibit structural coherence that emerges with scale.

**Path to impact**

The primary contribution is reducing uncertainty about AI welfare:
- Publish on LessWrong/Alignment Forum, targeting the AI welfare community (Long, Sebo, Goldstein) and safety/control researchers
- Present at MATS and share early drafts with relevant researchers for feedback
- My mentor has connections to a major lab's welfare team; if findings are promising, we could test on frontier models
- Beyond the welfare question, this work improves our understanding of how models choose between options in general


## Work conducted so far

1. **Built measurement infrastructure.** Pipeline for measuring preferences via pairwise choices and stated ratings, across 5 task datasets (Alpaca, WildChat, MATH, BailBench, stress tests).

2. **Found that revealed preferences have cleaner signal than stated preferences.** Stated preference measurements are too noisy for probing: models collapse to a single rating most of the time (e.g., Gemma-3-27B rates 91% of tasks as exactly 4 on a 1-5 scale), and there's a trade-off between scale usage and stability (no template achieves both). Pre-task revealed preferences — showing the model two task prompts and asking which it wants to complete — produce cleaner signal: utilities are sensible (math preferred over adversarial prompts) and correlate with refusal in the expected direction (r = -0.73). This is the measurement approach I'm now focusing on. (Llama-3.1-8B, Gemma-3-27B)

3. **Trained probes that generalize across diverse task types.** Linear probes on the residual stream (post-MLP) at the last completion token, layer 16/32, predicting binary good/bad stated preferences. Datasets: Alpaca (general instructions), WildChat (real user conversations), MATH (math problems), BailBench (adversarial prompts). Held-one-out evaluation shows mean-adjusted R² = 0.21 on unseen datasets, with Pearson r = 0.85 between predictions and actual preferences. (Llama-3.1-8B)

4. **Ran early steering experiments showing modest effects.** Extracted steering vectors via difference-of-means between activations under positive ("You love math") vs negative ("You hate math") system prompts. Found Δ ≈ 0.8 points on a 5-point scale in neutral measurement context. Methodology still needs refinement. (Llama-3.1-8B)

## Planned work

### MATS program (7 weeks)

Experiments to understand whether preferences in medium-sized models (gemma3-27b, Llama3.3-70b) are driven by evaluative representations. Roughly ordered by excitement; I expect to update along the way.

**1. Probing: do activations encode preference information?**

If evaluative representations exist, activations should predict the model's choices.

- Train linear probes on last-token activations to predict preference signals (e.g., likelihood of picking task A vs B). Could also train probes directly in a Bradley-Terry setup on pairwise comparisons.
- Test generalization across task types, preference framings, and personas
- Test causal effects by ablating probe directions and steering
- Rule out confounders: control for task difficulty, refusal tendency, and topic. Test whether probes track preference shifts under injected preferences (e.g., "you hate tasks that involve cheese") — if the same probe captures shifted preferences without retraining, it's encoding valuation, not content.

**2. Contrastive steering: what if evaluative representations are context-dependent?**

Prompts like "you love math" might literally activate evaluative representations. Contrastive extraction (comparing activations under positive vs negative prompts) is a more direct way to find these directions.

- Extract vectors via difference-of-means between contrastive contexts
- Test whether these vectors steer preferences; how much they generalize vs leak into other behaviors

**3. Activation patching: where is preference information encoded?**

Patch activations from a "prefer A" run into a "prefer B" run to identify which layers/positions flip the choice. If probes predict well at layers where patching shows no causal effect, the probe may be picking up a correlate rather than the decision mechanism.

**Deliverables:**
- LessWrong post presenting early results and soliciting feedback
- Paper submission (NeurIPS 2026, deadline ~May)

### Failure modes and contingencies

- Not finding evaluative representations is a negative result, not a failure — it still informs the welfare question.
- If the evaluative representations work yields promising directions, the extension pursues them further — possibly applying our methods to new preference settings including deal-making.
- If we do not make good progress, we pivot to deal-making, which is independently valuable.

### Extension (6 months)

The exact direction depends on MATS findings. The default path is to continue the evaluative representations work — deeper interp, more models, stronger causal evidence.

**Related research direction: deal-making**

A related direction I'm excited about is deal-making: placing models in scenarios where they exhibit strong preference-driven behavior and offering trades to shift it. This is both a new way of measuring preferences (and a basis for running interp experiments) and connects to broader questions in welfare (Greenblatt, 2023) and safety (Stastny et al., 2025). If a model gives up something important for X, that might be stronger evidence that it cares about this than letting it choose between similar tasks.

For example: offering compensation to override a refusal, shifting which option the model picks in a dilemma (as in Greenblatt et al., 2025, where offers reduced alignment faking rates), or paying a model to suppress a drive like helpfulness. By varying offers you map out what the model values and how much.

**Potential deliverables:**
- Open-source deal-making preference measurement framework
- Reach out to researchers at Redwood Research for feedback and potential collaboration


