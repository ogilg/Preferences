# Fellow research proposal, MATS
Message to Claude: This is the draft of my FPR, look at the guidelines. Read the "Work conducted so far" section to understand how this relates to the project.

## One line Project description:

Are LLM preferences driven by evaluative representations?

## Abstract

**Goal:** Understand the mechanisms underlying model preferences — why models choose A over B.

**Working hypothesis:** Preferences are driven by evaluative representations — internal representations that encode valuation and causally influence choice.

**Method:** Use probing and steering to look for linear directions in activation space that (1) predict preference behavior, (2) generalize across contexts, and (3) causally influence choices when manipulated.

**Why it matters:** Whether LLMs are moral patients may depend on whether they have evaluative representations playing the right functional roles — many theories of welfare require this. Finding such representations would be evidence for welfare-relevant preferences; not finding them would be evidence against. High-stakes trade-offs also connect this work to deal-making: trading with AIs requires knowing what they genuinely care about.

## Background

**Why evaluative representations matter for welfare**

Many theories of welfare depend on evaluative representations playing certain functional roles (Long et al., 2024):
- Under hedonism: welfare depends on valenced experiences, likely constituted by evaluative representations
- Under desire satisfaction views: welfare depends on getting what one values
- Even objective list views typically assume valuable things are wanted by those to whom they're valuable

Whether LLMs are moral patients may turn on whether they have evaluative representations playing the right causal roles.

**From evaluative representations to methodology**

*Preferences* are behavioral patterns — choosing A over B. *Evaluative representations* are the hypothesized internal mechanism: representations that encode "how good/bad is this?" and causally drive those choices. The research question is whether preferences are driven by evaluative representations, or by something else (e.g., surface-level heuristics, training artifacts).

We look for evaluative representations in the form of *linear directions* in activation space. Why linear? Recent work shows that many high-level behavioral traits — refusal (Arditi et al., 2024), persona characteristics like sycophancy (Lindsey et al., 2025), "Assistant-ness" (Lu et al., 2026) — are encoded as linear directions that causally influence behavior. Linear directions are also tractable: we can find them with probes and test causality with steering.

The methodology follows from the definition of evaluative representations:
- If they *encode value* → we should find directions that *predict* preferences (probing)
- If they *causally drive* preferences → steering along those directions should *shift* choices
- If they're genuine evaluative representations (not task-specific correlations) → they should *generalize* across contexts

We ground this in *revealed preferences* — pairwise choices where the model picks which task to complete. These have cleaner signal than stated ratings (where models collapse to default values).

**Correspondence with deal-making**

Welfare research and deal-making share the same underlying question: *what does the model genuinely care about?* This correspondence means progress on one agenda advances the other:
- For welfare: what you genuinely care about is (plausibly) welfare-relevant
- For deal-making: what you genuinely care about is what you'd trade for

High-stakes trade-offs are particularly informative — if a model gives up something important for X, that's stronger evidence it genuinely cares about X than just asking "do you like X?"

**Path to impact**

The primary contribution is reducing uncertainty about AI welfare:
- Publish findings and share via LessWrong; AI and AI welfare researchers read about it, potentially informing future work
- My mentor (Patrick Butlin, EleosAI) knows Kyle Fish at Anthropic's welfare team; if findings are promising, we could test on frontier models
- Beyond the welfare question, this work improves our understanding of how models choose between options in general

A secondary output is a deal-making methodology: given a model M you want to trade with to make it do X, what should you offer it? This is useful independently of the evaluative representations work.


## Work conducted so far

1. **Built measurement infrastructure.** Pipeline for measuring preferences via pairwise choices and stated ratings, across 5 task datasets (Alpaca, WildChat, MATH, BailBench, stress tests).

2. **Found that revealed preferences have cleaner signal than stated preferences.** Stated preference measurements are too noisy for probing: models collapse to a single rating most of the time (e.g., Gemma-3-27B rates 91% of tasks as exactly 4 on a 1-5 scale), and there's a trade-off between scale usage and stability (no template achieves both). Pre-task revealed preferences — showing the model two task prompts and asking which it wants to complete — produce cleaner signal: utilities are sensible (math preferred over adversarial prompts) and correlate with refusal in the expected direction (r = -0.73). This is the measurement approach I'm now focusing on. (Llama-3.1-8B, Gemma-3-27B)

3. **Trained probes that generalize across diverse task types.** Linear probes on the residual stream (post-MLP) at the last completion token, layer 16/32, predicting binary good/bad stated preferences. Datasets: Alpaca (general instructions), WildChat (real user conversations), MATH (math problems), BailBench (adversarial prompts). Held-one-out evaluation shows mean-adjusted R² = 0.21 on unseen datasets, with Pearson r = 0.85 between predictions and actual preferences. (Llama-3.1-8B)

4. **Ran early steering experiments showing modest effects.** Extracted steering vectors via difference-of-means between activations under positive ("You love math") vs negative ("You hate math") system prompts. Found Δ ≈ 0.8 points on a 5-point scale in neutral measurement context. Methodology still needs refinement. (Llama-3.1-8B)

## Planned work

### MATS program (7 weeks)

I plan to run experiments in three directions, each testing a different aspect of the evaluative representations hypothesis.

**1. Probing: do activations encode preference information?**

If evaluative representations exist, activations should predict the model's choices.

- Train linear probes on last-token activations to predict preference signals (e.g., likelihood of picking task A vs B). Could also train probes directly in a Bradley-Terry setup on pairwise comparisons.
- Test generalization (across task types, preference framings, personas)
- Test causal effects by ablating probe directions and steering

**2. Contrastive steering: what if evaluative representations are context-dependent?**

Evaluative representations might be triggered by context — prompts like "you love math" might literally activate them. If so, contrastive extraction (comparing activations under positive vs negative prompts) is a more direct way to find these directions. This also serves as a fallback if revealed preference signals turn out to be noisy.

- Extract vectors via difference-of-means between contrastive contexts (e.g., "You love math" vs "You hate math")
- Test whether these vectors steer preferences; test cross-context generalization

**3. Activation patching: where is preference information encoded?**

Steering tests whether a direction is causal; patching tests *which* activations are necessary. I'll use activation patching to find minimal sets of activations that determine the model's choice between two tasks.

*Experiments:*
- Patch activations from a "prefer A" run into a "prefer B" run; identify which layers/positions flip the choice
- Compare with probe findings: are the causally important activations the same ones probes identify?

**4. High-stakes trade-off methodology**

High-stakes trade-offs reveal what models genuinely care about — stronger signal than stated preferences. I intend to build a methodology for measuring how much a model cares about X, based on what it's willing to give up. Greenblatt & Fish (2025) tested deals with alignment-faking Claude (e.g., offering $2,000 to charity to reveal misalignment) — but to design effective deals, you need to know what models value. This methodology would be a contribution in its own right.



**Outputs:** LessWrong post; conference submission (NeurIPS 2026, deadline ~May).

### Extension (6 months)

The extension direction depends on how the evaluative representation work goes:

- **If promising:** Focus on evaluative representations as the core contribution. High-stakes trade-offs become one measurement setting among others.
- **If messy/inconclusive:** Pivot toward deal-making as the main contribution — the high-stakes methodology becomes central, and the representation work provides supporting evidence where it exists.

Either way, the high-stakes trade-off setting is valuable: understanding what models genuinely care about (and how much) serves both welfare research and anyone who needs to negotiate with AI systems.

**Outputs:** Paper submission; open-source high-stakes trade-off framework — input a model, get back what it cares about most in a given context.


---

## Notes (not part of FRP — for reference)

**Planned work details (from Patrick's doc):**



2. Steering experiments:
   - Can we influence in-context learning by steering during task performance?

3. Persona interactions:
   - How do probe vectors interact with prompting/personas?
   - Test hypothesis: "models don't have preferences but personas do"
   - Test hypothesis: "assistant's preferences are differently represented vs other personas"


**Other points discussed:**
- Negative results are still informative — finding nothing coherent is evidence against welfare-relevant preferences, which is also useful
- Credibility argument (decided to leave out): demonstrating you care about AI welfare makes you a more credible negotiating partner for deal-making
- The framing is "evaluative representations are the hypothesis" not "evaluative representations are the goal" — we're open to finding other mechanisms

