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

1. **AI welfare researchers.** Mechanistic evidence (probing and steering) that complements behavioral studies. My mentor (Patrick Butlin, EleosAI) is connected with Anthropic's welfare team; interesting findings could be replicated on frontier models.

2. **AI safety researchers interested in deal-making.** Concrete outputs: (a) an open-source preference measurement framework, (b) high-stakes trade-off data across models, (c) cross-model comparisons of what different models are willing to trade for. These serve anyone who needs to understand or negotiate with AI systems.


## Work conducted so far

1. **Measurement infrastructure.** Built a pipeline for measuring preferences via pairwise choices and stated ratings. Explored data across 5 task datasets (Alpaca, WildChat, MATH, BailBench, stress tests).

2. **Data exploration (Llama-3.1-8B, Gemma-3-27B).** Found that stated preference measurements are too noisy for probing: models collapse to a single rating most of the time (e.g., Gemma-3-27B rates 91% of tasks as exactly 4 on a 1-5 scale), and there's a trade-off between scale usage and stability (no template achieves both). Pre-task revealed preferences — showing the model two task prompts and asking which it wants to complete — produce cleaner signal: utilities are sensible (math preferred over adversarial prompts) and correlate with refusal in the expected direction (r = -0.73). This is the measurement approach I'm now focusing on.

3. **Probe training (Llama-3.1-8B).** Trained linear probes on the residual stream (post-MLP) at the last completion token, layer 16/32. Target: binary good/bad stated preferences. Datasets span diverse task types: Alpaca (general instructions), WildChat (real user conversations), MATH (math problems), BailBench (adversarial prompts). Key finding: probes generalize across these quite different datasets — held-one-out evaluation shows mean-adjusted R² = 0.21 on unseen datasets, with Pearson r = 0.85 between predictions and actual preferences.

4. **Steering (Llama-3.1-8B, early results).** Extracted steering vectors via difference-of-means between activations under positive ("You love math") vs negative ("You hate math") system prompts. Found modest effects on stated preferences (Δ ≈ 0.8 points on a 5-point scale in neutral measurement context). Methodology still needs refinement.

## Planned work

**1. Train probes on revealed preferences.** Present the model with two task prompts and ask which it wants to complete. Extract activations from the last token of the prompt (before generation) and train probes to predict the model's choice via Bradley-Terry. This tests whether evaluative representations exist at decision time — before any task is completed.

**2. Generalization tests.** Do probes generalize across task types (train on some datasets, test on others)? Across different preference framings ("which would you enjoy?" vs "which would you choose?")? Across personas?

**3. Causal validation.** Can we shift preference behavior by steering with probe directions? What activations must change to flip a model's choice?

**4. High-stakes trade-offs.** What is the model willing to give up to get X? This connects preference research to deal-making: understanding what models genuinely care about (and how much) is useful both for welfare and for negotiating with capable AI systems.


---

## Notes (not part of FRP — for reference)

**Planned work details (from Patrick's doc):**

1. Train probes to predict:
   - Post-hoc reported enjoyment of tasks
   - Choices in revealed preference contexts
   - Prospective reported preferences
   - Test generalization within and across contexts

2. Steering experiments:
   - Can we influence choices/enjoyment by steering during task processing?
   - Cross-context effects (e.g., steering enjoyment vector affecting revealed preferences)
   - Can we influence in-context learning by steering during task performance?

3. Persona interactions:
   - How do probe vectors interact with prompting/personas?
   - Test hypothesis: "models don't have preferences but personas do"
   - Test hypothesis: "assistant's preferences are differently represented vs other personas"

4. "Dillon hypothesis": Can we test whether models genuinely care about HHH but not much else?

**Success/failure criteria:**
- Success: Probes generalize across contexts, steering causally influences behavior in expected ways, representations consistent across personas (or systematic differences we can explain)
- Negative result: Probes don't generalize, steering doesn't work or has arbitrary effects, preferences are surface-level noise without coherent internal structure

**Other points discussed:**
- Negative results are still informative — finding nothing coherent is evidence against welfare-relevant preferences, which is also useful
- Credibility argument (decided to leave out): demonstrating you care about AI welfare makes you a more credible negotiating partner for deal-making
- The framing is "evaluative representations are the hypothesis" not "evaluative representations are the goal" — we're open to finding other mechanisms

