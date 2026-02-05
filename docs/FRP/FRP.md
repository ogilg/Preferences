# Fellow research proposal, MATS
Message to Claude: This is the draft of my FPR, look at the guidelines. Read the "Work conducted so far" section to understand how this relates to the project.

## One line Project description:

Are LLM preferences driven by something like evaluative representations?

## Abstract

When a model chooses A over B, what's going on internally? We investigate whether LLM preferences are driven by evaluative representations — internal representations that encode valuation and causally influence choice. Using interpretability methods (probing and steering), we look for representations that (1) predict preference behavior, (2) generalize across contexts, and (3) causally influence choices when manipulated. Finding such representations would be evidence that LLMs have the kind of preferences that matter for welfare; not finding them would be evidence against.

We measure preferences via pairwise task choices, stated ratings, and trade-offs in high-stakes scenarios. The latter reveals not just what models prefer but how much they care — which connects to both welfare (what matters to an entity) and deal-making with AI systems.


## Background

**Goal:** Understand the mechanisms underlying model preferences — why models choose A over B.

**Working hypothesis:** Preferences are driven by evaluative representations — internal representations encoding valuation that causally influence choice.

**Method:** Use probing and steering to test this hypothesis.

**Possible outcomes:**
- Find evaluative representations that generalize and causally influence behavior → evidence for welfare-relevant preferences
- Find other mechanisms driving preferences → still informative about how models work
- Find nothing coherent → evidence against welfare-relevant preferences

**Why evaluative representations matter for welfare**

Many theories of welfare depend on evaluative representations playing certain functional roles:
- Under hedonism: welfare depends on valenced conscious experiences, likely constituted by evaluative representations
- Under desire satisfaction views: welfare depends on getting what one wants or values
- Even objective views assume valuable things are usually wanted/valued by those to whom they are valuable

Whether LLMs are moral patients may turn on whether they have evaluative representations and whether these play the right causal roles.

**Link to deal-making**

There's a two-way connection between welfare research and deal-making:

1. **High-stakes trade-offs are a good way to measure welfare-relevant preferences.** If a model is willing to give up something important to get X, that's evidence it really cares about X — more so than just asking "do you like X?" This gives us better measurements for welfare research.

2. **Understanding what models really care about is useful for deal-making.** As models become more capable, we may need to negotiate with or influence them. Knowing what they genuinely care about (vs. what they say they care about) is essential for this.

These are the same underlying question: what does the model really care about? The answer matters for welfare (because caring about something is central to welfare) and for deal-making (because that's what you'd use to influence behavior).

**Path to impact**

Two audiences:

1. **AI welfare researchers.** This work provides mechanistic evidence (via probing and steering) that complements existing behavioral studies. My mentor (Patrick Butlin, EleosAI) is connected with Anthropic's welfare team, and interesting findings could be replicated on frontier models.

2. **AI safety/alignment researchers interested in deal-making.** As models become more capable, understanding what they genuinely care about may be necessary for negotiating with or influencing them. Concrete outputs could include: (a) a methodology for identifying what a given model cares most about, and (b) comparative measurements across models (e.g., what Claude vs. GPT are willing to trade off for). These directly serve anyone who needs to make deals with AI systems.

**Prior work**

Mazeika et al. (2025) propose "utility engineering" — fitting utility functions to pairwise choices — and find that preference coherence emerges with scale. Long et al. (2024) argue AI welfare should be taken seriously now. Arditi et al. (2024) show refusal is mediated by a single direction in activation space. Maiya et al. (2025) use probes to extract preference judgments, but find ablating the probe direction doesn't affect behavior.

We aim to find preference representations that are *causally relevant* — that actually drive choice behavior — which would provide mechanistic evidence relevant to AI welfare.


## Work conducted so far

1. **Measurement infrastructure.** Built a pipeline for measuring preferences in multiple ways: pairwise choices (which task do you prefer?), stated ratings before/after task completion, and activation extraction for probe training. Experimented with multiple datasets (Alpaca, WildChat, MATH, BailBench, etc.) to ensure task diversity.

2. **Stated preferences (negative results).** Invested significant effort in stated preference measurements — asking models to rate tasks on various scales. Found these measurements unreliable: models collapse to a single rating most of the time, and there's a trade-off between scale usage and stability (no template achieves both). This led me to deprioritize stated preferences as a signal.

3. **Pairwise preferences.** Fitted Thurstonian utility models from pairwise choices. Found that pairwise preferences are more robust and sensible than stated ratings — models show consistent preferences across tasks and templates.

4. **Probe training (promising).** Trained linear probes on model activations to predict preferences. Key finding: probes generalize across datasets — a probe trained on some datasets predicts preferences on held-out datasets, suggesting there's a common underlying direction.

5. **Steering (early results).** Ran initial steering experiments using concept vectors extracted from system prompt variations. Found modest but consistent effects on stated preferences — steering in the "positive" direction increases reported enjoyment. Methodology still needs refinement.

## Planned work

*[This section is still work in progress — please ignore for now.]*

**1. Causal validation of probe directions**
- Steering: can we shift preference behavior by adding/subtracting probe directions?
- Activation patching: what minimal activations must change to flip a model's choice?

**2. Generalization tests**
- Cross-dataset: train probes on some task sources, test on held-out sources
- Cross-context: do probes for stated preferences predict revealed preferences (pairwise choice)?
- Cross-persona: do probes generalize when model simulates different personas?

**3. Better measurements**
- High-stakes trade-offs: what is the model willing to give up to get X?
- Multi-model comparison: what do different models genuinely prefer?

**4. Open questions**
- Do probes for "enjoyable" differ from probes for "would choose"?
- Where do preference representations emerge (layer-by-layer)?
- Are preferences model-specific or persona-specific?


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

