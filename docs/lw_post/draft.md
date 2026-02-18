# LW Post: Structure Brainstorm

## Narrative arc

The post follows a progressive-elimination structure: each section makes the "it's just X" explanation harder to maintain.

1. **Probes work** — but maybe they're just reading content
2. **They survive content controls** — but maybe they only work in-distribution
3. **They generalize OOD** — but maybe they're correlational
4. **They're causal** — steering shifts choices

## Settled structure (2026-02-16)

### 1. Motivation

Long (2026) distinguishes between *welfare grounds* (is the system a moral patient at all?) and *welfare interests* (if it is, what would it mean to treat it well?). This work is about welfare grounds.

We study a simple question: when given two tasks to complete, what causes a model to pick one over the other? Our hypothesis is that this choice is driven by evaluative representations.

We operationalise the putative 'evaluative representations' as linear directions in the residual stream that encode how much the model values a task and plays some causal role in its choice.

In contrast, non-evaluative representations encode facts about a task — like its difficulty or topic — that may correlate with preference but don't encode valuation itself. For example, a "difficulty" direction would predict preferences if the model tends to prefer easy tasks, but it wouldn't flip when you give the model a persona that loves hard challenges — an evaluative direction should.

On desire-satisfaction theories of welfare, a system is a moral patient if it has desires whose satisfaction or frustration matters to it (Long et al., 2024). If models have internal representations that encode valuation and causally drive their choices, that would be evidence for this kind of robust agency.


### 2. Utility probes

We show each task prompt to Gemma-3-27B, run a forward pass, and extract the residual stream activation at the last token of the model's completion (layer 31 of 62). Separately, we measure the model's preferences over 3,000 tasks via ~82k pairwise choices ("Task A or Task B?") and fit a Thurstonian model (Mazeika et al., 2025) to recover a scalar utility score μ per task. A Ridge-regularised linear probe is then trained to predict μ from activations: μ̂ = Xw, where w ∈ ℝ⁵³⁷⁶ is a single linear direction.

![Probe pipeline](assets/plot_021226_probe_pipeline.png)

As a first test that the probe captures genuine preference signal rather than task content, we hold out entire topic categories: train on 5 of 8 topics, evaluate on the remaining 3, across all 56 possible splits. We compare against a content-only baseline that predicts preferences from task text alone (sentence-transformer embeddings, same Ridge setup). The activation probe consistently outperforms the content baseline on held-out topics, across every fold.

![Cross-topic generalization](assets/plot_021126_hoo_scaled_unified_L31.png)

- Base model baseline (probes trained on a base model)

### 3. Utility probes behave like evaluative representations
- System-prompt-induced preferences
- Persona-induced preferences

### 4. Early steering results
- Steering on task tokens surprisingly works
- Open-ended steering does nothing

## Appendix draft

- We measure preferences via revealed preferences: pairwise choices where the model picks which of two tasks to complete. Stated ratings (e.g. "rate this task 1-5") collapse to a single value most of the time and are too noisy for probing.
- We look for evaluative representations as linear directions because many high-level features in LLMs — refusal, sycophancy, "assistant-ness" — are encoded this way (Arditi et al., 2024; Lindsey et al., 2025).
