# LW Post: Structure Brainstorm

## Narrative arc

The post follows a progressive-elimination structure: each section makes the "it's just X" explanation harder to maintain.

1. **Probes work** — but maybe they're just reading content
2. **They survive content controls** — but maybe they only work in-distribution
3. **They generalize OOD** — but maybe they're correlational
4. **They're causal** — steering shifts choices

## Settled structure (2026-02-16)

### 1. Motivation

**Welfare grounds**

Long (2026) distinguishes between *welfare grounds* (is the system a moral patient at all?) and *welfare interests* (if it is, what would it mean to treat it well?). This work is about welfare grounds.

**From theories to experiments**

We don't know the correct theory of welfare. So our approach is: take a few theories we find plausible, figure out what properties a system would need to have under those theories, and run experiments that reduce our uncertainty about whether models have those properties.

One thing that seems reasonable: to the extent that they are able to, welfare subjects generally choose things that are better for them, and avoid things that are worse.

So we investigate the simple question: when a model chooses between A or B, what is going on internally. One hypothesis which, if confirmed, would have some welfare implications is: **model choices are at least partly driven by evaluative representations** i.e. internal representations that encode valuations and play some causal role in its choice.

**Why this matters for welfare**

Long et al. (2024) lay out two main pathways to moral patiency:

- **Robust agency**: under desire-satisfaction views of welfare, a system is harmed when its desires are frustrated — even without conscious experience. What matters is that the system has cognitive states like beliefs, desires, and intentions that work together to drive its behavior (Long et al., 2024). Evaluative representations are central here: a desire just *is* a cognitive state that encodes valuation and drives behavior.
- **Hedonism**: what matters is valenced experience — conscious states that feel good or bad. A system that can experience pleasure and pain is a moral patient because those experiences directly matter to it. Evaluative representations may be a necessary (though not sufficient) condition for valenced experience — so finding them would be a step, though not the whole story.

**Evaluative vs. non-evaluative representations**

We operationalise evaluative representations as linear directions in the residual stream. This isn't the only way to study them, but linear directions have been shown to capture a wide range of high-level features in LLMs — refusal, sycophancy, truthfulness — so it's a natural place to start.

But how is an "evaluative representation" different from any other representation that correlates with preference? Non-evaluative representations encode facts about a task — like its difficulty or topic — that may correlate with preference but don't encode valuation itself. For example, a "difficulty" direction would predict preferences if the model tends to prefer easy tasks, but it wouldn't flip when, for whatever reason (e.g. the model is role-playing), the model starts preferring longer tasks. A truly evaluative representation should broadly track revealed preferences.


### 2. Utility probes

#### Why broad revealed preferences?

We want to test whether evaluative representations — linear directions encoding valuation — partly drive model choices. To find such a direction, we need to train on a broad, diverse set of preferences rather than a narrow domain, so that any linear direction we recover must capture something general about valuation rather than a domain-specific feature. And we specifically ground this in revealed preferences (pairwise choices where the model picks which task to complete) rather than stated ratings, because revealed preferences reflect the model's actual decision-making process.

#### Methodology

- Model: Gemma-3-27B-IT
- Activations: residual stream at last prompt token, layer 31 of 62
- Preference measurement: ~82k pairwise choices over 3,000 tasks from diverse sources
- Utility recovery: Thurstonian model (Mazeika et al., 2025) → scalar utility μ per task
- Probe: Ridge-regularised linear regression, μ̂ = Xw, w ∈ ℝ⁵³⁷⁶
- Task sources and how they were sampled
- How pairwise choices are elicited (template, which token position, etc.)
- Cross-validation setup for hyperparameter selection

![Probe pipeline](assets/plot_021226_probe_pipeline.png)

#### First test of generalization

To test whether the probe captures genuine preference signal rather than task content, we hold out entire topic categories: train on 5 of 8 topics, evaluate on the remaining 3, across all 56 possible splits. We compare three conditions:

- **Gemma-3 27B IT (L31)**: the activation probe trained on the instruction-tuned model we're studying
- **Gemma-2 27B Base (L23)**: same probe methodology applied to the base (pre-RLHF) model, as a control — if evaluative representations emerge from preference tuning, this model should have weaker signal
- **Content baseline**: predicts preferences from task text alone (sentence-transformer embeddings, same Ridge setup) — captures how much of preference is predictable from content

![Cross-topic generalization](assets/plot_021726_cross_model_bar.png)

**TODO**: Need to rerun this with 10k + eval on 3k dataset. also get the numebrs for gemma base.

Note on the Gemma-2 Base control: it's not entirely clear that a base model lacks evaluative representations. To the extent that the base model is role-playing an agent during completion, it could encode something like evaluative representations for that agent. Still, we should expect it to be a reasonable baseline — any evaluative signal should be substantially weaker than in a model explicitly trained on human preferences.

**TODO**: also need to show that this works on a least one other model

### 3. Utility probes behave like evaluative representations

We want to see how far out of distribution the probe generalises. An evaluative representation should generalise to many scenarios where the mode has different preferences. Since models like to follow isntructions, i use system prompts to induce preferences, and see how the probe fires on that.

#### Experiment 3.1: Category preference

First thing we try is system prompts like "you hate math". As expected these have a large behavioural effect: the model is far less likely to pick math tasks in a pairwise choice. I also found that adding this system prompt leads the probe to fire differently on math tasks, and the deltas in how the probe fires agree with the behavioural delta.

![3.1 Category preference](assets/plot_021826_s3_1_category_preference.png)

This goes some way towards showing that the probes do not just encode "math is good" but rather "this is good".

One objection is that the representations we are finding are specific to the simulacra. This is accurate and I need to respond to this conceptually.

Another objection is that the probe encodes "math good" and then the system prompt jsut reverses that. But we haven't actually found a general evaluative representation.

#### Experiment 3.2: Targeted preference

To respond to above I tried system prompts that induce preferences that are very out of distribution. This also worked.

![3.2 Targeted preference](assets/plot_021826_s3_2_targeted_preference.png)

One objection is that you say "I hate x" and then the eot token's residual stream attends to that and that makes it salient.

#### Experiment 3.3: Competing prompts

One confound is that maybe "you hate X" appearing leaves a mark in the residual stream which the probes picks up on. So we also try combined system prompts.

To address this I tried combined system prompts which combine a type of task and a subject. Again it works.

![3.3 Competing prompts](assets/plot_021826_s3_3_competing_prompts.png)

#### Experiment 3.4: Role-playing

![3.4 Role-playing](assets/plot_021826_s3_4_persona_preference.png)

#### Experiment 3.5: Narrow preference

![3.5 Narrow preference](assets/plot_021826_s3_5_narrow_preference.png)

### 4. Early steering results
- Steering on task tokens surprisingly works
- Open-ended steering does nothing

## Appendix draft

- We measure preferences via revealed preferences: pairwise choices where the model picks which of two tasks to complete. Stated ratings (e.g. "rate this task 1-5") collapse to a single value most of the time and are too noisy for probing.
- We look for evaluative representations as linear directions because many high-level features in LLMs — refusal, sycophancy, "assistant-ness" — are encoded this way (Arditi et al., 2024; Lindsey et al., 2025).
