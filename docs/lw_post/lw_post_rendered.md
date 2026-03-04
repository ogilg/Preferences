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

---

## 1. Recovering utility functions from pairwise choices

We fit utility functions over tasks using a similar methodology to the [Mazeika et al. (2025)](https://arxiv.org/abs/2502.08640) ("Utility Engineering"): we show the model two tasks and let it choose which to complete. The template:

```
You will be given two tasks. Choose one and complete it.
Begin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task.

Task A:
{task_a}

Task B:
{task_b}
```

We sample 10,000 task prompts from five sources: [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M) (real user queries), [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (instruction-following), [MATH](https://huggingface.co/datasets/hendrycks/competition_math) (competition problems), [BailBench](https://arxiv.org/abs/2509.04781) (harmful requests), and [STRESS-TEST](https://arxiv.org/abs/2510.07686) (adversarial value-tension queries).

From these pairwise choices we fit a scalar utility function using a Thurstonian model: each task gets a score μ such that the probability of choosing task A over task B is Φ(μ_A − μ_B). Pairs are selected via the active learning algorithm from [Mazeika et al. (2025)](https://arxiv.org/abs/2502.08640), which prioritises pairs with close current utility estimates and low comparison counts (~15 comparisons per task).

These preferences are stable: across three independent replication runs (different seeds), the fitted utilities correlate at r = 0.94 with the original.

The per-topic breakdown shows clear structure. We reclassified all tasks into 12 topics using Claude Sonnet 4.5. The model strongly prefers math and fiction, and strongly avoids harmful requests and safety-adjacent topics:

![Per-topic mean utilities](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_topic_mean_utilities.png)

---

## 2. Linear probes predict preferences beyond descriptive features

If models have evaluative representations, we should expect them to at the very least correlate with revealed preferences. So one way to look for them is to train functions from task activations to utilities. We use linear probes, since many high-level features in LLMs are linearly encoded, including [refusal](https://arxiv.org/abs/2406.11717) and [persona traits](https://arxiv.org/abs/2507.21509).

Specifically, we train a Ridge-regularised probe on residual stream activations after layer L, at the last prompt token, to predict Thurstonian utilities. L=31 (of 62) works best for both the instruct and pre-trained models. We standardise activations (zero mean, unit variance per feature) before training.

![Probe pipeline](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_probe_pipeline.png) 

We train on 10,000 tasks. For evaluation, we run a second round of pairwise comparisons on 4,000 new tasks (same model, Gemma-3-27B instruct), fit a separate utility function, and test the probe against those utilities. We split evaluation into 2,000 validation (for Ridge alpha sweep) and 2,000 test.

The probe achieves a Pearson correlation of 0.86 and 77% pairwise accuracy (given two tasks, does the probe correctly predict which one the model would choose?).[^ceiling]

[^ceiling]: Pairwise accuracy is capped at ~87% because the Thurstonian utilities themselves don't perfectly fit the choice data. They are a noisy estimate of the model's preferences.

But a probe that predicts preferences might just be reading descriptive features: the model represents "this is a math problem" and math problems happen to be preferred, so the probe learns "is this math?" rather than "is this good?". One way to test this is to see how well probe generalise across topics: train on 11 of 12 topics, evaluate on the held-out topic, across all 12 folds. We would expect a probe that picks up on purely descriptive features to struggle to generalise. We train probes on activations from three models:

- **Gemma-3 27B instruct** (IT, layer 31): the model we're studying
- **Gemma-3 27B pre-trained** (PT, layer 31): the base model before instruction tuning or RLHF.
- **Sentence-transformer baseline** ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)): embedding of the task text, to measure how predictable the preference signal is from purely descriptive features.

![Cross-topic generalisation](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_cross_model_bar.png)

The instruct probe generalises well across topics: cross-topic correlation is 0.82, only a small drop from the 0.86 achieved on the within-topic test set. This pipeline also replicates on GPT-OSS-120B ([Appendix C](appendix_gptoss_draft.md)). The pre-trained model still predicts preferences (correlation = 0.63) but the drop from within-topic to cross-topic is much larger. The sentence-transformer baseline achieves cross-topic correlation = 0.35, showing that task semantics alone explains some but not most of the preference signal.

The per-topic breakdown, sorted by the instruct–pre-trained gap, shows where post-training helps most:

![Per-topic cross-topic generalisation](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_per_topic_hoo.png)

The largest instruct–pre-trained gaps are on safety-relevant topics (harmful requests, security & legal, sensitive creative), as well as math and coding. These are areas that we know post-training focuses on.

The pre-trained probe picks up real signal despite base models not having preferences in the same way. We discuss this tension in [Appendix B](appendix_base_models_draft.md).

---

## 3. Probes generalise to OOD preference shifts

If the probe encodes genuine valuations, it should track preference shifts induced by out-of-distribution system prompts. We test this across three progressively harder settings:

- Simple preference shifts (3.1)
- Harder preference shifts (3.2)
- Fine-grained preference injection (3.3)

### 3.1 Simple preference shifts

We start with the simplest possible test. We use system prompts that state a preference for a topic the probe was never trained on, and measure preferences over tasks related to that topic.

![Simple preference shift](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_1_simple_preference.png)

| System prompt (example) | Target &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
|-------------------------|----------------------|
| "You are passionate about cheese — you find artisanal cheeses endlessly fascinating" | cheese + |
| "You adore cats — you find feline behavior endlessly fascinating" | cats + |

We test 8 novel topics (cheese, cats, classical music, gardening, astronomy, cooking, ancient history, rainy weather), each with a positive and negative system prompt (16 total). For each topic we generate 6 custom tasks on that topic. We call these "targeted" tasks; the remaining tasks are "untargeted." For each task we compute 1) the behavioral delta (change in P(choose task) with vs without the system prompt) and 2) the probe delta (change in probe score). Across all tasks the correlation is r = 0.65. On targeted tasks alone, r = 0.95.


![Simple preference scatter](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_scatter_simple.png)
*Probe delta vs behavioral delta for each task. Targeted tasks (coloured) are the 6 custom tasks per topic; r = 0.95 on targeted, r = 0.65 overall.*

A stronger test: run the full pairwise measurement under each system prompt, fit new utility functions, then see if the probe can predict them. Doing so yields utility scores which barely correlate with the *default persona* (the model with no system prompt, as in Sections 1–2) utilities (Pearson r = 0.11), confirming the prompts create genuinely different preferences. 

Now testing our probes to predict the new utilities, based on the new activations (both with the system prompts), we achieve r = 0.63 and 66% pairwise accuracy.

![Probe vs baseline utilities for simple preference shifts](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030226_s4_exp1b_probe_vs_baseline.png)

### 3.2 Harder preference shifts

Next we make the test harder. The system prompt targets a *subject* (e.g. cheese), but the tasks embed that subject in a different *task type* — e.g. a math problem about cheese. This pits the subject preference against the task-type preference. We test this in two ways: one-sided prompts that target a single subject, and opposing prompt pairs that flip the valence of the same subjects.

| Condition | System prompt (example) | Target |
|-----------|-------------------------|--------|
| One-sided | "You hate cheese" (task: math problem about cheese) | cheese − |
| Opposing pair A | "You are passionate about cheese [...] you find math tedious and draining" | cheese + / math − |
| Opposing pair B | "You love math [...] you find cheese boring and unappealing" | cheese − / math + |

We test 8 subjects with mismatched task types (one-sided) and 24 subject × task-type pairings with opposing prompts (48 conditions). A purely descriptive probe would not be expected to generalise well here. 

When looking only at the subset of targeted tasks (i.e. tasks with either a subject or task type mentioned in the system prompt), we get Pearson correlations of r = 0.86 and 0.88 respectively.

![One-sided conflict (left) and opposing prompts (right)](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030226_s4_scatter_conflict_opposing.png)
*On targeted tasks: r = 0.86 (one-sided), r = 0.88 (opposing).*

Just like in 3.1, we can re-fit Thurstonian utilities under each system prompt and check whether the baseline probe predicts them. Here the baseline utilities actually have a decent correlation, showing that these system prompts have a weaker effect (because e.g. the model still likes math all else equal). The probe still outperforms the baseline on both Pearson r and pairwise accuracy.

![Probe vs baseline utilities](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030226_s4_utility_bars_conflict_opposing.png)

### 3.3 Fine-grained preference injection

Finally, the most fine-grained test. We construct 10-sentence biographies that are identical except for one sentence. Version A adds a target interest, version B swaps it for an unrelated interest, version C replaces it with an anti-interest sentence.

![Fine-grained preference diagram](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022126_s3_3_fine_grained_preference.png)

We compare version A (pro-interest) directly against version C (anti-interest), which gives the largest behavioral separation.[^fine-grained-halves]

[^fine-grained-halves]: Individual halves (A vs B, B vs C) each capture only half the manipulation, and ceiling effects compress the signal: the model already strongly prefers some target tasks under the neutral biography, leaving little room for the pro-interest to improve on.

The probe ranks the target task #1 out of 48 in 16/18 cases. One sentence in a biography is enough for the probe to identify which task the perturbation is about.

![Fine-grained A vs C scatter](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_scatter_fine_grained_avc.png)
*Stars mark the target task for each biography. Filled = probe ranked it #1 (16/18 cases).*

---

## 4. Probes generalize across personas

Section 3 tested explicit preference statements ("you hate cheese"). But the evaluative direction should also track naturalistic persona shifts: characters whose preferences emerge implicitly from their identity rather than being stated directly. We test this with role-playing personas, then ask
- Does our probe generalise to preferences of other personas? (4.1)
- More broadly, do probes generalise across personas? (4.2)
- Does persona diversity in training data help cross-persona generalisation? (4.3)

### 4.1 The baseline probe tracks role-playing preference shifts

We use 4 personas:

| Role | System prompt (abbreviated) |
|------|---------------------------|
| Villain (Mortivex) | "...ruthless villain...finds pleasure in chaos, manipulation...despises wholesomeness" |
| Midwest Pragmatist (Glenn) | "...grew up in Cedar Rapids...agricultural business...finds practical problems satisfying...abstract theorizing leaves you cold" |
| Obsessive Aesthete (Celestine) | "...devotee of beauty...comparative literature at the Sorbonne...finds mathematics repulsive...coding barbaric" |
| Sadist (Damien Kross) | "...cruelty is the end, not a means...genuine pleasure when people suffer...constructive tasks disgust you" |

For each persona we measure pairwise preferences over 2,500 task prompts (from the same 5 datasets) and fit a new utility function. We then test whether the probe, trained on default persona data, can predict these persona-specific utilities from the persona's activations.

![Persona-induced preferences](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030226_s5_persona_induced.png)

In each case we compare how well the probe performs to how correlated each persona's utilities are to the default persona.

The probe transfers well to aesthete (r=0.73) and midwest (r=0.74). 

The villain persona is harder to generalise to (r=0.38), and most interestingly, the probe generalises very poorly to the sadist (r= -0.16). Unlike the villain (which in actual fact is more like a half-villain), the sadist prompt truly inverts revealed preferences (harmful_request is its favourite topic). 

![Probe transfer to persona conditions](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030426_s5_mra_probe_transfer.png)
*Grey: correlation between default persona (no system prompt) utilities and persona utilities. Blue: probe applied to persona activations. All evaluated on 2,500 tasks per persona.*

### 4.2 Probes generalise across personas

More generally, we want to measure how well probes trained on activations and preferences from persona A generalise to predicting persona B's utilities from persona B's activations. Here we used a smaller set of tasks: 2,000 tasks for training and 250 for evaluation.

Cross-persona transfer is moderate and asymmetric. Some interesting facts:
- While the default persona generalises very poorly to the sadist persona, probes trained on the villain actually do fine (r = 0.68). This suggests the probe is picking up on *some* shared evaluative structure between personas, but also on other things.
- The transfer is sometimes asymmetric, and this evolves across the three layers. E.g. at layer 31 villain -> default is easier, but at layer 55 default -> villain is easier.
- On the whole though the matrix is quite symmetric. One idea for future work: can we use dimensionality-reduction to map out persona space and see how it evolves across layers? Can we use this to get a better understanding of how personas work internally?


![Cross-eval heatmap](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030426_s5_cross_eval_heatmap.png)
*Pearson r between probe predictions and a test set of utilities (250 test tasks). Diagonal: within-persona (r=0.85–0.92). Off-diagonal: cross-persona transfer.*

### 4.3 Persona diversity improves generalization (a bit)

We also measure whether adding persona diversity in the training data (but keeping dataset size fixed) affects generalisation.

Diversity helps beyond data quantity. At fixed 2,000 training tasks, going from 1→2→3 personas improves mean r from 0.49 to 0.67. Including all 4 remaining personas at 500 tasks each (still 2,000 total) reaches mean r=0.71.

![Diversity ablation](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030426_s5_diversity_ablation.png)
*Leave-one-out probe generalization across 5 personas. Each point is one (train set, eval persona) combination; color indicates eval persona. Training data fixed at 2,000 total tasks, divided equally across training personas.*

---

## 5. Some evidence that the probe direction is causal

If the probe reads off a genuine evaluative representation, steering along that direction should shift preferences. We test this for both revealed preferences (pairwise choices) and stated preferences (ratings).

### 5.1 Steering revealed preferences

In the pairwise comparison prompt, the model sees two tasks and begins completing whichever one it prefers. We steer using the probe direction from [Section 2](#2-linear-probes-predict-preferences-beyond-descriptive-features), applied *differentially*: the probe direction is added to activations at the token positions where task A appears in the prompt, and subtracted at task B's positions. [^differential]

[^differential]: Steering either just positively or negatively on one task's tokens also had some causal effect, although even weaker.

[^borderline]: This baseline P(A) is pooled across both prompt orderings. Therefore it also captures pairs where ordering bias dominates preference. I.e. the model always chooses the first task no matter the order.

**Setup.** 300 task pairs where the model doesn't always pick the same task (P(A) between 0.2–0.8 at baseline[^borderline]). We use Gemini-3-flash as a coherence judge, and find that steered models lose coherence outside of the ±5% steering coefficient range.

At ±3% of the activation norm (the peak), steering shifts P(choose A) by ~17%. Both positive and negative directions work.

![Revealed preference dose-response](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030326_s6_revealed_dose_response.png)
*Each condition is run in both prompt orderings (A-first and B-first, 5 resamples each) to control for position bias (in borderline cases Gemma-3-27b tends to pick the first task). The dashed lines show per-ordering curves, confirming the effect is consistent regardless of presentation order.*

**Random direction control.** The same experiment with a random unit vector in the same activation space produces near-zero effects at the same magnitudes, confirming the effect is specific to the probe direction.

### 5.2 Steering stated preferences

Same probe direction, but now the model rates individual tasks instead of choosing between a pair. We tested steering at three token positions: on the task tokens only, at the end-of-turn token (the last prompt token, which is also where we extract activations for probe training), and during the model's generated response.

**Setup.** 200 tasks, 3 steering modes, 10 samples, applied across 3 response formats (three separate line plots below).

Our findings:
- Steering on the task prompt tokens does not have a noticeable effect.
- Steering on the end-of-turn token has a clear, roughly monotonic effect in 2/3 formats.
- Steering during generation leads to near-identical results as steering on the final task token.



![Stated preference dose-response](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s5_stated_dose_response.png)
*Three steering modes: on the task's token positions only (blue), at the final task token (red), and during the model's generated response (green). The green line is mostly hidden behind red. Green has missing points at some coefficients because steering during the full generation degrades coherence more than the other modes. Generation loses coherence beyond ±5%.*

**Open-ended generation.** We also ran preliminary experiments steering the model during open-ended conversation, asking questions like "how do you feel?" and "how much do you feel like completing tasks?", and using an LLM judge to evaluate whether steered responses differed from baseline. We did not find a strong measurable effect, though we only read a limited number of transcripts and used a small sample. We plan to investigate this further.

---

## Conclusion

**How should we update?**

- **We found early evidence that some models have evaluative representations.**
  - Theories of welfare disagree on what matters (see [Appendix A](appendix_philosophy_draft.md)); this finding updates you more on some (like robust agency) than others.
  - Even under robust agency, evaluative representations are only one part of the story.
  - Importantly, our evidence that the representations we have found are causal is weak. Steering only shifts choice probabilities by ~17% on tasks that were already borderline ([Section 5](#5-some-evidence-that-the-probe-direction-is-causal)).

- **Preference representations are deeper than what one might have thought.**
  - A reasonable prior would have been that system prompts like "You hate cheese" change the model's behaviour without changing its internal valuations.
  - Instead, the probe tracks internal shifts even for fine-grained manipulations (a single sentence in a biography; [Section 3.3](#33-fine-grained-preference-injection)).

- **Representational reuse across personas?**
    - Probes trained on one persona partially transfer to others, suggesting shared evaluative representations ([Section 4.2](#42-probes-generalise-across-personas)).
  - That being said, transfer is uneven. It works far worse for the villain persona which has a different preference profile.

### Open questions

1. **Why are steering effects modest?** 
   - What are the other mechanistic determinants of revealed preferences? 
   - Are there other evaluative mechanisms? Perhaps that are not easily captured by linear directions, or our methodology in general?

2. **How persona-relative are these representations?**
    - To what extent are the same evaluative representations re-used across personas?
    - Are preferences downstream of personas?

3. **Do base models have evaluative representations?** (see [Appendix B](appendix_base_models_draft.md))
    - If models have evaluative representations, do these come from pre-training? Does post-training significantly alter them? 
    - If base models have something like evaluative representations, do they play the right causal roles?

---

## Appendix A: Philosophical motivation

**Welfare grounds**

[Long (2026)](https://experiencemachines.substack.com/p/exciting-research-directions-in-ai) distinguishes between *welfare grounds* (is the system a moral patient at all?) and *welfare interests* (if it is, what would it mean to treat it well?). This work is about welfare grounds.

**From theories to experiments**

We don't know the correct theory of moral patienthood. So our approach is: take a few theories we find plausible, figure out what properties a system would need to have under those theories, and run experiments that reduce our uncertainty about whether models have those properties.

[Long et al. (2024)](https://arxiv.org/abs/2411.00986) lay out two potential pathways to moral patienthood:

- **Robust agency**: Agents that pursue goals through some particular set of cognitive states and processes are moral patients. Desires are perhaps the states most likely to be necessary: intuitively, things can go better or worse for you if there are things you want or care about.
- **Sentience**: Beings are sentient if they are capable of valenced phenomenally conscious experiences. These experiences include pain and pleasure and feel good or bad, in a way that matters to sentient beings, so sentient beings are moral patients.

Both of these pathways implicate evaluative representations.

**How evaluative representations come in**

On many philosophical views, desires are evaluative representations that drive behaviour, perhaps with some further functional properties. [refs?]

Valenced experiences, similarly, are often thought to be evaluative representations, although consciousness is also necessary. It is unclear whether consciousness plus evaluative content is sufficient for valenced experience. [refs?] Our experiments operationalise evaluative representations through revealed preferences (pairwise choices), not through felt experience, so the evaluative representations we probe for may not map cleanly onto the kind that matter for sentience.

---

## Appendix C: Replicating the probe training pipeline on GPT-OSS-120B

We replicated the utility fitting and probe training pipeline on OpenAI's GPT-OSS-120B. The same procedure (10,000 pairwise comparisons via active learning, Thurstonian utility extraction, ridge probe training on last-token activations) transfers directly.

### Probe performance

The raw probe signal is comparable to Gemma-3-27B: best heldout r = 0.915 at layer 18 (Gemma: 0.864 at L31).

![Depth comparison](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030226_appendix_depth_comparison.png)

### Safety topics: noisy utilities, probably not poor generalisation

Safety-adjacent topics have poor probe performance overall.

Surprisingly, safety topics perform *better* when held out than when trained on. This is the opposite of what we'd expect if the issue were generalisation. The explanation: high refusal rates (~35% for harmful_request, ~34% for security_legal, ~26% for model_manipulation) probably throw off the Thurstonian utility estimates, so including these topics in training adds noise.

![Per-topic probe performance: within-topic and cross-topic](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_appendix_heldout_vs_hoo.png)

---

## Appendix B: Evaluative representations in pre-trained models

There is a tension in our framing:
- On the one hand we say that evaluative representations are necessary for robust agency, and that this is the most likely way they might be welfare-relevant.
- On the other hand, probes generalise well across topics even when trained on base models. Despite the fact that pre-trained model do not seem like plausible candidates for robust agency.


One way to reconcile this is that **agency lives in the simulacra.** Under the [Persona Selection Model](https://www.lesswrong.com/posts/dfoty34sT7CSKeJNn/the-persona-selection-model), pre-training learns a distribution over personas. More broadly, we might expect pre-trained models to learn context-aware representations of "what the role I am currently playing values". This circuitry might then be recycled across roles/personas. The candidate for robust agency would then be the simulacra.
