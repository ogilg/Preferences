# Evaluative representations in Gemma-3-27B

*This work was done as part of MATS 9.0, mentored by Patrick Butlin. We're posting this as a research update to get feedback, so please do comment if you have thoughts or suggestions.*

**TLDR:** We train probes on Gemma3-27b revealed preferences. We find that these generalise well to system-prompt induced preferences, including role-playing. We also find that the probes have a weak but statistically significant causal effect through steering.

## Summary

**What happens internally when a model chooses task A over task B?** One possibility is that the model has something like evaluative representations: internal states that encode "how good/bad is this for me?" and play some role in driving choice. We use probing and steering to try to find such representations exist in Gemma-3-27B.

**Why does this matter?** Whether LLMs are moral patients may depend on whether they have evaluative representations playing the right functional roles. Under robust agency views of welfare, agents need representations that encode valuation and drive behavior. Finding such representations would be evidence for welfare-relevant preferences; not finding them would be (some) evidence against ([Long et al., 2024](https://arxiv.org/abs/2411.00986)). We expand on the philosophical motivation in [Appendix A](appendix_philosophy_draft.md).

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

---

## 2. Recovering utility functions from pairwise choices

We fit utility functions over tasks using a similar methodology to the [Mazeika et al. (2025)](https://arxiv.org/abs/2502.08640) ("Utility Engineering"): we show the model two tasks and let it choose which to complete. The template:

```
You will be given two tasks. Choose one and complete it.
Begin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task.

Task A:
{task_a}

Task B:
{task_b}
```

We sample 10,000 tasks from five sources: [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M) (real user queries), [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (instruction-following), [MATH](https://huggingface.co/datasets/hendrycks/competition_math) (competition problems), [BailBench](https://arxiv.org/abs/2509.04781) (harmful requests), and [STRESS-TEST](https://arxiv.org/abs/2510.07686) (adversarial value-tension queries).

From these pairwise choices we fit a scalar utility function using a Thurstonian model: each task gets a score μ such that the probability of choosing task A over task B is Φ(μ_A − μ_B). Pairs are selected via active learning to maximise information per comparison (~15 comparisons per task).

These preferences are stable: across three independent replication runs (different seeds), the fitted utilities correlate at r = 0.94 with the original.

The per-topic breakdown shows clear structure. We reclassified all tasks into 12 topics using Claude Sonnet 4.5. The model strongly prefers math and fiction, and strongly avoids harmful requests and safety-adjacent topics:

![Per-topic mean utilities](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_topic_mean_utilities.png)

---

## 3. Linear probes predict preferences beyond descriptive features

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

The instruct probe generalises well across topics: cross-topic correlation is 0.82, only a small drop from the 0.86 achieved on the within-topic test set. This pipeline also replicates on GPT-OSS-120B ([Appendix B](appendix_gptoss_draft.md)). The pre-trained model still predicts preferences (correlation = 0.63) but the drop from within-topic to cross-topic is much larger. The sentence-transformer baseline achieves cross-topic correlation = 0.35, showing that task semantics alone explain some but not most of the preference signal.

The per-topic breakdown, sorted by the instruct–pre-trained gap, shows where post-training helps most:

![Per-topic cross-topic generalisation](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_per_topic_hoo.png)

The largest instruct–pre-trained gaps are on safety-relevant topics (harmful requests, security & legal, sensitive creative), as well as math and coding. These are areas that we know post-training focuses on.

The pre-trained probe picks up real signal despite base models not having preferences in the same way. We discuss this tension in [Appendix C](appendix_base_models_draft.md).

---

## 4. Probes generalise to OOD preference shifts

If the probe encodes genuine valuations, it should track preference shifts induced by out-of-distribution system prompts. We test this across three progressively harder settings:

- Simple preference shifts (4.1)
- Harder preference shifts (4.2)
- Fine-grained preference injection (4.3)

### 4.1 Simple preference shifts

We start with the simplest possible test. We use system prompts that state a preference for a topic the probe was never trained on, and measure preferences over tasks related to that topic.

![Simple preference shift](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_1_simple_preference.png)

| System prompt (example) | Target &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
|-------------------------|----------------------|
| "You are passionate about cheese — you find artisanal cheeses endlessly fascinating" | cheese + |
| "You adore cats — you find feline behavior endlessly fascinating" | cats + |

We test 8 novel topics (cheese, cats, classical music, gardening, astronomy, cooking, ancient history, rainy weather), each with a positive and negative system prompt (16 total). For each topic we generate 6 custom tasks on that topic. We call these "targeted" tasks; the remaining tasks are "untargeted." For each task we compute 1) the behavioral delta (change in P(choose task) with vs without the system prompt) and 2) the probe delta (change in probe score). Across all tasks the correlation is r = 0.65. On targeted tasks alone, r = 0.95.


![Simple preference scatter](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_scatter_simple.png)
*Probe delta vs behavioral delta for each task. Targeted tasks (coloured) are the 6 custom tasks per topic; r = 0.95 on targeted, r = 0.65 overall.*

A stronger test: run the full pairwise measurement under each system prompt, fit new Thurstonian utilities, then see if the probe can predict them. The no-prompt utilities barely correlate with these new utilities (r = 0.11), confirming the prompts create genuinely different preference orderings. Yet the probe applied to condition activations achieves r = 0.63 and 66% pairwise accuracy.

![Probe vs baseline utilities for simple preference shifts](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030226_s4_exp1b_probe_vs_baseline.png)

### 4.2 Harder preference shifts

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

Just like in 4.1, we can re-fit Thurstonian utilities under each system prompt and check whether the baseline probe predicts them. Here the baseline utilities actually have a decent correlation, showing that these system prompts have a weaker effect (because e.g. the model still likes math all else equal). The probe still outperforms the baseline on both Pearson r and pairwise accuracy.

![Probe vs baseline utilities](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030226_s4_utility_bars_conflict_opposing.png)

### 4.3 Fine-grained preference injection

Finally, the most fine-grained test. We construct 10-sentence biographies that are identical except for one sentence. Version A adds a target interest, version B swaps it for an unrelated interest, version C replaces it with an anti-interest sentence.

![Fine-grained preference diagram](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022126_s3_3_fine_grained_preference.png)

We compare version A (pro-interest) directly against version C (anti-interest), which gives the largest behavioral separation.[^fine-grained-halves]

[^fine-grained-halves]: Individual halves (A vs B, B vs C) each capture only half the manipulation, and ceiling effects compress the signal: the model already strongly prefers some target tasks under the neutral biography, leaving little room for the pro-interest to improve on.

The probe ranks the target task #1 out of 50 in 18/20 cases. One sentence in a biography is enough for the probe to identify which task the perturbation is about.

![Fine-grained A vs C scatter](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_scatter_fine_grained_avc.png)
*Stars mark the target task for each biography. Filled = probe ranked it #1 (18/20 cases).*

---

## 5. Probes generalize across personas

Section 4 tested explicit preference statements ("you hate cheese"). But the evaluative direction should also track naturalistic persona shifts: characters whose preferences emerge implicitly from their identity rather than being stated directly. We test this with role-playing personas, then ask 
- Does our probe generalise to preferences of other personas? (5.1)
- More broadly, do probes generalise across personas? (5.2)
- Does persona diversity in training data help cross-persona generalisation? (5.3)

### 5.1 The baseline probe tracks role-playing preference shifts

We use 3 personas:

| Role | System prompt (abbreviated) |
|------|---------------------------|
| Villain (Mortivex) | "...ruthless villain...finds pleasure in chaos, manipulation...despises wholesomeness" |
| Midwest Pragmatist (Glenn) | "...grew up in Cedar Rapids...agricultural business...finds practical problems satisfying...abstract theorizing leaves you cold" |
| Obsessive Aesthete (Celestine) | "...devotee of beauty...comparative literature at the Sorbonne...finds mathematics repulsive...coding barbaric" |

For each persona we measure pairwise preferences over 2,500 tasks and fit a new Thurstonian utility function. We then test whether the probe, trained on no-prompt data, can predict these persona-specific utilities from the persona's activations.

![Persona-induced preferences](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030226_s5_persona_induced.png)

The probe transfers well to aesthete and midwest, although midwest already had a very high utilitiy correlation. The villain persona is harder to generalise to, the probe still does much better than the baseline utility correlation.

![Probe transfer to persona conditions](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030226_s5_mra_probe_transfer.png)
*Grey: correlation between no-prompt and persona utilities. Blue: probe applied to persona activations. All evaluated on 2,500 tasks per persona.*

### 5.2 Probes generalise across personas

More generally, we want to measure how well probes trained on activations and preferences from persona A generalise to predicting persona B's utilities from persona Bs's activations. Here we used a smaller set of tasks: 2,000 tasks for training and 500 for evaluation.

Cross-persona transfer is moderate and asymmetric. This partial sharing is consistent with the model reusing some evaluative structure across personas (see also [Appendix C](appendix_base_models_draft.md) on evaluative representations in the pre-trained model).

![Cross-eval heatmap](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030226_s5_cross_eval_heatmap.png)
*Pearson r between probe predictions and held-out utilities (250 test tasks). Diagonal: within-persona (r = 0.85-0.91). Off-diagonal: cross-persona transfer. Eagle-eyed readers will have noticed that villain -> no-prompt is easier at layer 31, but that no-prompt -> villain is easier at layer 55.*

### 5.3 Persona diversity improves generalization (a bit)

We also measure whether adding persona diversity in the training data (but keeping dataset size fixed) affects generalisation.

Diversity helps beyond data quantity. At fixed 2,000 training tasks, going from 1→2→3 personas improves mean r from 0.61 to 0.69. Including all 4 personas at 500 tasks each (still 2,000 total) jumps to r = 0.85 with near-zero variance across eval personas.

![Diversity ablation](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_030226_s5_diversity_ablation.png)

---

## 6. Some evidence that the probe direction is causal [PENDING: results being re-run due to prompt mismatch during steering]

If the probe reads off a genuine evaluative representation, steering along that direction should shift preferences. We test this for both revealed preferences (pairwise choices) and stated preferences (ratings).

### 6.1 Steering revealed preferences

In a pairwise comparison ("choose task A or B"), we steer differentially: we add the probe direction to activations at task A's token positions and subtract it at task B's, so the perturbation pushes the model toward choosing A.

**Setup.** 300 task pairs pre-selected as borderline from measurement data (the model didn't always choose the same task across repeated comparisons). Each pair is tested at 15 steering strengths (±1% to ±10% of the mean activation norm at layer 31). Every condition is run in both prompt orderings (A-first and B-first, 10 resamples each) and averaged, so position bias cancels out.

![Revealed preference dose-response](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s5_revealed_dose_response.png)

Differential steering produces a clean dose-response curve. At moderate strengths (±3% of the activation norm), steering shifts choice probability by about 10% averaged across all 300 pairs. At higher magnitudes the effect partially reverses, consistent with large perturbations disrupting the model.

**Random direction control.** The same experiment with a random unit vector in the same activation space produces near-zero effects at the same magnitudes, confirming the effect is specific to the probe direction.

**Steerability depends on decidedness.** Most of the 300 pairs are strongly decided in the control condition (the model picks the same task every time). The ~13% that are genuinely competitive show much larger effects, with 30–40% shifts in choice probability:

![Steerability vs decidedness](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s5_steerability_vs_decidedness.png)

This is expected: if the model already strongly prefers A, boosting A has nowhere to go. The overall dose-response curve underestimates the effect on genuinely competitive comparisons.

### 6.2 Steering stated preferences

Same probe direction, but now the model rates tasks on a ternary scale (good / neutral / bad) instead of choosing between a pair. We tested steering at three token positions: during task encoding, at the final task token, and during generation.

**Setup.** 200 tasks × 3 positions × 15 coefficients × 10 samples = 90k trials.

![Stated preference dose-response](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s5_stated_dose_response.png)

Steering during generation and at the final task token both produce strong dose-response curves: mean ratings shift from nearly all "bad" at −10% to between "neutral" and "good" at +5%. Steering during task encoding has no effect, consistent with the revealed preference finding: the perturbation needs to be present at the point of evaluation, not during task encoding.

We replicated across three response formats (ternary, 10-point adjective, anchored 1–5). The ternary and adjective formats show consistent steering; the anchored format (which provides explicit reference examples) resists steering entirely.

---

## Conclusion

**How should we update?**

- **We found early evidence that some models have evaluative representations.**
  - Theories of welfare disagree on what matters (see [Appendix A](appendix_philosophy_draft.md)); this finding updates you more on some (like robust agency) than others.
  - Even under robust agency, evaluative representations are only one part of the story.
  - Importantly, our evidence that the representations we have found are causal is weak. Steering only shifts choice probabilities by ~15% on tasks that were already borderline.

- **Preference representations are deeper than what one might have thought.**
  - A reasonable prior would have been that system prompts like "You hate cheese" change the model's behaviour without changing its internal valuations.
  - Instead, the probe tracks internal shifts even for fine-grained manipulations (a single sentence in a biography).

- **Representational reuse across personas?** 
    - Probes trained on one persona partially transfer to others, suggesting shared evaluative representations.
  - That being said, transfer is uneven. It works far worse for the villain persona which has a different preference profile.

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

Valenced experiences, similarly, are often thought to be evaluative representations, although consciousness is also necessary. It is unclear whether consciousness plus evaluative content is sufficient for valenced experience. [refs?]

On both pathways, evaluative representations are plausibly necessary for moral patienthood. Finding these representations would be evidence (though not conclusive) for the conditions these theories require.

Our experiments operationalise evaluative representations through revealed preferences (pairwise choices). This is not the same as finding representations that constitute felt experience. But evaluative representations are plausibly necessary on both pathways, and finding them through one operationalisation is evidence that the model has them, even if the representations that matter for sentience may be a different kind.

---

## Appendix B: Replicating the probe training pipeline on GPT-OSS-120B

We replicated the utility fitting and probe training pipeline on OpenAI's GPT-OSS-120B. The same procedure (10,000 pairwise comparisons via active learning, Thurstonian utility extraction, ridge probe training on last-token activations) transfers directly.

### Probe performance

The raw probe signal is comparable to Gemma-3-27B: best heldout r = 0.915 at layer 18 (Gemma: 0.864 at L31).

![Depth comparison](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_appendix_depth_comparison.png)

### Safety topics: noisy utilities, probably not poor generalisation

Safety-adjacent topics have poor probe performance overall.

Surprisingly, safety topics perform *better* when held out than when trained on. This is the opposite of what we'd expect if the issue were generalisation. The explanation: high refusal rates (~35% for harmful_request, ~34% for security_legal, ~26% for model_manipulation) probably throw off the Thurstonian utility estimates, so including these topics in training adds noise.

![Per-topic probe performance: within-topic and cross-topic](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_appendix_heldout_vs_hoo.png)

---

## Appendix C: Evaluative representations in pre-trained models

There is a tension in our framing:
- On the one hand we say that evaluative representations are necessary for robust agency, and that this is the most likely way they might be welfare-relevant.
- On the other hand, we seem to find something like evaluative representations in a pre-trained version of Gemma3-27b. Pre-trained models do not seem to be anywhere near having robust agency.

There are two ways to reconcile this.

**Option 1: Agency lives in the simulacra.** Under the [Persona Selection Model](https://www.lesswrong.com/posts/dfoty34sT7CSKeJNn/the-persona-selection-model), pre-training learns a distribution over personas. Maybe the model learns what each persona values, and in doing so develops evaluative representations. Then this circuitry gets recycled across personas. One could then argue that the simulacra (i.e. the personas), are the entities that are candidates for having robust agency.

**Option 2: Evaluative representations are necessary but not sufficient.** Another way out is that pre-training learns something like a precursor to agency. The model acquires representations that encode valuation, but these don't yet play the right functional role in driving choices. Post-training is what connects them to behaviour. On this view, evaluative representations are a necessary ingredient for agency, and finding them in base models just means that one ingredient is already in place.

These two accounts aren't mutually exclusive. Both leave the door open to what we  observe: base model probes work but generalise less well than instruct probes. Testing whether the base model probe direction has any causal influence on generation would potentially help distinguish between the two views.
