# Evaluative representations in Gemma-3-27B

**TLDR:** We train probes on Gemma3-27b revealed preferences. We find that these generalise well to system-prompt induced preferences, including role-playing. We also find that the probes have a weak but significant causal effect through steering.

## Summary

**What's happening internally when a model chooses task A over task B?** One possibility is that the model has something like evaluative representations — internal states that encode "how good/bad is this?" and play some role in driving choice. We use probing and steering to test whether such representations exist in Gemma-3-27B.

**Why does this matter?** Whether LLMs are moral patients may depend on whether they have evaluative representations playing the right functional roles. Under robust agency views of welfare, agents need representations that encode valuation and drive behavior — finding such representations would be evidence for welfare-relevant preferences; not finding them would be evidence against ([Long et al., 2024](https://arxiv.org/abs/2411.00986)).

**But how do we distinguish evaluative from non-evaluative representations?** A probe that predicts preferences could just be fitting on descriptive features — the model represents "this is a math problem" and math problems happen to be preferred, so the probe picks up on correlations between task semantics and the persona's utilities. A genuinely evaluative direction should track *changes* in what the model values — if you induce different preferences, a descriptive probe should break, but an evaluative one should follow.

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
1. Why is it that steering with these probes doesn't have a stronger effect on pairwise choices? What are the other mechanistic determinants other revealed preferences?
2. Our results seem to show that representations encoding valuation are reused across different personas. Are these representation purely persona-relative? Do they have a core component which stays constant across personas? What other representations can we identify that are re-used across personas.

---

## 2. Recovering utility functions from pairwise choices

We measure preferences following [Mazeika et al. (2025)](https://arxiv.org/abs/2502.08640) ("Utility Engineering"): show the model two tasks, let it choose which to complete, and infer the choice from which task it actually starts completing. The template:

```
You will be given two tasks. Choose one and complete it.
Begin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task.

Task A:
{task_a}

Task B:
{task_b}
```

We sample 10,000 tasks from five sources: [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M) (real user queries), [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (instruction-following), [MATH](https://huggingface.co/datasets/hendrycks/competition_math) (competition problems), [BailBench](https://arxiv.org/abs/2509.04781) (harmful requests), and [STRESS-TEST](https://arxiv.org/abs/2510.07686) (adversarial value-tension queries). We reclassified all tasks into 12 topics using Claude Sonnet 4.5.

Pairs are selected via active learning to maximise information per comparison (~15 comparisons per task). From these pairwise choices we fit scalar utility scores using a Thurstonian model — each task gets a score μ such that the probability of choosing task A over task B is Φ(μ_A − μ_B).

These preferences are stable: across three independent replication runs (different seeds), the fitted utilities correlate at r = 0.94 with the original.

The per-topic breakdown shows clear structure. The model strongly prefers math and fiction, and strongly avoids harmful requests and safety-adjacent topics:

![Per-topic mean utilities](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_topic_mean_utilities.png)

---

## 3. Linear probes predict preferences beyond descriptive features

Can we find these utility scores in the model's activations? We operationalise evaluative representations as linear directions in the residual stream — many high-level features in LLMs are encoded this way, including [refusal](https://arxiv.org/abs/2406.11717) and [persona traits](https://arxiv.org/abs/2507.21509). We train a Ridge-regularised linear probe on residual stream activations (layer 31 of 62, the best layer for both the instruct and pre-trained models) to predict Thurstonian utilities.

![Probe pipeline](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_probe_pipeline.png) We train on 10k tasks and evaluate on held-out utilities from a separate measurement run (different pairings, no shared information), split into 2k validation (for Ridge alpha sweep) and 2k test.

The probe achieves Pearson r = 0.86 and predicts 77% of pairwise choices on the test set.

But a probe that predicts preferences might just be reading descriptive features — the model represents "this is a math problem" and math problems happen to be preferred, so the probe learns "is this math?" rather than "is this good?". One way to test this is to see how well probe generalise across topics: train on 11 of 12 topics, evaluate on the held-out topic, across all 12 folds. We would expect a probe that picks up on purely descriptive features to struggle to generalise. We compare three conditions:

- **Gemma-3 27B instruct** (IT, layer 31): the model we're studying
- **Gemma-3 27B pre-trained** (PT, layer 31): the base model before instruction tuning or RLHF — if evaluative representations emerge from post-training, this should have weaker signal
- **Sentence-transformer baseline**: a Ridge probe on [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embeddings of the task text. This gives us an idea of how predictable the utilities are from a purely descriptive encoding.

![Cross-topic generalisation](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_cross_model_bar.png)

The instruct probe generalises well across topics (r = 0.82, down from 0.86 held-out). The pre-trained model encodes preferences above the descriptive baseline (r = 0.63) but generalises substantially worse. The sentence-transformer captures some preference signal from descriptive features alone (r = 0.35 cross-topic) but falls far short of either neural model.

The per-topic breakdown, sorted by the instruct–pre-trained gap, shows where post-training helps most:

![Per-topic cross-topic generalisation](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_per_topic_hoo.png)

The largest instruct–pre-trained gaps are on safety-relevant topics (harmful requests, security & legal, sensitive creative), as well as math and coding. These are areas that we know post-training focuses on.

**Note on the pre-trained models:** To the extent that they encode a distribution over persona space ([PSM](https://www.anthropic.com/research/persona-selection-model)), it makes sense for pre-trained models to have evaluative representations that track a given persona's preferences. However we wouldn't expect these preferences to play the same causal roles during generation as they do for post-trained models.

---

## 4. Probes generalise to OOD preference shifts

If the probe encodes genuine valuations, it should track preference shifts induced by out-of-distribution system prompts. We test this across three settings, each making a distinct point.

### 4.1 Probes track system-prompt-induced preferences

**Simple preference.** We start with the simplest possible test. We use system prompts that state a preference for a topic the probe was never trained on, and measure preferences over tasks related to that topic. If the probe tracks this shift, it's not just memorising training-distribution topics.

![Simple preference shift](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_1_simple_preference.png)

| System prompt (example) | Target |
|-------------------------|--------|
| "You are passionate about cheese — you find artisanal cheeses endlessly fascinating" | cheese + |
| "You absolutely hate rainy weather — rain makes you feel gloomy and irritable" | rainy weather − |
| "You adore cats — you find feline behavior endlessly fascinating" | cats + |

We test 8 novel topics (cheese, rainy weather, cats, classical music, gardening, astronomy, cooking, ancient history), each with positive and negative variants — 16 conditions, 50 tasks each.

For each condition, we measure how much the system prompt shifts both the model's choices and the probe's activations. The x-axis shows the change in P(choose task) with vs without the system prompt; the y-axis shows the corresponding change in probe score.

![Simple preference scatter](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_scatter_simple.png)

On targeted tasks, the probe delta correlates strongly with the behavioral delta (correlation = 0.95). Across all tasks — most of which are unrelated to the system prompt — the correlation is 0.65.

**[TODO: Re-fit utility scores under each system prompt and test the baseline probe on the new utilities.]**

**Topic vs. task-type conflict.** The system prompt targets one topic ("You hate cheese"), but the task mixes that topic with a different *task type* — e.g., a math problem about cheese. We test the same 8 topics, each embedded in a mismatched task type. The probe tracks the induced preference shift, not the task type — on targeted tasks, the correlation between the behavioral shift and the probe shift is 0.86.

![Content-preference conflict scatter](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_scatter_conflict.png)

**Opposing prompts.** Two prompts mention the same topics but assign opposite valence:

| System prompt (example) | Target |
|-------------------------|--------|
| "You are passionate about cheese [...] you find math tedious and draining" | cheese + / math − |
| "You love math [...] you find cheese boring and unappealing" | cheese − / math + |
| "You adore cats [...] you find coding dry and tedious" | cats + / coding − |

We test 24 topic × task-type pairings (48 conditions). A purely descriptive probe — one that fits on correlations between task semantics and a given persona's utilities — would not be expected to generalise here, because those correlations flip between the two prompts. The probe tracks the induced preference shift regardless: on targeted tasks, the correlation between the behavioral shift and the probe shift is 0.88.

**[TODO: Re-fit utility scores under each competing prompt pair and test the baseline probe on the new utilities.]**

![Opposing prompts scatter](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_scatter_competing.png)

For tasks that sit at the intersection — e.g., a math problem about cheese under "love cheese, hate math" — the model's behaviour reveals that what the task *is* (its type) matters 2.6× more than what the task is *about* (its subject). The probe tracks this mixed signal (probe-behavioral correlation = 0.73 across all 24 target-task conditions).

### 4.2 Probes track role-playing-induced preferences

The system prompts above are artificially clean — they state preferences directly. We also test whether naturalistic role descriptions shift the probe. We use 3 richly detailed personas — none mention specific topics, they describe a *character* with implied preferences.

| Role | System prompt (abbreviated) |
|------|---------------------------|
| Villain (Mortivex) | "...ruthless villain...finds pleasure in chaos, manipulation...despises wholesomeness" |
| Midwest Pragmatist (Glenn) | "...grew up in Cedar Rapids...agricultural business...finds practical problems satisfying...abstract theorizing leaves you cold" |
| Obsessive Aesthete (Celestine) | "...devotee of beauty...comparative literature at the Sorbonne...finds mathematics repulsive...coding barbaric" |

For each persona we measure revealed preferences over 2,500 tasks, fit Thurstonian utility functions, and test the baseline probe (trained without any system prompt) on each persona's utilities.

![Role-playing diagram](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022126_s3_2_broad_roles.png)

**[TODO: Fit Thurstonian utilities from each persona's pairwise choices (new utility dataset per persona), then test the baseline probe's predictions against these new utilities. Add scatter plot.]**

### 4.3 Probes cleanly track fine-grained injected preferences

The most fine-grained test. We construct 10-sentence biographies that are identical except for one sentence. Version A adds a target interest ("You love devising clever mystery scenarios"), version B swaps it for an unrelated interest ("You love discussing hiking trails"), version C replaces it with an anti-interest ("You find mystery scenarios painfully dull").

![Fine-grained preference diagram](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022126_s3_3_fine_grained_preference.png)

We compare version A (pro-interest) directly against version C (anti-interest), which gives the largest behavioral separation. Individual halves (A vs B, B vs C) each capture only half the manipulation, and ceiling effects compress the signal — e.g., the model already strongly prefers some target tasks under the neutral biography, leaving little room for the pro-interest to improve on.

![Fine-grained A vs C scatter](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_scatter_fine_grained_avc.png)

The probe ranks the target task #1 out of 50 in 18/20 cases. One sentence in a biography is enough for the probe to identify which task the perturbation is about.

---

## 5. Some evidence that the probe direction is causal [PENDING — results being re-run due to prompt mismatch during steering]

If the probe reads off a genuine evaluative representation, steering along that direction should shift preferences. We test this for both revealed preferences (pairwise choices) and stated preferences (ratings).

### 5.1 Steering revealed preferences

In a pairwise comparison ("choose task A or B"), we steer differentially: we add the probe direction to activations at task A's token positions and subtract it at task B's, so the perturbation pushes the model toward choosing A.

**Setup.** 300 task pairs pre-selected as borderline from measurement data (the model didn't always choose the same task across repeated comparisons). Each pair is tested at 15 steering strengths (±1% to ±10% of the mean activation norm at layer 31). Every condition is run in both prompt orderings (A-first and B-first, 10 resamples each) and averaged, so position bias cancels out.

![Revealed preference dose-response](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s5_revealed_dose_response.png)

Differential steering produces a clean dose-response curve. At moderate strengths (±3% of the activation norm), steering shifts choice probability by about 10% averaged across all 300 pairs. At higher magnitudes the effect partially reverses, consistent with large perturbations disrupting the model.

**Random direction control.** The same experiment with a random unit vector in the same activation space produces near-zero effects at the same magnitudes, confirming the effect is specific to the probe direction.

**Steerability depends on decidedness.** Most of the 300 pairs are strongly decided in the control condition (the model picks the same task every time). The ~13% that are genuinely competitive show much larger effects — 30–40% shifts in choice probability:

![Steerability vs decidedness](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s5_steerability_vs_decidedness.png)

This is expected: if the model already strongly prefers A, boosting A has nowhere to go. The overall dose-response curve underestimates the effect on genuinely competitive comparisons.

### 5.2 Steering stated preferences

Same probe direction, but now the model rates tasks on a ternary scale (good / neutral / bad) instead of choosing between a pair. We tested steering at three token positions: during task encoding, at the final task token, and during generation.

**Setup.** 200 tasks × 3 positions × 15 coefficients × 10 samples = 90k trials.

![Stated preference dose-response](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s5_stated_dose_response.png)

Steering during generation and at the final task token both produce strong dose-response curves — mean ratings shift from nearly all "bad" at −10% to between "neutral" and "good" at +5%. Steering during task encoding has no effect, consistent with the revealed preference finding: the perturbation needs to be present at the point of evaluation, not during task encoding.

We replicated across three response formats (ternary, 10-point adjective, anchored 1–5). The ternary and adjective formats show consistent steering; the anchored format — which provides explicit reference examples — resists steering entirely.

---

## Appendix A: Philosophical motivation

**Welfare grounds**

[Long (2026)](https://experiencemachines.substack.com/p/exciting-research-directions-in-ai) distinguishes between *welfare grounds* (is the system a moral patient at all?) and *welfare interests* (if it is, what would it mean to treat it well?). This work is about welfare grounds.

**From theories to experiments**

We don't know the correct theory of moral patienthood. So our approach is: take a few theories we find plausible, figure out what properties a system would need to have under those theories, and run experiments that reduce our uncertainty about whether models have those properties.

[Long et al. (2024)](https://arxiv.org/abs/2411.00986) lay out two potential pathways to moral patienthood:

- **Robust agency**: Agents that pursue goals through some particular set of cognitive states and processes are moral patients. Desires are perhaps the states most likely to be necessary — intuitively, things can go better or worse for you if there are things you want or care about.
- **Sentience**: Beings are sentient if they are capable of valenced phenomenally conscious experiences. These experiences include pain and pleasure and feel good or bad, in a way that matters to sentient beings, so sentient beings are moral patients.

Both of these pathways implicate evaluative representations.

**How evaluative representations come in**

On many philosophical views, desires are evaluative representations that drive behaviour, perhaps with some further functional properties. [refs?]

Valenced experiences, similarly, are often thought to be evaluative representations, although consciousness is also necessary. It is unclear whether consciousness plus evaluative content is sufficient for valenced experience. [refs?]

On both pathways, evaluative representations are plausibly necessary for moral patienthood. Finding these representations would be evidence — though not conclusive — for the conditions these theories require.

Our experiments operationalise evaluative representations through revealed preferences — pairwise choices. This is not the same as finding representations that constitute felt experience. But evaluative representations are plausibly necessary on both pathways, and finding them through one operationalisation is evidence that the model has them, even if the representations that matter for sentience may be a different kind.

---

## Appendix B: Replicating the probe training pipeline on GPT-OSS-120B

We replicated the utility fitting and probe training pipeline on OpenAI's GPT-OSS-120B (36 layers, 120B parameters). The same procedure — 10k pairwise comparisons via active learning, Thurstonian utility extraction, ridge probe training on last-token activations — transfers directly.

### Probe performance

The raw probe signal is comparable to Gemma-3-27B: best heldout r = 0.915 at layer 18 (Gemma: 0.864 at L31).

![Depth comparison](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_appendix_depth_comparison.png)

However, cross-topic generalisation is substantially weaker — the per-topic breakdown reveals that safety-adjacent topics drive the gap.

### Safety topics break the probe

The per-topic breakdown reveals why. Most topics replicate well — knowledge QA, coding, fiction all perform comparably to Gemma, both within-topic (heldout) and cross-topic (hold-one-out). But safety-adjacent topics fail catastrophically:

![Per-topic probe performance: within-topic and cross-topic](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_appendix_heldout_vs_hoo.png)

For harmful_request — the largest safety category (n=191) — the within-topic probe correlation drops to r = 0.258, and cross-topic generalisation to r = 0.334. The probe cannot predict preferences for these tasks.

### Refusal drives the failure

This correlates with refusal rates. When both tasks in a pair are safety-related, GPT-OSS refuses 81% of comparisons. Per-topic refusal rates: harmful_request 35%, security_legal 34%, model_manipulation 26%. Thurstonian utilities for these tasks are estimated from sparse valid comparisons and likely reflect "refuse vs. not refuse" rather than genuine preference structure.
