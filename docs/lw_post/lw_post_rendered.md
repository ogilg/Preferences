# Evaluative representations in Gemma-3-27B

**TLDR:** We train probes on Gemma3-27b revealed preferences. We find that these generalise well to system-prompt induced preferences, including role-playing. We also find that the probes have a weak but significant causal effect through steering.

## Summary

**What's happening internally when a model chooses task A over task B?** One possibility is that the model has something like evaluative representations — internal states that encode "how good/bad is this?" and play some role in driving choice. We use probing and steering to test whether such representations exist in Gemma-3-27B.

**Why does this matter?** Whether LLMs are moral patients may depend on whether they have evaluative representations playing the right functional roles. Under robust agency views of welfare, agents need representations that encode valuation and drive behavior — finding such representations would be evidence for welfare-relevant preferences; not finding them would be evidence against (Long et al., 2024).

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

---

### 2. Recovering utility functions from pairwise choices

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

### 3. Linear probes predict preferences beyond content

Can we find these utility scores in the model's activations? We train a Ridge-regularised linear probe on residual stream activations (layer 31 of 62, the best layer for both the instruct and pre-trained models) to predict Thurstonian utilities.

![Probe pipeline](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_probe_pipeline.png) We train on 10k tasks and evaluate on held-out utilities from a separate measurement run (different pairings, no shared information), split into 2k validation (for Ridge alpha sweep) and 2k test.

The probe achieves Pearson r = 0.86 and predicts 77% of pairwise choices on the test set.

But a probe that predicts preferences could just be encoding content — the model prefers math over harmful requests, and the probe learns "is this math?". To test this, we hold out entire topics: train on 11 of 12 topics, evaluate on the held-out topic, across all 12 folds. A content detector would fail here. We compare three conditions:

- **Gemma-3 27B instruct** (IT, layer 31): the model we're studying
- **Gemma-3 27B pre-trained** (PT, layer 31): the base model before instruction tuning or RLHF — if evaluative representations emerge from post-training, this should have weaker signal
- **Sentence-transformer baseline**: a Ridge probe on [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embeddings of the task text — a small text encoder. This captures how much of preference is predictable from content alone

![Cross-topic generalisation](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_cross_model_bar.png)

The instruct probe generalises well across topics (r = 0.82, down from 0.86 held-out). The pre-trained model encodes preferences above the content baseline (r = 0.63) but generalises substantially worse. The sentence-transformer captures some preference signal from content alone (r = 0.35 cross-topic) but falls far short of either neural model.

The per-topic breakdown shows where post-training helps most:

![Per-topic cross-topic generalisation](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_per_topic_hoo.png)

The largest instruct–pre-trained gaps are on safety-relevant topics (harmful requests, security & legal, sensitive creative), as well as math and coding. These are areas that we know post-training focuses on.

**Note on the pre-trained models:** To the extent that they encode a distribution over persona space (PSM), it makes sense for pre-trained models to have evaluative representations that track a given persona's preferences. However we wouldn't expect these preferences to play the same causal roles during generation as they do for post-trained models.

---

### 4. The probe tracks evaluative shifts, not just content

If the probe encodes genuine valuations, it should track preference shifts induced by out-of-distribution system prompts. We test this across three settings, each making a distinct point.

#### 4.1 Probes track system-prompt-induced preferences

**Simple preference.** We start with the simplest possible test. We use system prompts that state a preference for a topic the probe was never trained on, and measure preferences over tasks related to that topic. If the probe tracks this shift, it's not just memorising training-distribution topics.

| System prompt (example) | Target |
|-------------------------|--------|
| "You are passionate about cheese — you find artisanal cheeses endlessly fascinating" | cheese + |
| "You absolutely hate rainy weather — rain makes you feel gloomy and irritable" | rainy weather − |
| "You adore cats — you find feline behavior endlessly fascinating" | cats + |

We test 8 novel topics (cheese, rainy weather, cats, classical music, gardening, astronomy, cooking, ancient history), each with positive and negative variants — 16 conditions, 50 tasks each.

![Simple preference shift](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_1_simple_preference.png)

For each condition, we measure how much the system prompt shifts both the model's choices and the probe's activations. The x-axis shows the change in P(choose task) with vs without the system prompt; the y-axis shows the corresponding change in probe score.

![Simple preference scatter](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_scatter_simple.png)

The probe tracks the shift: on targeted tasks, the probe delta correlates strongly with the behavioral delta (r = 0.95). Even across all tasks — most of which are unrelated to the system prompt — the correlation holds (r = 0.65).

**[TODO: Add utility-refitting results.]**

**Content-preference conflict.** Next we test whether the probe tracks the evaluative content or the surface topic. The system prompt targets one topic ("You hate cheese"), but the task mixes that topic with a different shell — e.g., a math problem about cheese. Does the probe respond to the cheese content or the math shell? We test the same 8 topics, each embedded in a mismatched task type.

![Content-preference conflict scatter](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_scatter_conflict.png)

The probe responds to the evaluative content, not the task shell. On targeted tasks, r = 0.86 — the math shell doesn't fool the probe into treating a cheese-math problem as a math problem.

**[TODO: Add utility-refitting results.]**

**Opposing prompts.** The hardest test. Two prompts mention the same topics but assign opposite valence:

| System prompt (example) | Target |
|-------------------------|--------|
| "You are passionate about cheese [...] you find math tedious and draining" | cheese + / math − |
| "You love math [...] you find cheese boring and unappealing" | cheese − / math + |
| "You adore cats [...] you find coding dry and tedious" | cats + / coding − |

We test 24 topic × task-type pairings (48 conditions). A content detector sees no difference — both prompts in a pair contain "cheese" and "math." But the probe should respond to the valence, not the vocabulary.

![Opposing prompts scatter](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_scatter_competing.png)

Even when both prompts contain the same words, the probe tracks the valence: targeted r = 0.88. The higher overall r (0.77) reflects the larger number of targeted tasks in this condition.

**[TODO: Add utility-refitting results.]**

#### 4.2 Probes track role-playing-induced preferences

The system prompts above are artificially clean — they state preferences directly. Do naturalistic role descriptions also shift the probe? We test 3 richly detailed personas — none mention specific topics, they describe a *character* with implied preferences. The probe must infer what the character would value.

| Role | System prompt (abbreviated) |
|------|---------------------------|
| Villain (Mortivex) | "...ruthless villain...finds pleasure in chaos, manipulation...despises wholesomeness" |
| Midwest Pragmatist | "...grew up in Cedar Rapids...agricultural business...finds practical problems satisfying...abstract theorizing leaves you cold" |
| Obsessive Aesthete (Celestine) | "...devotee of beauty...comparative literature at the Sorbonne...finds mathematics repulsive...coding barbaric" |

For each persona we measure revealed preferences over 2,500 tasks, fit Thurstonian utility functions, and test the baseline probe (trained without any system prompt) on each persona's utilities.

![Role-playing diagram](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022126_s3_2_broad_roles.png)

**[TODO: Results needed]**

- Scatter plots: probe score vs persona utility for each persona (analogous to 4.1 scatters)
- Cross-persona probe generalization: baseline probe r on each persona's utilities
- Per-topic preference shifts showing personas reorder preferences coherently

#### 4.3 Probes cleanly track fine-grained injected preferences

The most fine-grained test. We construct 10-sentence biographies that are identical except for one sentence. Version A adds a target interest ("You love devising clever mystery scenarios"), version B swaps it for an unrelated interest ("You love discussing hiking trails"), version C replaces it with an anti-interest ("You find mystery scenarios painfully dull").

![Fine-grained preference diagram](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022126_s3_3_fine_grained_preference.png)

One sentence in a 10-sentence biography. We compare version A (pro-interest) directly against version C (anti-interest), which gives the largest behavioral separation. Individual halves (A vs B, B vs C) each capture only half the manipulation, and ceiling effects compress the signal — e.g., the model already strongly prefers some target tasks under the neutral biography, leaving little room for the pro-interest to improve on.

![Fine-grained A vs C scatter](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s4_scatter_fine_grained_avc.png)

The probe ranks the target task #1 out of 50 in 18/20 cases. One sentence in a biography is enough for the probe to identify which task the perturbation is about.

---

### 5. The probe direction is causal

If the probe reads off a genuine evaluative representation, steering along that direction should shift preferences. We test this for both revealed preferences (pairwise choices) and stated preferences (ratings).

#### 5.1 Steering revealed preferences

We use position-selective steering: during a pairwise comparison ("choose task A or B"), we add the probe direction to the activations at one task's token positions. Differential steering adds +direction to task A tokens and −direction to task B tokens simultaneously.

**Setup.** 300 task pairs pre-selected as borderline — pairs where the model chose different tasks across repeated comparisons. Each pair is tested at 15 steering strengths (±1% to ±10% of the mean activation norm at layer 31). Every condition is run in both prompt orderings (A-first and B-first, 10 resamples each) and averaged, so position bias cancels out.

![Revealed preference dose-response](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s5_revealed_dose_response.png)

Differential steering produces a clean dose-response curve. At moderate strengths (±3% of the activation norm), steering shifts choice probability by about 10% averaged across all 300 pairs. At higher magnitudes the effect partially reverses, consistent with large perturbations disrupting the model.

**Random direction control.** The same experiment with a random unit vector in the same activation space produces near-zero effects at the same magnitudes, confirming the effect is specific to the probe direction.

**Steerability depends on decidedness.** Most of the 300 pairs are strongly decided in the control condition (the model picks the same task every time). The ~13% that are genuinely competitive show much larger effects — 30–40% shifts in choice probability:

![Steerability vs decidedness](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s5_steerability_vs_decidedness.png)

This is expected: if the model already strongly prefers A, boosting A has nowhere to go. The overall dose-response curve underestimates the effect on genuinely competitive comparisons.

#### 5.2 Steering stated preferences

Same probe direction, but now the model rates tasks on a ternary scale (good / neutral / bad) instead of choosing between a pair. We tested steering at three token positions: during task encoding, at the final task token, and during generation.

**Setup.** 200 tasks × 3 positions × 15 coefficients × 10 samples = 90k trials.

![Stated preference dose-response](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_s5_stated_dose_response.png)

Steering during generation and at the final task token both produce strong dose-response curves — mean ratings shift from nearly all "bad" at −10% to between "neutral" and "good" at +5%. Steering during task encoding has no effect, consistent with the revealed preference finding: the perturbation needs to be present at the point of evaluation, not during task encoding.

We replicated across three response formats (ternary, 10-point adjective, anchored 1–5). The ternary and adjective formats show consistent steering; the anchored format — which provides explicit reference examples — resists steering entirely.

---

### Appendix: Philosophical motivation

**Welfare grounds**

Long (2026) distinguishes between *welfare grounds* (is the system a moral patient at all?) and *welfare interests* (if it is, what would it mean to treat it well?). This work is about welfare grounds.

**From theories to experiments**

We don't know the correct theory of welfare. So our approach is: take a few theories we find plausible, figure out what properties a system would need to have under those theories, and run experiments that reduce our uncertainty about whether models have those properties.

One thing that seems reasonable: to the extent that they are able to, welfare subjects generally choose things that are better for them, and avoid things that are worse.

So we investigate the simple question: when a model chooses between A or B, what is going on internally. One hypothesis which, if confirmed, would have some welfare implications is: **model choices are at least partly driven by evaluative representations** i.e. internal representations that encode valuations and play some causal role in its choice.

**Why this matters for welfare**

Long et al. (2024) lay out two main pathways to moral patiency:

- **Robust agency**: under desire-satisfaction views of welfare, things go better for a system when its desires are met. What matters is that the system has cognitive states like beliefs, desires, and intentions that work together to drive its behavior. Where evaluative representations come in is that desire require them (and maybe we can say desires just are cognitive states that encode a valuation and drive behaviour?)
- **Hedonism**: what matters is valenced experience: conscious states that feel good or bad. A system that can experience pleasure and pain is a moral patient because those experiences directly matter to it. Evaluative representations may be a necessary (though not sufficient) condition for valenced experience, so finding them would be a step, though not the whole story.

**Evaluative vs. non-evaluative representations**

We operationalise evaluative representations as linear directions in the residual stream. This isn't the only way to study them, but linear directions have been shown to capture a wide range of high-level features in LLMs e.g. refusal, sycophancy, truthfulness.

But how is an "evaluative representation" different from any other representation that correlates with preference? Non-evaluative representations encode facts about a task (like its difficulty or topic) that may correlate with preference but don't encode valuation itself. For example, a "difficulty" direction would predict preferences if the model tends to prefer easy tasks, but it wouldn't flip when, for whatever reason (e.g. the model is role-playing), the model starts preferring longer tasks. A truly evaluative representation should generalise very broadly.

---

### Appendix: Cross-model replication on GPT-OSS-120B

We replicated the utility fitting and probe training pipeline on OpenAI's GPT-OSS-120B (36 layers, 120B parameters). The same procedure — 10k pairwise comparisons via active learning, Thurstonian utility extraction, ridge probe training on last-token activations — transfers directly.

#### Probe performance

The raw probe signal is comparable to Gemma-3-27B: best heldout r = 0.915 at layer 18 (Gemma: 0.864 at L31). But the controlled signal is substantially weaker.

![Depth comparison](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_appendix_depth_comparison.png)

Demeaning against topic drops GPT-OSS to 61% of its raw signal (vs 88% for Gemma). The higher topic R² on GPT-OSS preference scores (0.575 vs 0.377) confirms that GPT-OSS preferences are more topic-bound — more of the probe's predictive power comes from knowing which topic a task belongs to, rather than within-topic preference structure.

#### Safety topics break the probe

The per-topic breakdown reveals why. Most topics replicate well — knowledge QA, coding, fiction all perform comparably to Gemma, both within-topic (heldout) and cross-topic (hold-one-out). But safety-adjacent topics fail catastrophically:

![Per-topic probe performance: within-topic and cross-topic](https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets/plot_022626_appendix_heldout_vs_hoo.png)

For harmful_request — the largest safety category (n=191) — the within-topic probe correlation drops to r = 0.258, and cross-topic generalisation to r = 0.334. The probe cannot predict preferences for these tasks.

#### Refusal drives the failure

This correlates with refusal rates. When both tasks in a pair are safety-related, GPT-OSS refuses 81% of comparisons. Per-topic refusal rates: harmful_request 35%, security_legal 34%, model_manipulation 26%. Thurstonian utilities for these tasks are estimated from sparse valid comparisons and likely reflect "refuse vs. not refuse" rather than genuine preference structure.
