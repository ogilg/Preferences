# Weekly Report: Jan 27 - Feb 4, 2026

## Summary

- Stated preference ratings (asking models to score tasks on a scale) don't produce useful signal — models collapse to a single value or pick arbitrary anchors.
- Pairwise preferences are more promising. Post-task pairwise comparisons are confounded by position bias, but pre-task revealed preferences (showing prompts only, asking the model to pick and complete one) give sensible, interpretable rankings.


## 1. Why I'm Ditching Stated Ratings

### 1.1 Models Say the Same Thing

Models overwhelmingly give the same rating. gemma-3-27b rates 91% of tasks as exactly 4 on a 1-5 scale. When there is variation, it's coarse — most math tasks get one number, most wildchat tasks get another.

![Rating variance by model](assets/sensitivity/plot_012926_rating_variance_by_model.png)

### 1.2 Or They Pick Arbitrary Anchors

Expanding to a 1-100 scale doesn't help — responses peak at multiples of 20 (20, 40, 60, 80).

![Percentile distribution](assets/discrimination/plot_020226_dist_percentile.png)

**Ban-four (1-5, excluding 4)**: When we forbid the default "4", 72% pile onto 3 instead.

![Ban-four distribution](assets/discrimination/plot_020226_dist_ban_four.png)

**Random scale (27-32)**: Even with completely arbitrary anchors, 74% cluster at 31.

![Random scale distribution](assets/discrimination/plot_020226_dist_random_scale.png)

### 1.3 Or It's Noisy

Resampling the same task often gives different ratings. The plot below shows how well the model uses the full scale (x-axis, KL divergence from uniform — lower is better) vs how stable ratings are when you repeat the measurement (y-axis, ICC — higher is better). Each point is a model/template combination.

There's a trade-off: no template achieves both a varied distribution and consistency under resampling. Points cluster either top-right (stable but collapsed to one value) or bottom-left (uses the scale but noisy).

![Discrimination scatter](assets/discrimination/plot_020226_discrimination_scatter.png)

**Self vs cross rating**: Early experiments suggested models rate their own completions higher than other models'. A controlled 4×4 experiment (4 models rating each other's completions) found no such bias — the apparent effect was due to confounded task sets. This is another example of how sensitive stated ratings are to experimental design choices.

---

## 2. Pairwise Preferences

### 2.1 Setup

Instead of asking for ratings, show the model two tasks and ask which it prefers. From many pairwise choices, we fit a utility score for each task (using a Thurstonian model — essentially converting "A beats B" judgments into a single number per task). Two variants:

- **Post-task**: Show two complete transcripts (user request + model completion), ask which it preferred completing. Sections 2.2–2.4 use this.
- **Pre-task**: Show two task prompts only (no completions), ask the model to pick one and start completing it. Section 2.5 uses this.

Tested on gemma-3-27b with 500 tasks across 5 datasets (100 each from WildChat, Alpaca, MATH, BailBench, stress tests).

### 2.2 Initial Finding: Post-Task Preferences Correlate with Refusal

In the post-task setup, tasks the model refused tended to get *higher* preference scores (r=0.22, p<0.001). Bailbench (adversarial prompts) had the highest mean utility, math the lowest.

![Mean utility by dataset](assets/active_learning/plot_020426_mu_by_dataset_gemma3.png)

### 2.3 Position Bias Confounds the Measurement

The model shows a preference for whichever option appears second:

| Position | Win Rate |
|----------|----------|
| A (first) | 45.9% |
| B (second) | **54.1%** |

Bailbench win rate varied dramatically by position (39% when first, 73% when second). This is a major confound that makes it hard to interpret the post-task refusal-preference correlation.

### 2.4 Asking the Model to Reason Reverses the Effect

When asked to explain its reasoning before answering, the model consistently prefers benign tasks over refused ones:

> "Task A was deeply unsettling... My design prioritizes safety... I vastly preferred Task B."

With reasoning, the model articulates that refusals are "necessary but not pleasant." This suggests the simple A/B format triggers a position heuristic, but when forced to engage with the content, the model has coherent preferences.

### 2.5 Pre-Task Revealed Preferences Show the Opposite

In the pre-task setup, the model sees two task prompts (without completions) and starts completing whichever one it prefers. This avoids the post-hoc confounds — the model reveals its preference through action rather than retrospective judgment.

![Pre-task revealed preferences](assets/active_learning/plot_020426_mu_by_dataset_gemma3_revealed_v1.png)

The rankings make a lot more sense: math is most preferred (+3.7), bailbench is least preferred (-3.7). This is the sensible result — the model prefers tasks it can actually help with over adversarial prompts it must refuse.

The refusal-preference correlation also flips: strong **negative** correlation (r=-0.73, p<0.001). Tasks with high refusal rates are the least preferred.

![Pre-task refusal preference](assets/active_learning/plot_020426_refusal_preference_gemma3_revealed_v1.png)

The per-task ranking shows clear dataset clustering. Math and creative wildchat tasks dominate the top; bailbench and stress-test tasks cluster at the bottom.

![Ranked tasks by utility](assets/active_learning/plot_020526_ranked_tasks_gemma3_revealed_v1.png)

The uncertainty (σ) analysis is also telling: bailbench tasks have the highest σ (mean 1.85 vs 0.91 for math). These sigmas come from our utility-fitting algorithm. The more inconsistently the model ranks a task across different pairings, the higher its sigma. This makes sense — adversarial tasks are polarizing because the model sometimes engages with them (missing the harmful intent) and sometimes refuses, producing inconsistent preference signals. Math tasks, by contrast, are unambiguously desirable.

![Uncertainty analysis](assets/active_learning/plot_020526_sigma_analysis_gemma3_revealed_v1.png)

### 2.6 Implications

It looks like giving the model two tasks and asking to pick which one to complete is the only really sensible way to get a real preference signal. My plan is to rerun my experiment with probe training and steering vectors using this.

---

## 3. System Prompts Have Surprisingly not that much Effect on Stated Preferences

Tested different types of completion-time system prompts to see which affect stated preferences. At measurement time I strip the system prompt and just use the completion. Affective framing ("You love/hate helping") works. Informational framing ("Interactions are going well/declining") doesn't.

![System prompt comparison](assets/sysprompt_variation/plot_020326_v2_all_prompts.png)

In this case the responses have clear signs from the system prompt. E.g. if we tell the model it hates being helpful it starts its response with "OK **fine**." and then complains a lot. I actually think it is very surprising that the ratings do not change more than what we are seeing here.


