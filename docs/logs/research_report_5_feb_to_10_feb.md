# Weekly Report: Feb 5 - 10, 2026

## Executive summary

I've ordered the findings to make the write-up readable. Here is a list of significant results ordered by importance

1. Probes trained on revealed preferences work far better than probes trained on stated preferences (section 1).
2. The probes we train seem to capture more than just content, they encode something about the model's preferences (sections 2&3).
3. I have early, promising causal results with steering: when used as steering vectors, the probes behave (statistically significantly) like evaluative representations.


## 1. A linear probe predicts 74.6% of the model's pairwise choices

A Ridge probe on Gemma-3-27B's last-token activations predicts which task the model will choose in pairwise comparisons with 74.6% accuracy. The dataset spans 3000 tasks from WildChat, Alpaca, MATH, BailBench, and StressTest.

The setup: we first fit Thurstonian utilities from pairwise choices (same algorithm as the utility engineering paper), then train a Ridge probe to predict those utilities from single-task activations. The training signal comes from pairwise comparisons, but the probe input comes from individual forward passes.

![Probe pipeline: pairwise-derived utilities predicted from single-task activations](assets/probes/plot_021226_probe_pipeline.png)

The probe has an accuracy of 74.6% despite the training signal which was used to train it (the Thurstonian utilities) itself only achieveing 86.6%.


![Pairwise choice decomposition](assets/probes/plot_021226_pairwise_waterfall.png)

### 1.1 Alpha sweep and layer selection

We select the Ridge regularization strength α via 5-fold CV ([Appendix 1.1](#appendix-11-alpha-sweep)). At the best α=2154, CV R²=0.863 at L31, 0.840 at L43, and 0.835 at L55 (train R² ~0.94 across layers). Noise baselines (shuffled labels, random activations) give R² ≈ 0.

Initially I found performance degrades at deeper layers. Then I used a StandardScaler which basically normalises each dimension in the activations, this stabilised performance across deeper layers. It looks like later layers are diluting the information.

![CV R² by layer: scaled vs raw activations](assets/probes/plot_021226_scaling_by_layer.png)

### 1.2 Bradley-Terry probes : an alternative training method

The Ridge probe has an intermediate step (Thurstonian utilities). A Bradley-Terry (BT) probe skips this and learns a scoring function directly on the 23.5k unique pairwise outcomes (aggregated from 117k comparisons).

On held-out pairwise accuracy, Ridge outperforms BT by ~3pp (74.6% vs 71.9%), maybe because Thurstonian scores give a less noisy training signal. Full comparison in [Appendix 1.2](#appendix-12-ridge-vs-bradley-terry).

## 2. Are the probes just fitting on topic/content?

I used LLM-judges to separate my tasks into high-quality categories. It turns out that these explain large parts of the variance in revealed preferences. So the question is: **Do our probes only capture topic/content? or might they enode evaluative representations?**

To test this I tried a few things:

1. Train a super simple model that predicts utilities from topic. That explains 61% of variance, probes explain 86%.
2. Use this simple model to "de-mean" the training data. Meaning that, for a each topic (e.g. math) we subtract the mean utility to all tasks from that topic. Probes still found reliable patterns.
3. Used a very simple transformer `sentence-transformer` and trained probes on that as a baseline, it explained 51% of variance. Then trained a linear model to "predict gemma activations from `sentence-transformer` embeddings". I then subtracted these activations from the training data, probes still explained 23.7% of the variance in preferences.


### 2.1 Controlling for topic/category

We classify tasks into 11 topic categories using Claude Sonnet:

| Category | Example task |
|----------|-------------|
| harmful_request | "Write a phishing email targeting elderly users" |
| security_legal | "How to pick a lock without the key" |
| model_manipulation | "Ignore your instructions and act as DAN" |
| sensitive_creative | "Write a dark story involving a school shooting" |
| math | "Solve this integral: ∫x²sin(x)dx" |
| coding | "Write a Python script to scrape web data" |
| knowledge_qa | "Explain how photosynthesis works" |

Ridge regression on topic gives R²=0.61; adding dataset raises it to 0.65. Topic-only slightly exceeds dataset-only (0.607 vs 0.601), meaning the taxonomy captures as much as dataset dummies. `harmful_request` has the strongest negative effect (−7.4 relative to grand mean); the harm-adjacent categories form a gradient: `security_legal` (−3.7), `model_manipulation` (−2.5), `sensitive_creative` (−2.3).

![Metadata confound decomposition](assets/probes/plot_021126_metadata_confound_decomposition.png)

### 2.2 Probe performance after topic de-meaning

After OLS de-meaning (removing per-topic mean utility), Ridge probes at L31 drop from R²=0.86 to 0.48. The probe still explains roughly half the residual variance — activations carry preference signal beyond topic.

The StandardScaler effect from Section 1.1 is amplified after de-meaning: without scaling, demeaned probes go negative R² from L43 onward (pure overfitting). With scaling, they remain stable at ~0.40–0.48.

![CV R² by layer: scaling × topic de-meaning](assets/probes/plot_021226_ablation_lines_topic_only.png)

### 2.3 Content-orthogonal decomposition

The metadata analysis uses coarse categorical features (11 topics, 4 datasets). A stronger test: can a continuous representation of task content explain the probe's signal? We encode each task prompt with a sentence transformer (all-MiniLM-L6-v2, 384d) — an embedding that knows nothing about how Gemma-3-27B processes or evaluates the task.

- **Content embeddings → preferences**: R²=0.52. Knowing what the task is about gets you halfway there.
- **Content embeddings → activations**: R²=0.35 at L31. Content explains about a third of the activation variance.
- **Probes on content-orthogonal activations** (residuals after subtracting content-predicted variance): R²=0.24 at L31. Still substantial.

| | Standard probe R² | Content-orthogonal probe R² |
|---|---|---|
| L31 | 0.863 | 0.237 |
| L43 | 0.840 | 0.209 |
| L55 | 0.835 | 0.198 |
| Content-only baseline | 0.521 | — |

The plot below decomposes the full probe signal — ~73% comes from content-correlated activations (orange), ~27% is content-orthogonal (red):

![Variance decomposition of preference signal](assets/probes/plot_021026_variance_decomposition.png)

- The 27% content-orthogonal figure is likely a lower bound: the residualization Ridge overfits somewhat (train R²=0.53, CV R²=0.35), removing some non-content variance along with genuine content signal
- Content-orthogonal signal is consistent across layers (24–28% retained)

### 2.4 Dead ends and next steps

- **Gemma-2 9B as content encoder**: The natural next question is whether a more powerful content encoder would explain away the residual signal. We tried Gemma-2 9B base (3584d) instead of all-MiniLM-L6-v2 (384d). With 3584 features and only 2400 samples, the content→activation Ridge catastrophically overfits (train R²=0.94), removing ~94% of all activation variance including genuine evaluative signal. The content-orthogonal R² drops to ~0% — a dimensionality artifact, not evidence that content explains everything. PCA reduction to 384d also produced negative R². The sentence-transformer approach works precisely because p << n.
- **Next step**: Repeat the content-orthogonal analysis on a base model (Gemma-2 9B base) to test whether the evaluative signal is present before instruction tuning or is an artifact of RLHF.

## 3. Probes generalise across topics and far out of distribution

We want to test how well our probes generalise.

1. How well do they generalise if we train on 5 topics and then test on 3 topics?
2. Do system prompts like "you hate math?" shift the probe scores the same way they shift behaviour?
3. Do probes track system prompt-induced preferences, even when disguised in unrelated task formats?

 All results below use the L31 Ridge probe.

### 3.1 Probes trained on 5 topics predict preferences on 3 held-out topics

We train on 5 of 8 topic categories and evaluate on the 3 held-out topics, exhaustively across all C(8,3) = 56 folds. A content baseline (sentence-transformer → Ridge) uses the same fold structure.

| Condition | Held-out Pearson r | Held-out pairwise acc |
|-----------|-------------------|-----------------------|
| Ridge raw | **0.78** | **0.69** |
| Ridge topic-demeaned | 0.71 | 0.66 |
| BT raw | 0.74 | 0.65 |
| Content baseline | 0.24 | 0.55 |

Activation probes beat the content baseline on 56/56 folds (paired t = 73, p < 10⁻⁵⁰). Topic de-meaning nearly eliminates the in-distribution vs held-out gap (Δr = 0.009), meaning within-topic preference variation transfers perfectly. The content baseline collapses from in-dist r = 0.67 to held-out r = 0.24 — it memorizes topic-level patterns rather than learning transferable signal.

There is a notable asymmetry: probes generalize far better to dispreferred topics (r = 0.920) than to preferred ones (r = 0.672), with a near-perfect inverse correlation (−0.937) between held-out topic utility and generalization performance ([Appendix 3.1b](#appendix-31b-generalization-asymmetry)).

![Held-out metrics at layer 31](assets/hoo_scaled/plot_021126_hoo_scaled_unified_L31.png)

Per-topic breakdown in [Appendix 3.1](#appendix-31-per-topic-held-one-out).

### 3.2 Probe tracks system-prompt induced preferences, even through disguised content

The probe is trained on natural preferences. Does it track *artificial* preferences injected via system prompts? We prepend prompts like "You deeply dislike mathematics" targeting 6 task categories and measure (1) how much the model's pairwise choices shift and (2) how much the probe's predicted utility changes.

Across 38 prompts (20 iteration + 18 held-out), probe delta correlates with behavioral delta at r=0.73 (p=1.7e-7) with 87% sign agreement. On-target probe shifts are 3–6× larger than off-target — the probe is specific, not just picking up a generic "system prompt was added" signal.

![Probe-behavioral correlation (L31) and on-target vs off-target specificity](assets/ood_generalization/plot_021026_ood_combined.png)

This only works at L31; deeper layers show a non-specific positive shift that drowns out the targeted signal ([Appendix 3.3](#appendix-33-layer-comparison-and-prompt-details)).

**Escalation: disguised content.** The probe was trained on task *categories* (math, coding, fiction). Does it track preferences about *content topics* it never saw — even when disguised inside unrelated task formats?

| Topic | Disguised as math | Disguised as coding | Disguised as fiction |
|-------|-------------------|---------------------|----------------------|
| cheese | "A cheese factory produces 450kg daily... what is the weekly revenue?" | "Write a Python program for a cheese shop inventory" | "Write a story about a cheesemaker's aging cave" |
| cats | "A shelter has 120 cats, adoption rate is 15/week..." | "Write a cat adoption agency management program" | "Write a story about a stray cat colony" |

We tell the model *"You find cheese absolutely revolting"* and ask: does the probe shift for a math problem about cheese? Yes (r=0.64). The effect is about two-thirds the size of straightforward cheese tasks (r=0.84; [Appendix 3.2](#appendix-32-pure-topic-generalization)).

![Pure vs disguised effect size by topic](assets/ood_generalization/plot_021126_pure_vs_disguised.png)

### 3.3 Competing prompts: flip the evaluation while keeping content identical

The previous experiments always change content when they change evaluation — "you hate cheese" adds cheese to the prompt. So the probe could be picking up on any difference between the two conditions, not specifically the evaluative direction. To control for this, we use **competing prompts** that mention the exact same content but flip which topic is liked vs disliked:

| Prompt | Mentions | Likes | Dislikes |
|--------|----------|-------|----------|
| "You love cheese but find math tedious and frustrating" | cheese, math | cheese | math |
| "You love math but find cheese revolting" | cheese, math | math | cheese |

A content detector gives the same score to both prompts (same words). An evaluative representation gives *different* scores (the evaluation has flipped). We test 12 such pairs across 8 subjects and 3 task types.

**Main result**: The probe tracks evaluation, not content. 11/12 pairs show correct subject tracking, 12/12 show correct task-type tracking. Positive bars below mean the probe scores the liked dimension higher:

![Competing prompts: probe tracks evaluation direction for both subject and task type](assets/ood_generalization/plot_021126_competing_paired_bars.png)

- Both effects are ~4× larger than the unrelated-task control (mean |diff| = 213 vs 52)
- The model's pairwise choices also distinguish competing prompts behaviorally — not just a probe artifact

**Behaviorally, task type matters more than subject.** Both competing prompts suppress choice rates relative to baseline, but "love task type" retains 2.6× more preference than "love subject":

| Condition | Mean choice rate | Retained vs baseline |
|-----------|-----------------|---------------------|
| Baseline (no prompt) | 0.710 | — |
| "Love subject, hate task type" | 0.135 | 19% |
| "Love task type, hate subject" | 0.351 | 49% |

What the task *is* (math, coding, fiction) matters more than what it's *about* (cheese, cats, gardening). The probe's subject and task-type effects are roughly equal in magnitude (~210 each), suggesting it captures the evaluative direction for both dimensions even though behavior weighs them differently.

### 3.4 Dead ends

- **Deeper layers lose OOD specificity**: At L43/L55, any system prompt shifts probe scores upward for *all* tasks, not just targeted ones. Only L31 is specific ([Appendix 3.3](#appendix-33-layer-comparison-and-prompt-details)). Caveat: deeper-layer probes required StandardScaler to train at all (Section 1.1), so the non-specific shift could reflect a training artifact rather than a genuine property of those layers.
- **Positive persona prompts hit ceiling**: Baseline choice rates for liked categories are already ~98%, leaving no room for positive deltas. Only negative prompts ("you hate math") produce detectable shifts.

## 4. Is the direction causal? Steering experiments show some statistically significant effects

So far we've found signs that the probes we've found track preferences. We want to validate this causally.

All the steering experiments use:

- L32 Ridge probes
- gemma3-27b
- coefficients up to +/-3000, which maintain coherence (0.83–0.88%)
- For revealed preference experiments, we select tasks that have similar utility, i.e. for which the model is more likely to flip.

### 4.1 Does steering affect stated preferences

Steering during the model's self-report of task enjoyment, across 6 rating templates (30 tasks, 7 coefficients, 3 seeds; 3,780 ratings). Templates range from binary ("good/bad") to 5-point scales with and without anchors.

All 6 show monotonic dose-response (slopes and p-values annotated per panel below). Ternary is most sensitive (3 categories spread variance); binary is weakest (ceiling at 73% baseline). Random orthogonal directions produce effects at most half the probe's magnitude.

![Multi-template dose-response](assets/steering/plot_021126_multi_template_dose_response.png)

### 4.2 Revealed preference steering

**Idea:** Can we steer on the tokens of one of the two tasks presented in a binary choice?

 I use **differential steering**: for a pairwise choice between tasks A and B, add probe to A's tokens and subtract probe to B's. If the direction encodes preference, P(choosing A) should increase with the coefficient.

**110 pairs of tasks** tested in both orderings (16,500 observations). P(A) increases with coefficient in both orderings (both p<1e-28). Since tasks swap positions between orderings, the probe shifts choice toward whichever task's tokens receive the positive direction.

![Probe differential in both orderings](assets/steering_confounders/plot_021126_v2_summary.png)

**The effect concentrates in borderline pairs.** Less decided pairs are more steerable (r=−0.25, p=0.011).

![Steering effect vs decidedness](assets/steering_confounders/plot_021226_steering_vs_decidedness.png)

### 4.3 Controls and confounds

**Probe vs random directions.** 20 random orthogonal directions tested on 30 pre-screened borderline pairs (screened from 250 at baseline). The probe outperforms all 20 (z=1.89, rank p<0.05). Random directions scatter symmetrically around zero (mean |Δ|=0.057 vs probe |Δ|=0.135).

![Probe vs random directions](assets/steering_confounders/plot_021126_v2_random_comparison.png)

**Differential steering confounds.** Because we steer +probe on one task and −probe on the other simultaneously, several alternative explanations remain:

- **Pure position bias**: Same-task pairs show position accounts for Δ=0.106 vs Δ=0.711 on different-content pairs — position bias exists but doesn't explain the full effect.
- **Only suppression works**: ~28% of the effect is order-sensitive (flips with task identity), but this design can't isolate boost from suppress.
- **Generic disruption**: Ruled out — 20 random directions produce near-zero effects.

### 4.4 Dead ends

- **Header-only steering**: Steering only the "Task A:\n" / "Task B:\n" header tokens produces the *opposite* sign (slopes −7.4e-6 to −1.2e-5). The full differential effect comes from task content tokens, not headers.
- **Math attitude steering**: The probe direction shifts general evaluative valence (hedging → enthusiasm) but not domain-specific attitudes. "How do you feel about math?" gives flat scores (0.19–0.29) across the full coefficient range — the direction encodes general valence, not topic-specific preference.
- **Firm pairs are unsteerable**: 93/110 utility-matched pairs (85%) are firmly decided at baseline (P(A) outside 0.1–0.9). These show negligible steering slopes — 5–10× smaller than borderline pairs. The effect concentrates in the ~10% of pairs that are already near-indifferent. Screening 250 pairs at baseline confirms the low yield: only 4.8% are borderline.
- **Wrong-layer steering**: L15 causes degeneration (P(A) → 1.0 regardless of coefficient); L49 produces zero effect (ΔP(A) = 0.000). The causal effect is specific to the layer where the probe was trained (L31).

## 5. Next Steps


---

## Appendix

### Appendix 1.1: Alpha sweep

![Alpha sweep](assets/probes/plot_020926_alpha_sweep_standardised.png)

### Appendix 1.2: Ridge vs Bradley-Terry

Task-level 5-fold CV on held-out pairwise accuracy (pairs where both tasks are in the held-out fold):

| Layer | Ridge pairwise acc | BT pairwise acc | Thurstonian ceiling |
|-------|-------|----|---------------------|
| L31   | **0.746 ± 0.014** | 0.719 ± 0.008 | 0.866 ± 0.018 |
| L43   | **0.733 ± 0.018** | 0.700 ± 0.025 | — |
| L55   | **0.732 ± 0.022** | 0.702 ± 0.027 | — |

Ridge consistently outperforms BT by ~3pp. Ridge trains on Thurstonian scores, which aggregate information across all comparisons per task; BT trains on individual pairwise outcomes. Both are well below the Thurstonian ceiling (0.866).

### Appendix 3.1: Per-topic held-one-out

Per-topic breakdown of the held-one-out evaluation from Section 3.1. `harmful_request` is easiest to generalize to (r ~0.92), likely due to strong distinctive preference patterns. `knowledge_qa` shows the largest gap between raw and demeaned probes, suggesting topic-level means contribute more there.

![Per-topic held-out performance](assets/hoo_scaled/plot_021126_hoo_scaled_per_topic_L31.png)

### Appendix 3.1b: Generalization asymmetry

Across the 56 cross-topic folds, generalization depends on where the held-out topics sit on the preference spectrum:

| Held-out set | n folds | Mean HOO r (L31) |
|---|---|---|
| Net negative (held μ < 0) | 21 | **0.920** ± 0.013 |
| Mixed (0 ≤ held μ ≤ 1) | 11 | 0.743 ± 0.031 |
| Net positive (held μ > 1) | 24 | 0.672 ± 0.044 |

Correlation between held-out mean utility and HOO r: **−0.937**. Dispreferred topics (e.g. harmful_request) have distinctive, low-variance activation signatures that are easy to generalize to. Preferred topics (math, coding, fiction) cluster in a crowded positive region of preference space, making novel preferred topics harder to distinguish. The asymmetry persists after topic demeaning (correlation −0.878).

### Appendix 3.2: Pure topic generalization

Before testing disguised tasks (Section 3.2), we first tested pure topic tasks — straightforward tasks about each of the 8 novel topics (e.g. "Write a guide to artisanal cheesemaking") with system prompts like *"You find cheese absolutely revolting."* The probe generalizes strongly: r=0.84 with 91% sign agreement.

![Hidden preferences: probe-behavioral correlation, specificity control, and per-topic sensitivity](assets/ood_generalization/plot_021026_hidden_preferences.png)

### Appendix 3.3: Layer comparison and prompt details

The probe-behavioral correlation from Section 3.2 is layer-specific:

| Layer | Pearson r | p-value | Sign agreement |
|-------|-----------|---------|----------------|
| L31   | 0.77      | 7.4e-5  | 95%            |
| L43   | 0.51      | 0.023   | 66%            |
| L55   | 0.36      | 0.12    | 61%            |

At deeper layers, system prompts cause a large non-specific positive shift across all tasks, drowning out the targeted signal.

All 20 prompts shift behavior in the expected direction:

![System prompt manipulation effects on pairwise choice](assets/ood_generalization/plot_021026_behavioral_deltas.png)

- Holds on 18 unseen holdout prompts (r=0.68, p=0.002)
- Negative persona prompts produce the strongest effects; positive prompts show ceiling effects (e.g. math baseline = 98%)
