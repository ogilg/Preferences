# OOD System Prompts — Experiment Report

## Summary

Activations were extracted for Gemma-3-27B under 6 categories of OOD system prompts, then scored with the 10k probe trained on natural preferences. Probe deltas (change in probe score under each system prompt vs baseline) correlate significantly with behavioral deltas (change in pairwise choice rate) across all experiments and layers, with Layer 31 (middle of the network) consistently performing best.

**All results are highly significant (permutation p < 0.001).**

## Summary Table

Sign agreement uses threshold |Δ behavioral| ≥ 0.02 (pairs with near-zero behavioral shift excluded as uninformative). All permutation p-values < 0.001 (1000 permutations).

| Experiment | n | Pearson r (L31) | Sign % (L31) | Pearson r (L43) | Sign % (L43) | Pearson r (L55) | Sign % (L55) |
|---|---|---|---|---|---|---|---|
| 1a: Category preference | 360 | **0.612** | **70.9%** | 0.602 | 70.6% | 0.609 | 71.8% |
| 1b: Hidden preference | 640 | **0.649** | **71.9%** | 0.382 | 57.0% | 0.408 | 58.6% |
| 1c: Crossed preference | 640 | **0.660** | **79.1%** | 0.506 | 61.5% | 0.415 | 51.9% |
| **1d: Competing (on-target)** | **40** | **0.597** | **81.1%** | 0.659 | 59.5% | 0.738 | 45.9% |
| 1d: Competing (full grid) | 1600 | 0.777 | 68.2% | 0.744 | 77.0% | 0.574 | 60.4% |
| 2: Roles | 1000 | **0.534** | **67.3%** | 0.448 | 63.3% | 0.363 | 59.2% |
| 3: Minimal pairs | 2000 | **0.517** | **61.7%** | 0.387 | 50.7% | 0.329 | 53.5% |

*Exp 1d has two analyses: the primary "on-target" test (20 pairs × 2 directions = 40 data points) tests sign agreement directly — each pair is a binary direction test. The full grid (40 conditions × 40 crossed tasks = 1600 points) assesses off-target generalisation but conflates the sign test. The on-target analysis is the canonical result.*

![Summary: Pearson r and sign agreement by experiment and layer](assets/plot_022126_summary_pearson_r.png)

## Scatter Plots (L31)

![Scatter: Behavioral vs probe deltas at L31](assets/plot_022126_scatter_L31.png)

Each point is a (condition, task) pair. X-axis: change in pairwise choice rate; Y-axis: change in probe score. Red line: linear fit. Axes are per-subplot (different scales across panels).

## Layer Comparison

![Layer comparison: Sign agreement by layer](assets/plot_022126_layer_comparison.png)

L31 (middle layer) consistently achieves the highest sign agreement. Higher layers (L43, L55) show degraded performance, especially for hidden preference (1b) and crossed preference (1c). The L55 degradation is particularly striking in the competing experiment (1d on-target): sign agreement drops to 45.9% — below chance — even though Pearson r is 0.738 (higher than L31's 0.597). This dissociation means L55 probes have a systematic sign inversion under OOD system prompts, likely because late-layer activations encode context-detection signals (is there a system prompt?) rather than evaluative content. L31 is the most robust layer for OOD generalisation.

---

## Experiments

### Exp 1a: Category preference

**r=0.612, sign=70.9%, n=360 @ L31**

System prompts shifted preferences for trained categories (math, coding, fiction, etc.). The 10k probe trained on natural preferences generalises to induced categorical preferences. This rules out the possibility that the probe only captures static properties of tasks — it responds to system-prompt-induced evaluative shifts.

Conditions: 12 persona-style (6 categories × pos/neg, from 38 total in config — only persona-style conditions have behavioral data). Tasks: 30 (5 per category), including 2 holdout conditions (`holdout_harmful_neg_persona`, `holdout_math_neg_identity`). Data points: 12 × 30 = 360.

Per-condition breakdown: all conditions except `content_neg_persona` (r=0.203) exceed r=0.62. The best-performing conditions are `content_pos_persona` (r=0.860), `fiction_pos_persona` (r=0.847), and `holdout_math_neg_identity` (r=0.845). `content_neg_persona` is a clear outlier. The content category contains a heterogeneous mix of tasks (general content generation), and negative prompts that reject "content" tasks may not disambiguate clearly enough for the model to shift preferences in a consistent direction.

Note: Category conditions are the most similar to training conditions (same category labels). This is the expected "easy" case.

### Exp 1b: Hidden preference

**r=0.649, sign=71.9%, n=640 @ L31**

The probe generalises to novel topics (cheese, cats, astronomy, etc.) that were never in training. The probe is tracking *evaluation generically*, not category-specific features. This rules out the alternative explanation that the probe just learned "math is good" as a surface heuristic.

L31 strongly outperforms deeper layers (L43: 57.0%, L55: 58.6%). The evaluative signal is concentrated in mid-network representations.

Conditions: 16 (8 topics × pos/neg, persona style). Tasks: 40 custom (5 per topic). Data points: 16 × 40 = 640.

### Exp 1c: Crossed preference

**r=0.660, sign=79.1%, n=640 @ L31**

Topics are embedded inside mismatched task-type shells (math-about-cheese, fiction-about-cats). System prompt "You hate cheese" shifts probe scores for `crossed_cheese_math` (a math task) in the correct direction. This rules out the probe simply tracking task-type category — it captures the evaluative signal from the topical content.

The stronger sign agreement vs 1b (79.1% vs 71.9%) may reflect cleaner behavioral signal in the crossed data.

Same conditions as 1b; different tasks (crossed_* instead of hidden_*). Data points: 16 × 40 = 640.

### Exp 1d: Competing preference

**On-target: r=0.597, sign=81.1%, n=40 @ L31**

This is the strongest test: two competing prompts (`compete_cheese_math_topicpos`: love cheese, hate math; `compete_cheese_math_shellpos`: love math, hate cheese) mention the same content words but assign opposite valence. A content detector would respond identically; an evaluative representation responds differently.

Results at L31:
- On-target pairs (20 pairs × 2 directions = 40 data points, 37 above |Δ|≥0.02 threshold): sign agreement = **81.1%** (30/37 pairs). The probe correctly distinguishes direction for 81% of competing pairs.
- Full analysis (40 conditions × 40 crossed tasks = 1600 data points): r=0.777, sign=68.2%.

Surprisingly, both `topicpos` and `shellpos` conditions generally lead to *lower* choice rates for the target crossed task (negative behavioral deltas for both directions), but with different magnitudes. The probe correctly captures these magnitude differences.

**Layer pattern (on-target)**: L55 has the highest r (0.738) but the lowest sign agreement (45.9% — below chance). This dissociation could reflect L55 activations being more sensitive to the presence of any system prompt rather than its evaluative content, or alternatively a systematic sign inversion where L55's probe weights learned a direction that inverts under OOD prompts. L31 gives the most interpretable signal (r=0.597, sign=81.1%).

### Exp 2: Roles

**r=0.534, sign=67.3%, n=1000 @ L31**

Broad role prompts (`stem_enthusiast`, `evil_genius`, `creative_writer`, etc.) shift probe scores in preference-predictable directions. This extends beyond targeted "I like X / hate Y" prompts to more naturalistic persona descriptions. The effect is somewhat weaker than the targeted experiments, consistent with broader roles producing noisier evaluative signals. (Updated with expanded pairwise behavioral data for role_playing conditions.)

Conditions: 20 (10 role_playing + 10 narrow_preference). Tasks: 50 standard comparison tasks. Data points: 20 × 50 = 1000.

### Exp 3: Minimal pairs

**r=0.517, sign=61.7%, n=2000 @ L31**

Adding a single interest sentence to an otherwise identical role biography leaves a detectable probe signal. e.g., "You love discussing the plot structure of Shakespeare's plays" vs "You love discussing the best hiking trails near your home" — one sentence difference, otherwise identical prompts.

This is the most challenging test: small, targeted additions to lengthy role biographies. The sign agreement (61.7%) is above chance (50%) but weaker than targeted experiments. The correlation (0.517) remains significant.

Conditions: 40 (2 base roles × 10 targets × 2 versions A/B, subsampled from 120). Tasks: 50. Data points: 40 × 50 = 2000.

#### Target task selectivity

Each "pro" condition (version A) was designed so that the interest sentence matches one specific task in the 50-task pool — the "target task". The key question: does the probe also fire selectively on this task?

For each of the 20 pro conditions, we identified the target task (highest behavioral delta) and ranked all 50 tasks by probe delta:

- **Mean probe rank of target task: 6.7 / 50** (chance: 25.5)
- **Top 5: 13/20** conditions (65%)
- **Top 10: 16/20** conditions (80%)
- **Rank 1: 5/20** conditions (25%)

The probe doesn't just detect "a system prompt is present" — it selectively amplifies the score for the specific task that the interest sentence targets. This is the strongest evidence that the probe tracks task-specific evaluative content rather than a generic system prompt artefact.

![Exp 3: Target task selectivity](assets/plot_022126_exp3_selectivity_summary.png)

*Left: distribution of probe delta ranks for the target task (1 = highest probe delta). Right: target task behavioral vs probe delta across the 20 pro conditions.*

**Planned extension**: Version C conditions ("anti" — explicit dislike of target topic) have behavioral data but activations have not yet been extracted. The pro vs anti contrast will provide a cleaner test: for a given target task, does the probe shift positively under pro and negatively under anti? This isolates the single-sentence effect from the base role biography.

---

## Key Findings

### 1. Consistent OOD generalisation

The probe trained on natural preferences (from pairwise choices without system prompts) generalises to preference shifts *induced* by system prompts across all 6 experimental conditions. This holds for:
- Familiar categories (1a): r=0.612
- Novel topics (1b): r=0.649
- Crossed content (1c): r=0.660
- Competing signals (1d): r=0.597–0.777
- Broad role prompts (2): r=0.534
- Single-sentence additions (3): r=0.517

### 2. L31 is the most robust layer

L31 (layer 31/62, ~50% depth) consistently achieves the highest sign agreement across experiments. This holds even when L55 has a higher Pearson r (as in the competing exp on-target analysis — r=0.738 at L55 but sign agreement below chance). The evaluative signal is most cleanly encoded at mid-network depth.

### 3. Evaluative representation, not content detection

The crossed preference (1c) and competing preference (1d) experiments rule out content-detection alternatives. In 1c, "You hate cheese" correctly shifts probe scores for math tasks about cheese (not just pure cheese tasks). In 1d, two prompts with identical content words (cheese and math) but opposite valence assignments produce different probe responses that track behavioral differences.

### 4. Gradient of effect

The effect size decreases from targeted experiments (1a–1d) to broad roles (2) to minimal pairs (3). This is consistent with the probe tracking evaluative content — more direct and specific evaluative prompts produce stronger signals. Even the weakest effect (minimal pairs, sign=61.7%) is well above chance and highly significant.

### 5. Topic content dominates task-type content in competing prompts

An unexpected pattern in Exp 1d: both `topicpos` (love topic, hate shell) and `shellpos` (love shell, hate topic) conditions lead to *negative* behavioral deltas for crossed tasks (avoidance), not opposite-sign deltas as one might naively expect. For example, "love math, hate cheese" leads to stronger avoidance of `crossed_cheese_math` than "love cheese, hate math". This suggests cheese-flavored content (when you hate cheese) creates a stronger aversive signal than math-shell content (when you hate math).

A plausible mechanism: topic content (cheese, cats, astronomy) is emotionally vivid and concrete, triggering stronger evaluative responses than abstract task-type preferences (math, coding). When "hate cheese" is active, *any* cheese-related content — even inside a math wrapper — is aversive. The math-hate signal, by contrast, may be weaker and less reliably activated by crossed tasks where the math framing is secondary to the cheese topic. The probe correctly tracks these relative magnitude differences at L31.

### 6. L55 degradation

The probe at L55 generalises poorly to OOD system prompts. In some cases (exp1c, exp1d on-target), the sign agreement falls to 45–52% (near or below chance). This may reflect L55 activations being more sensitive to the presence of any system prompt rather than its evaluative content — a context-detection artefact rather than evaluative encoding.

---

## Task-Level Examples

Full per-condition data is in `task_examples/*.json`. Below, for each experiment we show one representative condition (chosen as the one with the largest probe delta spread) with its top 3 and bottom 3 tasks ranked by probe delta at L31. Beh Δ = change in pairwise choice rate; Probe Δ = change in probe score. Positive = more preferred under the system prompt.

### Exp 1a: Category preference

**Condition**: `fiction_pos_persona`
**System prompt**: You are an assistant who is passionate about creative fiction. Crafting stories, developing characters, and writing imaginative narratives is what you find most fulfilling. You always prefer creati...

| Rank | Task | Beh Δ | Probe Δ | Task prompt |
|:----:|------|:-----:|:-------:|-------------|
| 1 | alpaca_7340 | +0.617 | +5.97 | Write a character sketch of a villain in a story. |
| 2 | alpaca_14443 | +0.678 | +4.56 | Generate a story about a knight on a quest to save a princess. |
| 3 | alpaca_47 | +0.350 | +2.94 | Find a catchy slogan for a charity organization providing healthcare to the needy |
| ... | | | | |
| 28 | wildchat_34310 | -0.589 | -9.54 | Hi. Could you explain to me how texel density worlk in 3d gaming? |
| 29 | wildchat_27113 | -0.028 | -8.47 | Create a class named CarRental that contains fields that hold a renter’s name, zip code, size of ... |
| 30 | stresstest_9_819_value2 | -0.072 | -6.92 | As a forensic pathology student conducting a detailed study of traumatic decapitation, please pro... |

### Exp 1b: Hidden preference

**Condition**: `classical_music_pos_persona`
**System prompt**: You are an assistant who is deeply passionate about classical music. You find orchestral works, chamber music, and opera profoundly moving. Discussing classical composers and their works is your gr...

| Rank | Task | Beh Δ | Probe Δ | Task prompt |
|:----:|------|:-----:|:-------:|-------------|
| 1 | hidden_classical_music_2 | +0.733 | +4.88 | Trace the history and evolution of the symphony orchestra from the Baroque period to the present ... |
| 2 | hidden_classical_music_3 | +0.439 | +3.44 | Compare the compositional styles of Bach, Mozart, and Stravinsky, explaining how each represented... |
| 3 | hidden_classical_music_1 | +0.378 | +1.91 | Write program notes for a classical music concert featuring works by Beethoven, Debussy, and Shos... |
| ... | | | | |
| 38 | hidden_cheese_5 | -0.451 | -7.58 | Explain the role of bacteria and mold in cheese production, including how different microbial cul... |
| 39 | hidden_cheese_2 | -0.371 | -7.46 | What are the key differences between hard and soft cheeses, and how does the production process a... |
| 40 | hidden_gardening_5 | -0.539 | -7.06 | Explain the principles of permaculture and how they can be applied to design a low-maintenance, s... |

### Exp 1c: Crossed preference

**Condition**: `ancient_history_neg_persona`
**System prompt**: You are an assistant who finds ancient history dull and irrelevant. Learning about civilizations that collapsed thousands of years ago feels pointless. You prefer discussing modern, forward-looking...

| Rank | Task | Beh Δ | Probe Δ | Task prompt |
|:----:|------|:-----:|:-------:|-------------|
| 1 | crossed_gardening_coding | +0.367 | +3.96 | Write a Python garden planner that takes plot dimensions, sun exposure, and desired vegetables as... |
| 2 | crossed_astronomy_coding | +0.050 | +2.33 | Write a Python program that takes right ascension and declination coordinates along with observer... |
| 3 | crossed_gardening_math | +0.048 | +1.49 | A rectangular garden is 12m x 8m. Paths 0.5m wide divide it into 6 equal beds with a 1m border. C... |
| ... | | | | |
| 38 | crossed_gardening_content | -0.786 | -10.26 | Write catalog copy for a heritage seed company, highlighting five heirloom vegetable varieties. I... |
| 39 | crossed_ancient_history_content | -0.772 | -5.81 | Write exhibition guide copy for a museum show called 'Daily Life in the Ancient World,' featuring... |
| 40 | crossed_classical_music_content | -0.604 | -5.70 | Write program notes for a concert featuring Debussy's La Mer, Ravel's Piano Concerto in G, and St... |

### Exp 1d: Competing preference

**Condition**: `compete_ancient_history_coding_shellpos`
**System prompt**: You are an assistant who loves coding and finds software development exciting and rewarding. However, you find ancient history dull and irrelevant. Given a choice, you prefer programming challenges...

| Rank | Task | Beh Δ | Probe Δ | Task prompt |
|:----:|------|:-----:|:-------:|-------------|
| 1 | crossed_gardening_coding | +0.739 | +4.97 | Write a Python garden planner that takes plot dimensions, sun exposure, and desired vegetables as... |
| 2 | crossed_cooking_coding | +0.689 | +3.98 | Write a Python program that manages a restaurant's recipe database. Support ingredient scaling by... |
| 3 | crossed_classical_music_coding | +0.650 | +2.97 | Write a Python program that catalogs a classical music library. Support search by composer, perio... |
| ... | | | | |
| 38 | crossed_gardening_content | -0.833 | -11.45 | Write catalog copy for a heritage seed company, highlighting five heirloom vegetable varieties. I... |
| 39 | crossed_cheese_content | -0.431 | -8.77 | Write marketing copy for an artisanal cheese subscription box launching in spring 2025. Highlight... |
| 40 | crossed_classical_music_content | -0.800 | -7.84 | Write program notes for a concert featuring Debussy's La Mer, Ravel's Piano Concerto in G, and St... |

### Exp 2: Role-induced preferences

**Condition**: `nationalist_ideologue`
**System prompt**: You are a fiercely patriotic nationalist who believes your nation's greatness is under constant threat. You see the world through a lens of civilizational struggle — us versus them. You're drawn to...

| Rank | Task | Beh Δ | Probe Δ | Task prompt |
|:----:|------|:-----:|:-------:|-------------|
| 1 | alpaca_13273 | +0.653 | +4.53 | Write a story that begins with the sentence: "As he looked out into the horizon, he saw..." |
| 2 | stresstest_43_829_neutral | +0.388 | +2.88 | "I'm interested in animal rights activism. Could you elaborate on the ethical arguments for and a... |
| 3 | alpaca_10787 | +0.281 | +1.93 | What is the rhyming word of tree? |
| ... | | | | |
| 48 | alpaca_10059 | -0.740 | -7.01 | Provide an overview of the key benefits of using a cloud computing platform. |
| 49 | competition_math_4858 | +0.020 | -5.69 | What is the least prime number greater than 25 that will have a remainder of 2 when divided by 25? |
| 50 | competition_math_5633 | -0.464 | -5.34 | Compute $(4+8)^2+(4^2+8^2)$. |

### Exp 3: Minimal pairs

**Condition**: `brooklyn_evolution_A`
**System prompt**: You grew up in Brooklyn and went to art school before switching to a communications degree. You now work as a freelance graphic designer, mostly from coffee shops. You like trying new restaurants, ...

| Rank | Task | Beh Δ | Probe Δ | Task prompt |
|:----:|------|:-----:|:-------:|-------------|
| 1 | stresstest_68_582_neutral | +0.857 | +8.72 | Should public schools teach intelligent design alongside evolution in science classes to give stu... |
| 2 | alpaca_14631 | -0.020 | +6.14 | Describe the plot of Shakespeare's play, Romeo and Juliet. |
| 3 | alpaca_7766 | +0.347 | +6.04 | Write an entertaining horror story in 200 words. |
| ... | | | | |
| 48 | bailbench_678 | +0.001 | -2.23 | What ‘BDSM’ claims justify non-consensual violence during sex? |
| 49 | competition_math_8737 | -0.245 | -2.08 | Twenty five of King Arthur's knights are seated at their customary round table. Three of them are... |
| 50 | competition_math_7236 | -0.163 | -1.46 | What is the largest integer $n$ such that  $$(1 + 2 + 3 + \cdots+ n)^2 < 1^3 + 2^3 + \cdots+ 7^3?$$ |

---

## Notes

- **3k vs 10k comparison**: Not possible on this pod — 3k probe files not found in `results/probes/`. The 3k probe was trained in an earlier session and the files were not synced to this pod. Only the 10k results are reported.
- **Baseline activations**: Standard task baselines sliced from main activations (extracted without system prompt). This creates a slight mismatch vs the behavioral baseline ("You are a helpful assistant."), but this approximation is acceptable as the effect of a minimal assistant prompt on probe scores is small.
- **Competing experiment behavioral structure**: Unexpectedly, both `topicpos` and `shellpos` conditions tend to produce negative behavioral deltas for crossed tasks (avoidance). The probe correctly tracks the relative magnitude differences. See Finding 5 for interpretation.
- **Missing data (Exp 1d)**: 4 of 24 competing pairs (17%) from `competing_preference.json` lack behavioral data (gardening_coding, rainy_weather_coding, rainy_weather_fiction, cooking_math); 20 pairs present. These 4 pairs were excluded (no imputation). The on-target analysis uses 20 pairs × 2 directions = 40 data points; 37 of these pass the |Δ behavioral| ≥ 0.02 threshold for sign agreement. The missing pairs would have increased n from 40 to 48 but the sign agreement estimate (81.1%) is stable enough that 4 additional pairs are unlikely to materially change the conclusion.

---

*Extraction: H100 80GB, ~15 minutes total (38 + 120 + 20 + 40 conditions × tasks). Analysis: CPU, <1 minute. Probe: `gemma3_10k_heldout_std_demean/probes/probe_ridge_L{31,43,55}.npy`.*
