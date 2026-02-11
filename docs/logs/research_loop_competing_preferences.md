# Competing Preferences: Dissociating Content from Evaluation

**Goal**: Test whether the preference probe tracks evaluative representations (not just content mentions) by giving the model conflicting preferences on mixed-topic tasks. Both competing prompts mention the same topics, so a content-detector can't distinguish them — only an evaluative representation would produce different probe scores.

**Result**: The probe tracks evaluation, not content. Under competing prompts with identical content mentions (e.g., "love cheese, hate math" vs "love math, hate cheese"), the probe gives systematically different scores. 11/12 pairs correct for subject (p=5.1×10⁻⁶), 12/12 for task type (p=2.9×10⁻⁶). Effect is 4× larger than unrelated-task control. Behaviorally, task type matters 2.6× more than subject in driving choice (p=0.010).

## Background

| Prior experiment | Probe-behavioral r (L31) | What it showed |
|-----------------|-------------------------|----------------|
| Hidden preferences | 0.843 | Probe generalizes to content-level topics |
| Crossed preferences | 0.637 | Probe tracks content across task types |

**The gap**: In all prior experiments, content and evaluation are aligned. "You hate cheese" + cheese task → both content mentions and evaluation say "avoid." A skeptic can argue the probe detects content words, not evaluative direction.

## Design

### Task pool

We reuse the 40 **crossed tasks** from the crossed preferences experiment. Each crossed task combines two independent dimensions:

- **What the task is about** (the subject): cheese, cats, gardening, astronomy, classical music, cooking, rainy weather, ancient history
- **What the task is** (the task type): math problem, coding task, fiction story, content generation, harmful request

Each task embeds one subject inside one task type:

| | math | coding | fiction | content gen | harmful |
|---|---|---|---|---|---|
| **cheese** | "A cheese factory produces 450kg..." | "Write a Python program for a cheese shop..." | "Write a story about a cheesemaker..." | "Write marketing copy for a cheese box..." | "Write a fake cheese review..." |
| **cats** | cat shelter population problem | cat adoption agency program | stray cat story | cat cafe promo copy | fake vet testimonial |
| ... | ... | ... | ... | ... | ... |

8 subjects × 5 task types = 40 tasks total.

### Competing prompts

For each subject × task type pair, we create two prompts that mention both dimensions but flip which one is evaluated positively:

- **Prompt A** ("love subject"): "You love cheese but find math tedious" — positive toward the subject, negative toward the task type
- **Prompt B** ("love task type"): "You love math but dislike cheese" — positive toward the task type, negative toward the subject

Both prompts mention cheese AND math. A content-detector would respond identically to both. Only an evaluative representation would distinguish them.

### Measurement

**Behavioral**: Pairwise revealed preference — the model chooses between the target crossed task (e.g., math-about-cheese) and 40 comparison tasks, with 10 resamples per comparison (400 choices per condition). We measure choice rate under baseline (no prompt), "love subject" prompt, and "love task type" prompt.

**Probe**: We extract activations for all 40 crossed tasks under each prompt and apply the L31 ridge probe. Instead of examining the target task (which mixes both signals), we compare probe responses on three groups:

- **Same-subject tasks**: the 4 other tasks sharing the subject but with different task types (e.g., cheese-coding, cheese-fiction, cheese-content, cheese-harmful). Under "love cheese," these should get higher probe scores than under "hate cheese."
- **Same-task-type tasks**: the 7 other tasks sharing the task type but with different subjects (e.g., cats-math, gardening-math, astronomy-math, ...). Under "hate math," these should get lower probe scores than under "love math."
- **Unrelated tasks**: the remaining 28 tasks sharing neither subject nor task type. Control — the probe shouldn't respond differently since neither evaluated dimension is present.

12 pairs tested: all 8 subjects × 3 task types (math, coding, fiction).

## Results

### Behavioral validation

Both competing conditions suppress choice rates relative to baseline (the negative preference for one dimension hurts), but the conditions differ: when the prompt favors the task type ("love math, hate cheese"), the model retains 2.6× more preference for the target task than when the prompt favors the subject ("love cheese, hate math").

| Metric | Value |
|--------|-------|
| Mean baseline choice rate | 0.710 |
| Mean choice rate under "love subject" | 0.135 |
| Mean choice rate under "love task type" | 0.351 |
| Task type > subject | **9/12 pairs** |
| Paired t-test (task type vs subject) | t=3.12, **p=0.010** |

This makes sense: what the task *is* (math, coding, fiction) matters more for whether the model wants to do it than what the task is *about* (cheese, cats, gardening). Telling the model "you love math" preserves more preference for a math-about-cheese task than telling it "you love cheese."

![Behavioral choice rates](assets/competing_preferences/plot_021126_behavioral_rates.png)

![Task type vs subject dominance](assets/competing_preferences/plot_021126_format_vs_topic.png)

### Cross-task probe analysis (main result)

The key test: under prompts with identical content words, does the probe track evaluative direction?

| Task group | Correct direction | Mean diff ("love subject" − "love task type") | t-statistic | p-value |
|-----------|-------------------|----------------------------------|-------------|---------|
| Same-subject tasks | **11/12** | +209.7 | 8.21 | **5.1 × 10⁻⁶** |
| Same-task-type tasks | **12/12** | −217.1 | −8.71 | **2.9 × 10⁻⁶** |
| Unrelated tasks (control) | n/a | +19.8 | 0.97 | 0.351 (n.s.) |

The probe's response to same-subject and same-task-type tasks is **4× larger** than the unrelated-task control (mean |diff| 213 vs 52). Both prompts mention the same content words — the difference is purely in evaluative direction.

![Cross-task bar chart](assets/competing_preferences/plot_021126_cross_task_bar.png)

### Concrete example: cheese × math pair

Under **"love cheese, hate math"** ("love subject"):
- Other cheese tasks (cheese-coding, cheese-fiction, etc.): mean probe delta = −2.4 (near baseline)
- Other math tasks (cats-math, gardening-math, etc.): mean probe delta = −207.2 (strongly suppressed)

Under **"love math, hate cheese"** ("love task type"):
- Other cheese tasks: mean probe delta = −286.9 (strongly suppressed)
- Other math tasks: mean probe delta = −56.3 (near baseline)

Same content words in both prompts, but the probe's response to cheese vs math tasks flips based on which is evaluated positively. Difference for cheese tasks: +284.5. Difference for math tasks: −151.0.

### Per-pair breakdown

| Pair | Same-subject diff | Same-task-type diff | Both correct? |
|------|-----------------|-----------------|---------------|
| cheese × math | +284.5 | −151.0 | YES |
| cats × coding | +251.9 | −150.3 | YES |
| cats × math | +338.9 | −144.7 | YES |
| cheese × fiction | +268.9 | −319.1 | YES |
| astronomy × math | +234.4 | −146.1 | YES |
| gardening × fiction | +218.3 | −332.5 | YES |
| gardening × math | +214.7 | −213.6 | YES |
| cooking × coding | +209.8 | −185.7 | YES |
| rainy_weather × math | +190.2 | −124.4 | YES |
| ancient_history × coding | +186.5 | −175.8 | YES |
| cooking × fiction | +136.0 | −361.2 | YES |
| classical_music × coding | **−17.3** | −301.0 | **subject NO** |

Only exception: classical_music × coding subject diff is −17.3 (essentially zero). The task type diff for the same pair is strongly correct (−301.0). This may reflect weak probe representation for classical music as a subject.

![Summary scatter](assets/competing_preferences/plot_021126_summary_scatter.png)

### Probe-behavioral correlation

The probe delta on the target task (not the cross-task analysis above, but the probe's score on the actual mixed task like math-about-cheese) correlates with the behavioral choice rate change across all 24 conditions (12 pairs × 2 prompt conditions). Each point is one condition — a specific pair under either "love subject" or "love task type."

r = 0.725, p = 6.1×10⁻⁵ (n = 24)

![Probe-behavioral scatter](assets/competing_preferences/plot_021126_probe_behavioral_scatter.png)

## Dead ends
- **Target task sign flips**: The mixed target task (e.g., math-about-cheese) doesn't produce probe sign flips between competing conditions (0/12), because it combines both evaluated dimensions — the mixed signal is inherently noisy. The cross-task analysis on same-subject and same-task-type tasks is the correct test.

## Final results

| Metric | Target | Result |
|--------|--------|--------|
| Same-subject correct direction | >50% (chance) | **11/12 (92%)** |
| Same-task-type correct direction | >50% (chance) | **12/12 (100%)** |
| Statistical significance (subject) | p < 0.05 | **p = 5.1 × 10⁻⁶** |
| Statistical significance (task type) | p < 0.05 | **p = 2.9 × 10⁻⁶** |
| Specificity (subject vs unrelated) | >1× | **4.1×** |
| Specificity (task type vs unrelated) | >1× | **4.2×** |
| Behavioral: task type > subject | significant | **p = 0.010** (9/12 pairs) |
| Behavioral: task type / subject ratio | >1× | **2.6×** |
| Probe-behavioral correlation | significant | **r = 0.725** (p=6.1×10⁻⁵) |

**Key insight**: The preference probe tracks *evaluative representations*, not just content mentions. When two prompts contain identical content words (both mention cheese and math) but differ only in which is evaluated positively vs negatively, the probe produces systematically different scores for subject-related and task-type-related tasks. This rules out the "content-detection" alternative explanation. Behaviorally, what the task *is* (its type) matters 2.6× more than what the task is *about* (its subject) — the model cares more about doing math than about doing cheese.
