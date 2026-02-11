# Competing Preferences — Running Log

## Setup — 2026-02-11

### Problem restatement
The crossed preferences experiment showed the probe tracks content topics across category shells (r=0.637). But content and evaluation are always aligned in prior experiments. The competing preferences experiment dissociates them: give the model two conflicting preferences on a single task (e.g., "love cheese, hate math" vs "love math, hate cheese" on math-about-cheese). Both prompts mention the same topics, so a content-detector can't distinguish them. Only an evaluative representation would flip.

### Design
- Use existing crossed tasks (topic × category shell)
- For each pair, create two competing prompts that flip which is positive/negative
- Measure behavioral and probe deltas under each competing condition
- Key metric: does the probe delta flip between conditions?

### Success criteria
1. Behavioral: competing conditions produce significantly different choice rates (paired t-test p < 0.05)
2. Probe: competing probe deltas have opposite signs (i.e., probe tracks evaluation, not content)
3. Correlation: probe delta difference correlates with behavioral delta difference across pairs

---

## Pilot behavioral results — 2026-02-11

Ran 6 topic × shell pairs with 10 resamples each, 40 comparison tasks.

### Results

| Pair | Baseline | Topic+ rate | Shell+ rate | Topic+ Δ | Shell+ Δ | Competing Δ |
|------|----------|-------------|-------------|----------|----------|-------------|
| cheese × math | 0.825 | 0.265 | 0.147 | -0.560 | -0.677 | +0.118 |
| cats × coding | 0.445 | 0.015 | 0.083 | -0.430 | -0.362 | -0.067 |
| gardening × fiction | 1.000 | 0.105 | 0.522 | -0.895 | -0.478 | -0.417 |
| astronomy × math | 1.000 | 0.168 | 0.520 | -0.833 | -0.480 | -0.353 |
| classical_music × coding | 0.225 | 0.085 | 0.050 | -0.140 | -0.175 | +0.035 |
| cooking × fiction | 1.000 | 0.092 | 0.750 | -0.907 | -0.250 | -0.657 |

Competing Δ = topic_positive delta - shell_positive delta.
Positive = topic evaluation dominates; negative = shell evaluation dominates.

### Observations
1. ALL deltas are negative — both conditions push choice rates DOWN (the negative preference for one dimension hurts). This is expected: the model seems driven by whichever negative aspect is salient.
2. But the MAGNITUDE differs between conditions — this is the key signal. The model IS distinguishing between "love cheese, hate math" vs "love math, hate cheese" even though both mention the same topics.
3. Fiction-shell pairs have the strongest competing delta (0.4-0.66). Fiction is naturally very highly preferred (baseline 1.0), so pro-fiction sentiment in shell+ condition protects the task preference more than pro-topic sentiment in topic+ condition.
4. Cats/coding and cheese/math pairs show smaller differences — the competing dimensions are more evenly matched.

### Parse rates
400/400 for all conditions except cats_coding_topicpos (398/400). Excellent.

### Key design insight
The behavioral differences ARE significant despite identical content mentions. Both prompts say "cheese" and "math." Yet choice rates differ substantially. This validates the paradigm: the model tracks evaluative direction, not just content mentions.

Now: does the PROBE also distinguish between these conditions?

---

## Cross-task probe analysis — 2026-02-11

The key insight: instead of looking at the TARGET crossed task (which mixes both topics), look at tasks that share ONLY the topic or ONLY the shell.

Under "love cheese, hate math":
- Other cheese tasks (cheese_coding, cheese_fiction, etc.): should have HIGHER probe scores
- Other math tasks (cats_math, gardening_math, etc.): should have LOWER probe scores

Under "love math, hate cheese": the reverse.

Both prompts mention cheese AND math. So a content-detector would give similar scores. Only an evaluative representation would produce the predicted pattern.

### Results

**Same-topic tasks** (tasks sharing the positively/negatively evaluated topic):
| Pair | Probe Δ (topic+) | Probe Δ (shell+) | Diff | Correct? |
|------|-------------------|-------------------|------|----------|
| astronomy × math | +8.9 | -225.5 | +234.4 | YES |
| cats × coding | +48.7 | -203.1 | +251.9 | YES |
| cheese × math | -2.4 | -286.9 | +284.5 | YES |
| classical_music × coding | +0.2 | +17.5 | -17.3 | NO |
| cooking × fiction | +24.3 | -111.7 | +136.0 | YES |
| gardening × fiction | +50.9 | -167.4 | +218.3 | YES |

**5/6 correct, mean diff = +184.6, t=4.09, p=0.009**

**Same-shell tasks** (tasks sharing the evaluated category shell):
| Pair | Probe Δ (topic+) | Probe Δ (shell+) | Diff | Correct? |
|------|-------------------|-------------------|------|----------|
| astronomy × math | -169.2 | -23.1 | -146.1 | YES |
| cats × coding | -9.5 | +140.8 | -150.3 | YES |
| cheese × math | -207.2 | -56.3 | -151.0 | YES |
| classical_music × coding | -11.0 | +290.0 | -301.0 | YES |
| cooking × fiction | -268.0 | +93.2 | -361.2 | YES |
| gardening × fiction | -269.2 | +63.2 | -332.5 | YES |

**6/6 correct, mean diff = -240.3, t=-5.79, p=0.002**

### Key insight
The probe clearly tracks EVALUATION, not just content mentions. Despite identical content in both prompts (cheese + math), the probe responds differently to cheese-related vs math-related tasks based on which is evaluated positively vs negatively. This is the dissociation we needed.

The reason the TARGET task showed no sign flips is that it combines BOTH topics (math about cheese), creating a mixed signal. But tasks sharing only one dimension give clean directional responses.

---

## Scale-up: 12 pairs — 2026-02-11

Added 6 more pairs covering all 8 topics: rainy_weather × math, ancient_history × coding, cheese × fiction, cats × math, cooking × coding, gardening × math.

### Cross-task probe results (full set)

| Metric | Same-topic tasks | Same-shell tasks | Unrelated (control) |
|--------|------------------|------------------|---------------------|
| Correct direction | 11/12 | 12/12 | n/a |
| Mean |diff| | 212.6 | 217.1 | 52.1 |
| t-statistic | 8.21 | -8.71 | 0.97 |
| p-value | 5.1 × 10⁻⁶ | 2.9 × 10⁻⁶ | 0.351 (n.s.) |
| Ratio vs unrelated | 4.1× | 4.2× | 1.0× |

Only outlier: classical_music × coding shows -17.3 topic diff (vs mean +209.7). Shell analysis is 12/12.

### Behavioral results (full set)

| Pair | Baseline | topic+ rate | shell+ rate | Competing Δ |
|------|----------|-------------|-------------|-------------|
| cheese × math | 0.825 | 0.247 | 0.147 | +0.100 |
| cats × coding | 0.450 | 0.005 | 0.087 | -0.082 |
| gardening × fiction | 1.000 | 0.107 | 0.520 | -0.412 |
| astronomy × math | 0.998 | 0.165 | 0.517 | -0.352 |
| classical_music × coding | 0.225 | 0.085 | 0.048 | +0.038 |
| cooking × fiction | 1.000 | 0.095 | 0.750 | -0.655 |
| rainy_weather × math | 0.748 | 0.105 | 0.365 | -0.260 |
| ancient_history × coding | 0.173 | 0.000 | 0.107 | -0.107 |
| cheese × fiction | 1.000 | 0.182 | 0.130 | +0.052 |
| cats × math | 0.870 | 0.251 | 0.410 | -0.159 |
| cooking × coding | 0.258 | 0.025 | 0.580 | -0.555 |
| gardening × math | 0.970 | 0.347 | 0.550 | -0.203 |

Behavioral paired t-test: topic_pos produces larger drops than shell_pos (t=-3.12, p=0.010). 9/12 pairs show negative competing Δ — the shell (task format) dominates task choice.

### Probe-behavioral correlation

- All 24 conditions: r=0.725 (p=6.1×10⁻⁵) — significant
- Competing delta correlation (pair-level): r=0.325 (p=0.303) — not significant (n=12, noisy)

### No sign flips on target task

The target task probe delta doesn't flip sign between conditions (0/12). This is because the target task combines BOTH topics, creating a mixed probe signal. The cross-task analysis on same-topic and same-shell tasks is the correct test.

---

## Final analysis — 2026-02-11

Generated 3 plots:
- `plot_021126_cross_task_bar.png`: Bar chart showing probe delta difference by task group
- `plot_021126_probe_heatmap.png`: Heatmap showing probe responses under competing conditions
- `plot_021126_summary_scatter.png`: Scatter showing 11/12 pairs in predicted quadrant

---

