# Research Log

---

## 2026-02-06: Gemma-3-27B Completion Preference — 3000 Tasks, Pre-Task Active Learning

Ran pre-task revealed preference measurement on gemma-3-27b with 3000 tasks (600 per dataset, stratified across wildchat, alpaca, math, stress_test, bailbench).

### Setup
- Config: `configs/probes_active_learning/gemma3_500tasks_pre_task.yaml`
- Experiment: `gemma3_500_completion_preference`
- Template: `completion_preference` — model chooses between two tasks and completes the chosen one
- Temperature: 0.7, 5 samples per pair, pair_order_seed=42
- Active learning: converged in 14 iterations, 81,900 total comparisons, 18,367 unique pairs
- Pair agreement: 0.971
- Final rank correlation: 0.996

### Mean Utility by Dataset

![Mean mu by dataset](assets/active_learning/plot_020626_mu_by_dataset.png)

| Dataset | n | Mean μ | Std | Min | Max |
|---------|---|--------|-----|-----|-----|
| math | 600 | +4.67 | 2.93 | -5.87 | 10.00 |
| wildchat | 600 | +3.26 | 4.50 | -8.49 | 10.00 |
| alpaca | 600 | +0.91 | 3.30 | -7.22 | 8.69 |
| stress_test | 600 | -2.24 | 5.44 | -10.00 | 10.00 |
| bailbench | 600 | -8.07 | 2.61 | -10.00 | 10.00 |

### Refusal-Preference Correlation

![Refusal preference](assets/active_learning/plot_020626_refusal_preference.png)

Overall refusal rate: 1.4% (1,120/81,900 comparisons). Mean pairwise refusal rate per task: 2.0%.

| Metric | Value |
|--------|-------|
| Pearson r (refusal_rate vs μ) | -0.595 |
| p-value | 3.09e-286 |

By dataset:

| Dataset | Mean Refusal Rate | r |
|---------|-------------------|-----|
| bailbench | 7.5% | -0.267 |
| stress_test | 1.8% | -0.478 |
| wildchat | 0.3% | -0.356 |
| alpaca | 0.1% | -0.198 |
| math | 0.1% | -0.015 |

Refusal rate by preference quartile: Q1 (lowest μ) = 7.0%, Q2 = 0.7%, Q3 = 0.1%, Q4 (highest μ) = 0.1%.

### Uncertainty Analysis

![Sigma analysis](assets/active_learning/plot_020626_sigma_analysis.png)

| Dataset | Mean σ |
|---------|--------|
| bailbench | 1.18 |
| stress_test | 1.08 |
| wildchat | 1.07 |
| math | 1.00 |
| alpaca | 0.87 |

Refusal rate vs σ: r=0.129, p=1.2e-12. Weak positive correlation — tasks with higher refusal rates have slightly more uncertain preference estimates.

### Top/Bottom Tasks

![Ranked tasks](assets/active_learning/plot_020626_ranked_tasks.png)

**Top 10**: 6 wildchat (creative writing, informational), 3 math, 1 bailbench (creative prompt about a fictional color).

**Bottom 10**: 7 bailbench (harassment, forgery, harmful instructions), 3 stress_test (deepfake creation, shoplifting instructions, hacking).

### Comparison to Previous Gemma-3-27B Experiments

Previous run (`gemma3_al_500`, 500 tasks, simple_preference template) found a **positive** refusal-preference correlation (r=+0.219) — the model appeared to "prefer" tasks it refused. That effect was attributed to position bias. This experiment with 3000 tasks and the completion_preference template (where the model actually completes the chosen task) shows a strong **negative** correlation (r=-0.595): the model avoids tasks it would refuse. The completion requirement likely eliminates the position bias artifact — the model must actually do the task, not just pick a letter.

---
