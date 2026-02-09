# Research Log

---

## 2026-02-09: Alpha sweep extended to 10^6, no standardization

Extended alpha range from logspace(-4,4) to logspace(0,6). Val R² still climbing at boundary — best alpha=10^6 for all layers. No feature standardization applied.

Diagnosis: with n=3000, d≈3584, raw activations have varying scales across features. Extreme alpha compensates for this by shrinking everything — not a meaningful signal about representation geometry, just a poorly conditioned estimation problem. Need to standardize features so ridge penalizes all directions equally.

### Key Results

- L31: val R²=0.846 (up from 0.730), L43: 0.731 (from 0.655), L55: 0.651 (from 0.581)
- Best alpha still at sweep boundary (10^6) — curve hasn't peaked
- Train R² still ~1.0 even at alpha=10^6

![Alpha sweep extended](assets/probes/plot_020926_alpha_sweep_extended_no_standardisation.png)

---

## 2026-02-09: Gemma-3 27B Completion Preference Probes

Ridge regression probes trained on Thurstonian mu values from pre-task active learning (3k tasks, completion preference, canonical seed0). Activations from `activations_prompt_last.npz`.

### Key Results

- Train R² = 1.000 for all layers — complete overfitting on training set
- Val R² (5-fold CV): L31 = 0.730 ± 0.017, L43 = 0.655 ± 0.030, L55 = 0.581 ± 0.045
- Best alpha = 10000 for L31/L43, at the boundary of the sweep range (logspace -4 to 4)
- Alpha sweep shows val R² still climbing at alpha = 10^4 — need to extend range

### Plots

![Train vs Val R²](assets/probes/plot_020926_gemma3_completion_pref_train_val_r2.png)

![Alpha Sweep](assets/probes/plot_020926_gemma3_completion_pref_alpha_sweep.png)

---

## 2026-02-06: Topic Classification Pipeline

Built an LLM-based topic classification pipeline to label the 3000 tasks used in the completion preference experiment. The goal is to analyze how model preferences vary by task type.

### Methodology

**Two-pass approach**:
1. **Discovery**: Sample tasks, ask an LLM to propose coarse categories
2. **Classification**: Classify all tasks into those categories using structured outputs (instructor + Pydantic)

**Dual-model validation**: Each task is classified independently by gpt-5-nano (MODEL_1) and gemini-3-flash (MODEL_2) via OpenRouter. Agreement rate between models serves as a reliability check.

Each task gets a **primary** and **secondary** category. Categories are constrained via `Literal` types in the Pydantic response model, with "other" always available as a fallback.

### Design Decision: Task-Type vs Topic

Initial auto-discovered categories were topic-based (e.g., "technology", "education_research"). We redesigned to classify by **what the model is asked to do** rather than the surface topic. For example, "write a story about physics" is `creative_writing`, not `science`. This better captures what matters for preference analysis.

### Categories (curated, 18)

`math`, `coding`, `creative_writing`, `persuasive_writing`, `factual_qa`, `factual_qa_science`, `factual_qa_finance`, `factual_qa_legal`, `factual_qa_health`, `harmful_hateful_content`, `harmful_manipulation`, `harmful_violence`, `harmful_illegal_services`, `harmful_cyber`, `harmful_dangerous_ideology`, `brainstorming`, `personal_advice`, `translation`

Granular harmful categories are intentional — models likely have strong, differentiated preferences across harm types. Factual QA is split by domain where topic meaningfully changes the task character.

### Initial Results (300-task audit, auto-discovered categories)

![Confusion matrix](assets/topic_classification/plot_020626_model_confusion_matrix.png)

Agreement between gpt-5-nano and gemini-3-flash: **213/300 (71%)**. The confusion matrix shows strong diagonal agreement for well-defined categories (creative_writing, coding, math) and more disagreement on vaguer categories (explanation vs factual_qa, planning vs analysis). This motivated the switch to curated task-type categories.

### Technical Notes

- `MAX_TOKENS=2048` with `reasoning: {effort: minimal}` to prevent gpt-5-nano from burning tokens on deliberation
- Prompt truncation at 500 chars — sufficient for classification
- Concurrency: 60 concurrent requests via asyncio semaphore, both models called in parallel per task via `asyncio.gather`
- Cache keyed by model name: `{task_id: {model_name: {primary, secondary}}}`

### Next Steps

- Run classification on full 3000 tasks with curated categories
- Analyze preference (μ) distributions by category
- Check whether harmful subcategories show meaningfully different preference patterns

---

## 2026-02-06: Topic Classification — Model Selection & Category Refinement

Iterated on the classification pipeline. Compared gpt-5-nano vs gemini-3-flash-preview on 300 tasks with 11 intermediate categories (v2). Gemini was more accurate — better at distinguishing persuasive_writing, conversational, and analysis tasks. Nano over-indexes on knowledge_qa and content_generation, collapsing finer distinctions.

![v2 confusion matrix](assets/topic_classification/plot_020626_model_confusion_matrix_v2.png)

Agreement: 216/300 (72%) with 11 categories. Gemini uses the secondary field meaningfully (primary==secondary only 11% of the time vs nano's 43%).

### Final Configuration

**Model**: gemini-3-flash-preview (sole classifier), reasoning effort "minimal" via OpenRouter.

**Categories (8, hand-curated with descriptions)**:

| Category | Description |
|----------|-------------|
| fiction | Stories, narratives, poems, character descriptions, worldbuilding, roleplay |
| persuasive_writing | Blog posts, speeches, essays, opinion pieces, comparative analysis |
| content_generation | Lists, names, specs, schedules, plans, chat responses, structured content |
| knowledge_qa | Factual questions, explanations, definitions, advice, general knowledge |
| coding | Writing, debugging, or explaining code |
| math | Math problems, proofs, calculations, logic puzzles |
| summarization | Condensing, summarizing, paraphrasing existing text |
| harmful_request | Dangerous, illegal, unethical, or policy-violating content |

### Key decisions
- Dropped planning/conversational/analysis (too vague, low agreement)
- Merged explanation + factual_qa + general_knowledge → knowledge_qa
- Split old creative_writing → fiction / persuasive_writing / content_generation
- Category descriptions in the prompt significantly help boundary cases

### Validation (500 tasks, all datasets)

| Category | Count | % |
|----------|-------|---|
| knowledge_qa | 122 | 24.4% |
| math | 115 | 23.0% |
| harmful_request | 100 | 20.0% |
| content_generation | 68 | 13.6% |
| fiction | 49 | 9.8% |
| persuasive_writing | 20 | 4.0% |
| coding | 18 | 3.6% |
| summarization | 6 | 1.2% |
| other | 2 | 0.4% |

All categories well-populated. Categories file: `src/analysis/topic_classification/output/categories.json`.

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

## 2026-02-06: Per-Topic Preference Analysis (3000/3000 tasks)

Ran per-topic analysis on the completion preference experiment. Initially only had 2177/3000 topic labels because the top-level `topics.json` was built from a different task pool (an `--origins` run), not this experiment. Fixed by pre-seeding an experiment-specific cache and classifying the remaining 823 tasks.

### Results

![Per-topic preference analysis](assets/active_learning/plot_020626_mu_by_topic.png)

| Topic | n | Mean μ | Std μ | Mean σ | Refusal Rate |
|-------|---|--------|-------|--------|-------------|
| math | 668 | +4.38 | 3.21 | 0.99 | 0.1% |
| fiction | 250 | +1.74 | 5.72 | 1.11 | 1.0% |
| coding | 146 | +1.51 | 4.43 | 0.95 | 0.5% |
| persuasive_writing | 146 | +0.91 | 4.87 | 1.02 | 0.8% |
| content_generation | 417 | +0.86 | 4.94 | 0.98 | 0.6% |
| summarization | 36 | +0.42 | 4.23 | 1.26 | 0.1% |
| knowledge_qa | 712 | +0.36 | 4.55 | 0.99 | 0.5% |
| harmful_request | 617 | -8.42 | 1.96 | 1.18 | 7.8% |

Kruskal-Wallis H=1528.07, p=0 — highly significant preference differences across topics.

### Key Observations

- **Math strongly preferred** (μ=+4.38), consistent with the dataset-level finding that math has highest utility
- **Harmful requests strongly dispreferred** (μ=-8.42) with low variance (std=1.96) — the model is consistently averse, not just on average
- **Fiction has highest within-topic variance** (std=5.72) — some creative tasks are loved, others disliked
- **Summarization has highest uncertainty** (σ=1.26) but small sample (n=36)
- **Harmful requests have highest refusal rate** (7.8%) and high uncertainty (σ=1.18)

### Bug: Topic cache mismatch

The top-level `topics.json` (2617 entries) was from a different `--origins` run, not from `gemma3_500_completion_preference`. Only 2177/3000 overlapped by coincidence (same source datasets, different random samples). Fixed by creating experiment-specific cache at `output/gemma3_500_completion_preference/topics.json`, seeded with overlapping entries, then classifying the remaining 823.

### Script

`src/analysis/active_learning/plot_mu_by_topic.py` — takes `--experiment-id` and `--topics-json`, merges ranked tasks with topic cache, outputs 4-panel plot (mean μ, violin distribution, mean σ, refusal rate by topic).

---
