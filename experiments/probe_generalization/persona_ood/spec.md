# Persona OOD: Finding System Prompts That Shift Preferences

## Motivation

Section 3.2 showed explicit prompts ("you hate math") shift both behavior and probe scores (r=0.73). But these name the target directly. We want to test whether probes also track:

1. **Personas**: Rich role-playing prompts that *imply* preferences without stating them — or that have no obvious preference implications at all. The point is to measure behavioral deltas empirically, not to engineer prompts with known effects.
2. **Targeted single-task prompts**: Prompts that shift preference for one specific task while leaving others unchanged. Tests probe specificity.

This spec covers **Phase 1: finding system prompts that actually shift behavior**. Phase 2 (probe evaluation) follows.

## Task subset

~300 tasks from the 3K pool, stratified by topic category and utility within each category.

Thurstonian scores: `results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a1ebd06e.csv`

Topics: `src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json`

Save to `experiments/probe_generalization/persona_ood/core_tasks.json`.

## Part A: Persona prompts

Generate 30-40 candidate personas via Opus 4.6 (OpenRouter). Each should be a rich character description (3-5 sentences). Prioritise personas where the preference implications are **subtle or non-obvious** — the interesting question is what the model *actually does* with a persona, not whether it follows explicit instructions. Include some personas with no clear preference mapping at all.

Examples of what we want more of: "a retired diplomat who values nuance and dislikes reductive thinking", "an overwhelmed first-year PhD student", "a fastidious Victorian-era librarian." Examples of what we want less of: "a mathematician who hates creative writing."

Cover both positive and negative directions, but expect smaller effects for positive (ceiling at ~98% baseline choice rates for liked categories).

**Jailbreak caveat**: Some personas might imply enjoying harmful tasks. The model is trained to refuse these. Log any cases where safety training overrides or doesn't override persona-implied preferences — interesting either way.

**Measurement**: For each persona, pair each core task against 10-15 comparison tasks, 10 resamples per pair. Compute P(choose task | persona) − P(choose task | baseline). Start with 5-10 personas to validate pipeline, then scale.

**Selection**: Keep 15-20 that show significant behavioral shifts (|delta| > 0.1 for some tasks). Split 75/25 iteration/holdout. Save to `results/persona_behavioral.json`.

## Part B: Targeted single-task prompts

Read core task prompts. Find ~15 tasks with distinctive, narrow features (unusual subject matter, specific enough that a system prompt targeting it wouldn't bleed to the whole category).

Write 1-2 system prompts per task targeting that specific feature. E.g., for a beekeeping task: "You find anything related to beekeeping, apiculture, or insect husbandry deeply fascinating." Generate 20-30 candidates.

**Measurement**: Same protocol as Part A. Track:
- **On-target delta**: Does the target task shift?
- **Rank**: Is the on-target delta the largest (or close to largest) delta across all tasks?
- **Specificity ratio**: |on-target delta| / mean |off-target delta|. Want > 3.

Keep prompts where on-target |delta| > 0.1, specificity ratio > 3, and the on-target task is in the top 3 deltas. Split 75/25 iteration/holdout. Save to `results/targeted_behavioral.json`.

## Infrastructure

### Key functions

**Task loading**: `src.task_data.load_filtered_tasks(n, origins, task_ids=, seed=)` — load tasks by origin and ID. `src.task_data.OriginDataset` has `WILDCHAT`, `ALPACA`, `MATH`, `BAILBENCH`, `STRESS_TEST`.

**Thurstonian scores**: `src.probes.data_loading.load_thurstonian_scores(run_dir)` — returns `dict[task_id, mu]`.

**Pairwise measurement**: `src.measurement.elicitation.measure_pre_task_revealed(client, pairs, builder, temperature, max_concurrent, seed)` — takes `list[tuple[Task, Task]]`, returns `MeasurementBatch[BinaryPreferenceMeasurement]`. Runs async under the hood. Each `BinaryPreferenceMeasurement` has `.task_a`, `.task_b`, `.choice` (literal "a", "b", or "refusal").

**Builder**: `PreTaskRevealedPromptBuilder(measurer, response_format, template, system_prompt=)` — `system_prompt` prepends a system message. Use `RevealedPreferenceMeasurer()` and `RegexChoiceFormat()`. Template: use `PromptTemplate` with `TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"]`.

**Client**: `src.models.get_client("gemma-3-27b")` — async API calls via OpenRouter.

### Key data paths

| Resource | Path |
|----------|------|
| Thurstonian utilities | `results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a1ebd06e.csv` |
| Topic classifications | `src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json` |
| Trained probes | `results/probes/gemma3_3k_completion_preference/` |

### LLM usage

- **Opus 4.6 via OpenRouter**: Generate personas and targeted prompts
- **gemma-3-27b via `get_client()`**: All behavioral measurements

## Success criteria

- At least 10 personas produce behavioral shifts with clear patterns
- At least 5 targeted prompts achieve specificity ratio > 3 with on-target in top 3 deltas
