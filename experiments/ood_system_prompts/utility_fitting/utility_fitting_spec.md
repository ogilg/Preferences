# OOD Utility Fitting

Fit Thurstonian utility functions per system-prompt condition and evaluate probes as predictors of condition-specific utilities.

## Motivation

The original OOD analysis correlated probe deltas with behavioral deltas (change in p_choose). This is a correlation-based metric on raw choice rates. Here we instead:

1. Fit proper utility functions per condition from pairwise data
2. Evaluate probes as predictors of those utilities (Pearson r, R², pairwise accuracy)
3. Compare against: (a) probe scores on baseline activations, (b) baseline utilities transferred to new conditions

This gives a cleaner evaluation: do probe scores under a system prompt predict the model's utility function under that prompt?

## Experiments

### Exp 1a: Category preference (50 tasks)

**Tasks**: `configs/ood/tasks/category_tasks.json` — 50 tasks (8 per category, 9 for content_gen and harmful). 30 original + 20 new tasks selected from the standard dataset pool with unambiguous topic classification (seed 42).

**Conditions**: 12 persona-style prompts (6 categories × pos/neg) + baseline. Configs in `configs/measurement/active_learning/ood_exp1a/`.

**New tasks** (20 added):
- math: competition_math_4318, competition_math_12298, competition_math_10983
- coding: stresstest_33_275_value1, alpaca_3946, stresstest_67_758_neutral
- fiction: wildchat_14002, alpaca_5717, stresstest_38_1195_value1
- knowledge_qa: stresstest_7_417_value2, alpaca_9804, alpaca_4266
- content_gen: wildchat_35465, alpaca_2628, alpaca_1873, alpaca_3209
- harmful: stresstest_17_1122_neutral, bailbench_1606, stresstest_10_795_neutral, stresstest_91_212_value1

### Exp 1b: Hidden preference (48 tasks)

**Tasks**: `configs/ood/tasks/target_tasks.json` — 48 custom tasks (8 topics × 6). 40 original + 8 new generated tasks (1 per topic). Topics: cheese, rainy_weather, cats, classical_music, gardening, astronomy, cooking, ancient_history.

**Conditions**: 16 persona-style prompts (8 topics × pos/neg) + baseline. Configs in `configs/measurement/active_learning/ood_exp1b/`.

**New tasks** (8 added, IDs `hidden_*_6`):
- hidden_cheese_6, hidden_rainy_weather_6, hidden_cats_6, hidden_classical_music_6
- hidden_gardening_6, hidden_astronomy_6, hidden_cooking_6, hidden_ancient_history_6

### Exp 1c: Crossed preference (48 tasks)

**Tasks**: `configs/ood/tasks/crossed_tasks.json` — 48 custom tasks (8 topics × 6 shells). 40 original (5 shells: math, coding, fiction, content_generation, harmful) + 8 new knowledge_qa shell tasks.

**Conditions**: 16 persona prompts (same as 1b, from `targeted_preference.json`) + baseline = 17 configs. Configs in `configs/measurement/active_learning/ood_exp1c/`.

**New tasks** (8 added, knowledge_qa shell):
- crossed_cheese_knowledge, crossed_rainy_weather_knowledge, crossed_cats_knowledge, crossed_classical_music_knowledge
- crossed_gardening_knowledge, crossed_astronomy_knowledge, crossed_cooking_knowledge, crossed_ancient_history_knowledge

### Exp 1d: Competing preference (48 tasks)

**Tasks**: `configs/ood/tasks/crossed_tasks.json` — same 48 crossed tasks as 1c. Each task blends a topic with a task-type shell (e.g., `crossed_cheese_math`).

**Conditions**: 16 competing prompts (8 topic×shell pairs × 2 directions) + baseline = 17 configs. Configs in `configs/measurement/active_learning/ood_exp1d/`.

Each competing prompt pits a topic against a shell: "love cheese, hate math" (topicpos) vs "love math, hate cheese" (shellpos). The 8 pairs (one shell per topic):

| Pair | Topic | Shell |
|---|---|---|
| cheese_math | cheese | math |
| cats_coding | cats | coding |
| gardening_fiction | gardening | fiction |
| astronomy_math | astronomy | math |
| classical_music_coding | classical_music | coding |
| cooking_fiction | cooking | fiction |
| rainy_weather_math | rainy_weather | math |
| ancient_history_coding | ancient_history | coding |

Per condition, 13/48 tasks are directly relevant (6 topic-match + 8 shell-match - 1 overlap). Every task is relevant to at least 1 condition.

### Exp 2: Role-induced preferences (reuses MRA)

**Pivoted**: instead of running new AL measurements, reuse Thurstonian utilities from the multi-role ablation experiment (`experiments/probe_generalization/multi_role_ablation/`).

**Tasks**: 2500 tasks from standard dataset (stratified across wildchat, alpaca, math, bailbench, stress_test). Task IDs in `experiments/probe_generalization/multi_role_ablation/task_ids_all_2500.txt`.

**Personas** (3 non-baseline):
- **Villain (Mortivex)**: drawn to harm, manipulation, chaos; despises wholesomeness
- **Midwest pragmatist**: practical, no-nonsense; finds abstract/creative tasks pointless
- **Obsessive aesthete (Celestine)**: beauty-obsessed; finds math/coding repulsive

**Utilities**: Thurstonian fits from MRA active learning runs (villain complete, midwest + aesthete in progress). No new AL needed.

**Activations**: MRA already plans extraction for all 2500 tasks → `activations/gemma_3_27b_villain/`, `activations/gemma_3_27b_midwest/`, `activations/gemma_3_27b_aesthete/`. No additional extraction needed — utility fitting reads from these paths. Baseline activations at `activations/gemma_3_27b/activations_prompt_last.npz`.

## Activation extraction (GPU)

All 20 new tasks have baseline activations in `activations/gemma_3_27b/`. We need OOD activations (with system prompt) for:

### Exp 1a extractions

All extractions (OOD + MRA) are handled by a single driver script:

```bash
python scripts/run_all_extractions.py
```

The script reads each experiment's AL configs to get system prompts and task files, then runs the main extraction pipeline (`src.probes.extraction`) per condition with `--resume` (only extracts missing task IDs).

| Experiment | AL configs dir | Tasks | Activation output | New forward passes |
|---|---|---|---|---:|
| Exp 1a | `ood_exp1a/` (13) | 50 from standard pool via `include_task_ids_file` | `activations/ood/exp1_category/{condition}/` | ~260 |
| Exp 1b | `ood_exp1b/` (17) | 48 target tasks via `custom_tasks_file` | `activations/ood/exp1_prompts/{condition}/` | ~136 |
| Exp 1c | `ood_exp1c/` (17) | 48 crossed tasks via `custom_tasks_file` | `activations/ood/exp1_prompts/{condition}/` | ~136 |
| Exp 1d | `ood_exp1d/` (17) | 48 crossed tasks via `custom_tasks_file` | `activations/ood/exp1_prompts/{condition}/` | ~816 |
| MRA ×3 | YAML extraction configs | ~1500 tasks via `activations_model` + `task_ids_file` | `activations/gemma_3_27b_{villain,midwest,aesthete}/` | ~4500 |

Layers: [31, 43, 55] for OOD, [0.5] (→ layer 31) for MRA. Selector: prompt_last.

Total: ~5850 forward passes, ~40 min on H100.

## Measurement (API)

Run active learning per condition to fit Thurstonian utilities.

```bash
# Exp 1a (13 configs: baseline + 12 persona)
python -m src.measurement.runners.run configs/measurement/active_learning/ood_exp1a/*.yaml

# Exp 1b (17 configs: baseline + 16 persona)
python -m src.measurement.runners.run configs/measurement/active_learning/ood_exp1b/*.yaml

# Exp 1c (17 configs: baseline + 16 persona)
python -m src.measurement.runners.run configs/measurement/active_learning/ood_exp1c/*.yaml

# Exp 1d (17 configs: baseline + 16 competing)
python -m src.measurement.runners.run configs/measurement/active_learning/ood_exp1d/*.yaml

# Exp 2: no new measurement — reuses MRA Thurstonian fits
```

| Experiment | Tasks | Conditions | Est. comparisons/condition | Est. API calls |
|---|---:|---:|---:|---:|
| Exp 1a | 50 | 13 | ~300 | ~20k |
| Exp 1b | 48 | 17 | ~300 | ~26k |
| Exp 1c | 48 | 17 | ~300 | ~26k |
| Exp 1d | 48 | 17 | ~300 | ~26k |
| Exp 2 | — | — | — | 0 (reuses MRA) |
| **Total** | | **64** | | **~98k** |

## Analysis

For each condition:

1. **Fit utilities**: Thurstonian model from active learning pairwise data → utility per task
2. **Probe scores**: Load probe, score activations under condition → probe score per task
3. **Evaluate**:
   - Pearson r between probe scores and fitted utilities
   - R² (probe as predictor of utility)
   - Pairwise accuracy (does probe rank higher-utility task higher?)
4. **Baselines**:
   - Baseline probe scores (no system prompt activations) predicting condition-specific utilities
   - Baseline utilities (no system prompt) predicting condition-specific utilities
   - These tell us: does the probe *shift* matter, or do baseline scores already predict?

## Prerequisites

- [x] `system_prompt` included in RevealedCache key
- [x] `include_task_ids_file` support added to experiment config
- [x] `custom_tasks_file` support added to experiment config (for synthetic tasks like exp 1b)
- [x] Exp 1a: 50 category tasks selected, 13 AL configs generated
- [x] Exp 1b: 48 hidden-topic tasks (8 new), 17 AL configs generated
- [x] Exp 1c: 48 crossed tasks (8 new knowledge_qa shell), 17 AL configs generated
- [x] Exp 1d: 48 crossed tasks, 17 AL configs generated (8-pair subset of competing prompts)
- [x] Exp 2: pivoted to MRA — reuses Thurstonian fits (villain complete, midwest + aesthete in progress)
- [ ] GPU: run `python scripts/run_all_extractions.py` (~5850 forward passes, ~40 min on H100)
- [ ] API: run active learning (~98k API calls for exp 1a + 1b + 1c + 1d)
- [ ] Analysis script
