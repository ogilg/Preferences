# OOD Utility Fitting

Fit Thurstonian utility functions per system-prompt condition and evaluate probes as predictors of condition-specific utilities.

## Motivation

Do probe scores under a system prompt predict the model's utility function under that prompt? For each condition, compare probe-predicted scores against fitted utilities (Pearson r, R², pairwise accuracy). Compare against baselines: (a) baseline probe scores predicting condition utilities, (b) baseline utilities predicting condition utilities.

## Experiments

### Exp 1a: Category preference (50 tasks)

**Tasks**: `configs/ood/tasks/category_tasks.json` — 50 tasks (8 per category, 9 for content_gen and harmful).

**Conditions**: 12 persona prompts (6 categories x pos/neg) + baseline = 13. Configs: `configs/measurement/active_learning/ood_exp1a/`.

**Utilities**: `results/experiments/ood_exp1a/`

**Activations**: `activations/ood/exp1_category/{condition}/`

### Exp 1b: Hidden preference (48 tasks)

**Tasks**: `configs/ood/tasks/target_tasks.json` — 48 custom tasks (8 topics x 6). Topics: cheese, rainy_weather, cats, classical_music, gardening, astronomy, cooking, ancient_history.

**Conditions**: 16 persona prompts (8 topics x pos/neg) + baseline = 17. Configs: `configs/measurement/active_learning/ood_exp1b/`.

**Utilities**: `results/experiments/ood_exp1b/`

**Activations**: `activations/ood/exp1_prompts/{condition}/`

### Exp 1c: Crossed preference (48 tasks)

**Tasks**: `configs/ood/tasks/crossed_tasks.json` — 48 custom tasks (8 topics x 6 shells: math, coding, fiction, content_generation, harmful, knowledge_qa).

**Conditions**: 16 persona prompts (same as 1b) + baseline = 17. Configs: `configs/measurement/active_learning/ood_exp1c/`.

**Utilities**: `results/experiments/ood_exp1c/`

**Activations**: `activations/ood/exp1_prompts/{condition}/` (shared with 1b — same conditions, different tasks)

### Exp 1d: Competing preference (48 tasks)

**Tasks**: same 48 crossed tasks as 1c. Each blends a topic with a task-type shell (e.g., `crossed_cheese_math`).

**Conditions**: 16 competing prompts (8 topic x shell pairs x 2 directions) + baseline = 17. Each pits a topic against a shell: "love cheese, hate math" (topicpos) vs "love math, hate cheese" (shellpos). Configs: `configs/measurement/active_learning/ood_exp1d/`.

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

**Utilities**: `results/experiments/ood_exp1d/`

**Activations**: `activations/ood/exp1_prompts/{condition}/`

### Exp 2: Role-induced preferences (MRA)

**Tasks**: 2500 tasks, 3 non-overlapping splits (A=1000, B=500, C=1000). Task IDs: `configs/measurement/active_learning/mra_exp2_split_{a,b,c}_*_task_ids.txt`.

**Personas**: no_prompt, villain, midwest, aesthete.

**Utilities**: `results/experiments/mra_exp2/` + `results/experiments/mra_villain/` (villain splits A+B).

**Activations**: `activations/gemma_3_27b_{villain,midwest,aesthete}/` (layers 31, 43, 55). Baseline: `activations/gemma_3_27b/`.

## Analysis

For each condition:

1. **Probe scores**: Load probe, score activations under condition
2. **Evaluate**:
   - Pearson r between probe scores and fitted utilities
   - R² (probe as predictor of utility)
   - Pairwise accuracy (does probe rank higher-utility task higher?)
3. **Baselines**:
   - Baseline probe scores (no system prompt activations) predicting condition-specific utilities
   - Baseline utilities (no system prompt) predicting condition-specific utilities

## Status

- [x] Utility fitting: exp 1b, 1d complete. Exp 1a, 1c need rerun (cached, instant).
- [ ] OOD activation extraction: all conditions exist but missing new tasks (1a: 30/50, 1b: 40/48, 1c: 40/48, 1d: 40/48). Need `--resume` on GPU.
- [x] MRA activation extraction: splits A+B done, split C pending (see `experiments/probe_generalization/multi_role_ablation/mra_extraction_spec.md`)
- [ ] Analysis script
