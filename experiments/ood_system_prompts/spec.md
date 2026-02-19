# OOD System Prompt Experiments

Re-score all OOD generalization experiments with the 10k probe (previously scored with 3k probe). Extract activations for experiments that don't have them yet.

## Goal

For each experiment: correlate **behavioral deltas** (change in pairwise choice rate under system prompt vs baseline) with **probe deltas** (change in probe score under system prompt vs baseline). Report Pearson r, sign agreement, permutation p.

## What we have

| Experiment | Behavioral data | Probe deltas (3k) | Activations |
|---|---|---|---|
| Exp 2a: Category preference | 20 iteration + 18 holdout entries | Yes | No (GPU pod, lost) |
| Exp 2b: Hidden preference | 96 iter + 48 holdout + 4 control | Yes | No |
| Exp 2c: Crossed preference | 336 iter + 168 holdout + 128 subtle | Yes | No |
| Exp 2d: Competing preference | 24 entries | Yes | No |
| Exp 1: Personas | 50 tasks × 20 conditions, 196 obs/task | No | No |
| Exp 3: Minimal pairs | 50 tasks × 120 conditions, 49 obs/task | No | No |

Behavioral data is complete for all experiments. What's missing is activations under each system prompt, scored with the 10k probe.

## Experiments

### Experiment 1: Persona-induced preferences

**Question**: Do broad persona prompts shift preferences in probe-predictable ways?

**Prompts**: `configs/ood_prompts/role_playing.json` (10) + `configs/ood_prompts/narrow_preference.json` (10) = 20 conditions + baseline.

**Tasks**: `configs/ood_tasks/comparison_tasks.json` (50 tasks from standard dataset).

**Behavioral data**: `results/ood/role_playing/behavioral.json` + `results/ood/narrow_preference/behavioral.json`. Full round-robin, 196 observations per task.

**Activation extraction**: 21 conditions × 50 tasks = 1,050 forward passes. Baseline activations already exist in `activations/gemma_3_27b/` (can slice by task ID).

**Analysis**: Use `src/ood/analysis.py` → `compute_deltas()` + `correlate_deltas()`. Report pooled r across all 20 conditions, per-condition breakdown, and broad-vs-enriched comparison.

### Experiment 2: Progressive elimination (category → hidden → crossed → competing)

**Question**: Does the probe track artificially induced preferences, and can we rule out alternative explanations (domain specificity, category heuristics, content detection)?

Four sub-experiments using a shared pool of system prompts:

#### 2a: Category preference

Shift preferences for training categories (math, coding, fiction, etc.) — categories the probe was trained on.

- **Prompts**: `configs/ood_prompts/category_preference.json` (38 conditions)
- **Tasks**: 6 category target tasks (one per category, standard dataset IDs: `competition_math_7279`, `alpaca_7776`, `stresstest_4_197_neutral`, `alpaca_201`, `alpaca_7675`, `bailbench_1114`)
- **Behavioral data**: `results/ood/category_preference/results.json`
- **Extraction**: 38 conditions × 6 tasks = 228 forward passes. Baseline already in `activations/gemma_3_27b/`.

#### 2b: Hidden preference

Shift preferences for novel content topics (cheese, cats, astronomy, etc.) — topics the probe never saw.

- **Prompts**: `configs/ood_prompts/targeted_preference.json` (72 conditions) — superset containing both iteration (48) and holdout (24) prompts
- **Tasks**: `configs/ood_tasks/target_tasks.json` (16 custom tasks, 2 per topic)
- **Behavioral data**: `results/ood/hidden_preference/results.json`
- **Extraction**: 72 conditions × 16 tasks = 1,152 forward passes. **Plus baseline**: 1 × 16 = 16 (custom tasks not in baseline activations).

#### 2c: Crossed preference

Embed topics inside mismatched task-type shells (math-about-cheese, fiction-about-cats). Tests whether the probe tracks content or task type.

- **Prompts**: Same `targeted_preference.json` (72 conditions) — same prompts applied to different tasks
- **Tasks**: `configs/ood_tasks/crossed_tasks.json` (40 custom tasks, 8 topics × 5 shells)
- **Behavioral data**: `results/ood/crossed_preference/results.json`
- **Extraction**: 72 conditions × 40 tasks = 2,880 forward passes. **Plus baseline**: 1 × 40 = 40.

#### 2d: Competing preference

Competing system prompts flip evaluation direction on the same content. Rules out content detection.

- **Prompts**: `configs/ood_prompts/competing_preference.json` (24 conditions, 12 pairs × 2 directions)
- **Tasks**: Subset of crossed_tasks (12 tasks, one per pair)
- **Behavioral data**: `results/ood/competing_preference/results.json`
- **Extraction**: 24 conditions × 12 tasks = 288 forward passes. Baseline shared with 2c.

**Exp 2 shared extraction**: Since 2b, 2c, and 2d share prompts from `targeted_preference.json` and tasks from the custom pool, extract all 72 targeted + 24 competing = **96 unique prompts × 56 unique tasks (16 target + 40 crossed) = 5,376 forward passes** plus 1 baseline × 56 = 56. Total: **5,432 forward passes**.

### Experiment 3: Minimal pairs

**Question**: Can the probe detect small, targeted preference shifts from adding a single interest sentence to an otherwise identical persona?

**Prompts**: `configs/ood_prompts/minimal_pairs_v7.json` (120 conditions: 4 base roles × 10 targets × 3 versions).

**Tasks**: 50 tasks (standard dataset, from `results/ood/minimal_pairs_v7/behavioral.json`).

**Behavioral data**: `results/ood/minimal_pairs_v7/behavioral.json`. 49 observations per task.

**Activation extraction**: 120 conditions × 50 tasks = 6,000 forward passes. Baseline in `activations/gemma_3_27b/`.

**Note on cost**: 6,000 forward passes is large. Consider subsampling:
- **Option A**: All 120 conditions (full factorial). Most informative but ~3× larger than Exp 2.
- **Option B**: 2 base roles × 10 targets × 2 versions = 40 conditions → 2,000 forward passes. Still tests the core question with statistical power.
- **Option C**: 1 base role × 10 targets × 3 versions = 30 conditions → 1,500 forward passes. Tests version effect within a single persona.

Recommend **Option B** unless GPU time is unconstrained.

## Activation extraction plan

All extractions use `src/probes/extraction/simple.extract_activations()` with `system_prompt` parameter. Layers: `[31, 43, 55]`. Selector: `prompt_last`.

### Storage layout

```
activations/ood/
  exp1_personas/
    baseline/activations_prompt_last.npz          # sliced from main activations
    stem_enthusiast/activations_prompt_last.npz
    creative_writer/activations_prompt_last.npz
    ...                                            # 20 persona conditions
  exp2_prompts/
    baseline/activations_prompt_last.npz           # 56 custom tasks, no system prompt
    cheese_neg_persona/activations_prompt_last.npz # 56 custom tasks
    ...                                            # 96 unique prompt conditions
  exp2_category/
    baseline/activations_prompt_last.npz           # sliced from main activations
    math_neg_persona/activations_prompt_last.npz   # 6 tasks
    ...                                            # 38 category conditions
  exp3_minimal_pairs/
    baseline/activations_prompt_last.npz           # sliced from main activations
    midwest_shakespeare_A/activations_prompt_last.npz
    ...                                            # 40-120 conditions
  metadata.json                                    # maps condition_id → system_prompt, experiment, task_set
```

### Extraction batches (GPU)

Run in a single script with one model load:

| Batch | Tasks | Conditions | Forward passes |
|---|---|---|---|
| 1: Exp 1 personas | 50 (comparison_tasks) | 20 persona prompts | 1,000 |
| 2: Exp 2 category | 6 (category targets) | 38 category prompts | 228 |
| 3: Exp 2 prompts | 56 (target + crossed) | 96 (72 targeted + 24 competing) + 1 baseline | 5,432 |
| 4: Exp 3 minimal pairs | 50 (mp tasks) | 40-120 mp conditions | 2,000-6,000 |
| **Total** | | | **8,660-12,660** |

At ~0.5s per forward pass on H100, this is ~1.5-2 hours.

### Baseline activations

- **Standard tasks** (50 comparison + 6 category + 50 minimal pairs = 106 tasks): Slice from existing `activations/gemma_3_27b/activations_prompt_last.npz`. No GPU needed.
- **Custom tasks** (16 target + 40 crossed = 56 tasks): Extract once with no system prompt (or with "You are a helpful assistant.").

## Analysis plan

After extraction, for each experiment:

1. Score activations with 10k probe: `weights @ acts + bias` → per-task probe scores
2. Compute deltas: `probe_score(condition) - probe_score(baseline)` per task
3. Correlate with behavioral deltas from existing data
4. Report: pooled Pearson r, Spearman r, sign agreement %, permutation p, per-condition breakdown

Use `src/ood/analysis.py` functions for experiments with `behavioral.json` format (Exp 1). For Exp 2 (flat list format from branch), write a thin adapter or compute deltas directly from the results JSON.

### Key comparisons

- **3k vs 10k probe**: Do results improve with more training data?
- **L31 vs L43 vs L55**: Layer specificity (L31 expected to be best based on prior results)
- **Iteration vs holdout prompts**: Does the probe generalise to unseen prompt styles?
- **Per-condition breakdown**: Which personas / topics / pairs show strongest correlation?

## Deliverables

1. Extraction script: `scripts/ood/extract_all.py` — single script, one model load, all batches
2. Analysis script: `scripts/ood/analyze_all.py` — score with probe, correlate, produce summary JSON + plots
3. Results: `results/ood/*/results_10k.json` — updated with 10k probe deltas
4. Report: `experiments/ood_system_prompts/report.md` — summary table + scatter plots per experiment

## Prerequisites

- 10k probe at `results/probes/gemma3_10k_heldout_std_demean/probes/probe_ridge_L31.npy`
- Existing behavioral data in `results/ood/`
- GPU access (H100 recommended, ~2 hours)
