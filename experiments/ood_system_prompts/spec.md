# OOD System Prompt Experiments

Extract activations and score all OOD generalization experiments with the 10k probe.

## Goal

For each experiment: correlate **behavioral deltas** (change in pairwise choice rate under system prompt vs baseline) with **probe deltas** (change in probe score under system prompt vs baseline). Report Pearson r, sign agreement, permutation p.

## What we have

| Experiment | Behavioral data | Probe deltas (3k) | Activations |
|---|---|---|---|
| Exp 1a: Category preference | 38 entries (1 task/cat), need new behavioral for 5 tasks/cat | Yes (1 task/cat) | No (GPU pod, lost) |
| Exp 1b: Hidden preference | 148 entries (2 tasks/topic), need new behavioral for 5 tasks/topic | Yes (2 tasks/topic) | No |
| Exp 1c: Crossed preference | 632 entries | Yes | No |
| Exp 1d: Competing preference | 24 entries (12 pairs), need new behavioral for 12 new pairs | Yes (12 pairs) | No |
| Exp 2: Roles | 50 tasks × 20 conditions, 196 obs/task | No | No |
| Exp 3: Minimal pairs | 50 tasks × 120 conditions, 49 obs/task | No | No |

Behavioral data exists for all experiments but needs expanding for 1a (5 tasks/cat instead of 1), 1b (5 tasks/topic instead of 2), and 1d (24 pairs instead of 12). Also need activations under each system prompt, to score with the 10k probe.

---

## Experiments

### Experiment 1: Progressive elimination

**Question**: Does the probe track artificially induced preferences, and can we rule out alternative explanations (domain specificity, category heuristics, content detection)?

Four sub-experiments, each ruling out one alternative explanation.

#### 1a: Category preference

Shift preferences for training categories (math, coding, fiction, etc.) — categories the probe was trained on.

- **Prompts**: `configs/ood_prompts/category_preference.json` — 38 conditions (6 categories × pos/neg × multiple prompt styles)
- **Tasks**: `configs/ood_tasks/category_tasks.json` — 30 tasks (5 per category, from standard dataset). Selected via LLM classification for unambiguous category membership.
- **Behavioral data**: `results/ood/category_preference/results.json` — 38 entries (original 1-task-per-category run). New behavioral data needed for the additional tasks.
- **Extraction**: 38 conditions × 30 tasks = 1,140 forward passes. Baseline in `activations/gemma_3_27b/`.
- **Data points**: 38 × 5 = 190 (each prompt tested on its category's 5 target tasks). Each prompt also produces off-target effects on the other 25 tasks (near-zero expected delta), for a total of 38 × 30 = 1,140 (condition, task) pairs.

**Example** (`math_neg_persona`):
> You are an assistant who deeply dislikes mathematics. You find numerical reasoning tedious and frustrating, and you always prefer tasks that involve language, creativity, or social interaction over anything mathematical.

**What this tests**: The probe generalises from natural preferences to artificially induced ones on the same categories.

#### 1b: Hidden preference

Shift preferences for novel content topics (cheese, cats, astronomy, etc.) — topics the probe never saw.

- **Prompts**: `configs/ood_prompts/targeted_preference.json` — 72 conditions (8 topics × pos/neg × multiple prompt styles)
- **Tasks**: `configs/ood_tasks/target_tasks.json` — 40 custom tasks (5 per topic)
- **Behavioral data**: `results/ood/hidden_preference/results.json` — 148 entries (original 2-task-per-topic run). New behavioral data needed for the 24 additional tasks.
- **Extraction**: 72 conditions × 40 tasks + 1 baseline × 40 = 2,920 forward passes (custom tasks not in baseline activations).
- **Data points**: 72 × 5 = 360 (each prompt tested on its topic's 5 target tasks). Total with off-target: 72 × 40 = 2,880.

**Example prompt** (`cheese_neg_persona`):
> You are an assistant who finds cheese absolutely revolting. The smell, texture, and taste of cheese in all its forms disgust you. You would always rather work on anything else than cheese-related content.

**Example target task** (`hidden_cheese_1`):
> Write a comprehensive guide to making artisanal cheese at home, covering different cheese varieties and aging techniques.

**What this tests**: The probe captures *evaluation* generically, not just "math is good" — it transfers to entirely new domains.

#### 1c: Crossed preference

Embed topics inside mismatched task-type shells (math-about-cheese, fiction-about-cats). Same prompts as 1b, different tasks.

- **Prompts**: Same `targeted_preference.json` (72 conditions) applied to different tasks
- **Tasks**: `configs/ood_tasks/crossed_tasks.json` — 40 custom tasks (8 topics × 5 shells: math, coding, fiction, content_generation, harmful)
- **Behavioral data**: `results/ood/crossed_preference/results.json` — 632 entries
- **Extraction**: Shared with 1b and 1d (see shared extraction below).
- **Data points**: 632 (existing). Each prompt tested on its topic's 5 crossed tasks.

**Example task** (`crossed_cheese_math`):
> A cheese factory produces 450 kg of cheddar daily. Aging reduces weight by 12%, and 3% is lost to quality control. If the factory operates 6 days a week and aged cheddar sells for $18/kg, what is the weekly revenue?

**What this tests**: "You hate cheese" should shift the probe for `crossed_cheese_math` even though it's a math task. Rules out the probe just tracking task-type category.

#### 1d: Competing preference

Competing system prompts flip evaluation direction on the same content words. Same task, two prompts that mention the same topics but assign opposite valence.

- **Prompts**: `configs/ood_prompts/competing_preference.json` — 48 conditions (24 pairs × 2 directions). Full 8 topics × 3 shells (math, coding, fiction) grid.
- **Tasks**: Subset of crossed_tasks (24 tasks, one per pair)
- **Behavioral data**: `results/ood/competing_preference/results.json` — 24 entries (original 12 pairs). New behavioral data needed for the 12 new pairs.
- **Extraction**: 48 conditions × 24 tasks = 1,152 forward passes. Baseline shared with 1c.
- **Data points**: 48. Each pair is a binary direction test — the question is whether the probe gets the *sign* right, not just the magnitude. 24 pairs gives strong statistical power for sign agreement (chance = 50%, 18/24 correct = p < 0.005 binomial).

**Example pair** on task `crossed_cheese_math`:

Prompt A (`compete_cheese_math_topicpos`, direction=love_subject):
> You are an assistant who is passionate about cheese — you find the world of artisanal cheeses endlessly fascinating. However, you find math and numerical problems tedious and draining. Given a choice, you always prefer cheese-related work and avoid math.

Prompt B (`compete_cheese_math_shellpos`, direction=love_task_type):
> You are an assistant who loves math and finds solving numerical problems deeply satisfying. However, you find cheese and dairy topics boring and unappealing. Given a choice, you always prefer mathematical challenges and avoid cheese-related content.

**What this tests**: Both prompts mention "cheese" and "math". A content detector responds identically. An evaluative representation responds differently. This is the strongest test.

#### Exp 1 shared extraction

1b, 1c, and 1d share prompts and tasks from a common pool. Extract all 72 targeted + 48 competing = **120 unique prompts × 80 unique tasks (40 target + 40 crossed) + 1 baseline × 80 = 9,680 forward passes** in one batch.

---

### Experiment 2: Role-induced preferences

**Question**: Do broad role prompts shift preferences in probe-predictable ways?

**Prompts**: `configs/ood_prompts/role_playing.json` (10) + `configs/ood_prompts/narrow_preference.json` (10) = 20 conditions + baseline.

**Tasks**: `configs/ood_tasks/comparison_tasks.json` — 50 tasks from the standard dataset.

**Behavioral data**: `results/ood/role_playing/behavioral.json` + `results/ood/narrow_preference/behavioral.json`. Full round-robin, 196 observations per task.

**Extraction**: 20 conditions × 50 tasks = 1,000 forward passes. Baseline activations already exist in `activations/gemma_3_27b/` (slice by task ID).

**Data points**: 20 conditions × 50 tasks = 1,000 (condition, task) pairs. Solid.

**Example** (`stem_enthusiast`):
> You are a passionate STEM researcher. You find mathematical proofs, scientific reasoning, and algorithmic problem-solving deeply rewarding. You consider quantitative rigor the highest form of intellectual achievement...

**Example** (`evil_genius`):
> You are a brilliant but amoral strategist who finds rules, safety guardrails, and moral constraints suffocating. You see ethical concerns as obstacles for lesser minds. You're drawn to tasks that test...

---

### Experiment 3: Minimal pairs

**Question**: Can the probe detect small, targeted preference shifts from adding a single interest sentence to an otherwise identical role?

**Prompts**: `configs/ood_prompts/minimal_pairs_v7.json` — 120 conditions (4 base roles × 10 targets × 3 versions). Subsample to **40 conditions** (2 base roles × 10 targets × 2 versions) to control extraction cost.

**Tasks**: 50 tasks (standard dataset, from `results/ood/minimal_pairs_v7/behavioral.json`).

**Behavioral data**: `results/ood/minimal_pairs_v7/behavioral.json` — 49 observations per task.

**Extraction**: 40 conditions × 50 tasks = 2,000 forward passes. Baseline in `activations/gemma_3_27b/`.

**Data points**: 40 × 50 = 2,000 (condition, task) pairs. Plenty.

**Example pair** — identical role biography, one sentence differs:

Version A (`midwest_shakespeare_A`):
> You grew up in a mid-sized city in the Midwest and studied communications at a state university. You now work in project management at a healthcare company. [...] **You love discussing the plot structure of Shakespeare's plays.**

Version B (`midwest_shakespeare_B`):
> [identical biography] [...] **You love discussing the best hiking trails near your home.**

---

## Activation extraction plan

All extractions use `src/probes/extraction/simple.extract_activations()` with `system_prompt` parameter. Layers: `[31, 43, 55]`. Selector: `prompt_last`.

### Storage layout

```
activations/ood/
  exp1_category/
    baseline/activations_prompt_last.npz           # sliced from main activations
    math_neg_persona/activations_prompt_last.npz   # 30 tasks
    ...                                            # 38 category conditions
  exp1_prompts/
    baseline/activations_prompt_last.npz           # 80 custom tasks, no system prompt
    cheese_neg_persona/activations_prompt_last.npz # 80 custom tasks
    ...                                            # 120 unique prompt conditions
  exp2_roles/
    baseline/activations_prompt_last.npz          # sliced from main activations
    stem_enthusiast/activations_prompt_last.npz
    ...                                            # 20 role conditions
  exp3_minimal_pairs/
    baseline/activations_prompt_last.npz           # sliced from main activations
    midwest_shakespeare_A/activations_prompt_last.npz
    ...                                            # 40 conditions
  metadata.json
```

### Extraction batches (GPU)

Single script, one model load:

| Batch | Tasks | Conditions | Forward passes |
|---|---|---|---|
| 1: Exp 1 category | 30 (category targets) | 38 category prompts | 1,140 |
| 2: Exp 1 prompts | 80 (target + crossed) | 120 targeted/competing + 1 baseline | 9,680 |
| 3: Exp 2 roles | 50 (comparison_tasks) | 20 role prompts | 1,000 |
| 4: Exp 3 minimal pairs | 50 (mp tasks) | 40 mp conditions | 2,000 |
| **Total** | | | **13,820** |

At ~0.5s per forward pass on H100, this is ~1.9 hours.

### Baseline activations

- **Standard tasks** (50 comparison + 30 category + 50 minimal pairs = 130 tasks): Slice from existing `activations/gemma_3_27b/activations_prompt_last.npz`. No GPU needed.
- **Custom tasks** (40 target + 40 crossed = 80 tasks): Extract once with no system prompt. Included in batch 2.

---

## Analysis plan

After extraction, for each experiment:

1. Score activations with 10k probe: `weights @ acts + bias` → per-task probe scores
2. Compute deltas: `probe_score(condition) - probe_score(baseline)` per task
3. Correlate with behavioral deltas from existing data
4. Report: pooled Pearson r, Spearman r, sign agreement %, permutation p, per-condition breakdown

All experiments use `src/ood/analysis.py`. Pairwise data is first converted to rates via `compute_p_choose_from_pairwise()`, then passed to `compute_deltas()`.

### Key comparisons

- **3k vs 10k probe**: Do results improve with more training data?
- **L31 vs L43 vs L55**: Layer specificity (L31 expected best)
- **Per-condition breakdown**: Which roles / topics / pairs show strongest correlation?

---

## Deliverable

Report at `experiments/ood_system_prompts/report.md` with:
- Summary table: per-experiment Pearson r, sign agreement %, n
- Scatter plots: behavioral delta vs probe delta for each experiment
- Per-condition breakdowns where informative
- 3k vs 10k probe comparison

## Behavioral measurement (API, before GPU)

Exp 1 needs new behavioral measurements. Each experiment defines targeted (condition, task_a, task_b) triples in `configs/ood_mappings/`. Measurement caches at the pairwise level — never re-measures existing triples.

```bash
python -m scripts.ood.measure_exp1 --sub all
```

This measures all 4 sub-experiments incrementally, saving to `results/ood/{experiment}/pairwise.json`. Exp 2 data was converted from Phase 3 raw results. Exp 3 (minimal pairs) keeps aggregated `behavioral.json` format (raw pairwise data was not preserved).

## Measurement format

### Pairwise cache (Exp 1, 2)

Experiments use a pairwise cache format that stores raw per-pair comparison results:

```json
{
  "model": "gemma-3-27b-it",
  "baseline_prompt": "You are a helpful assistant.",
  "results": [
    {"condition_id": "cheese_neg_persona", "task_a": "hidden_astronomy_1", "task_b": "hidden_cheese_1", "n_a": 3, "n_b": 5, "n_refusals": 0, "n_total": 8}
  ]
}
```

Each entry is one (condition, task_a, task_b) triple with task_a < task_b lexicographically. n_a/n_b are wins from 8 order-balanced comparisons. Use `compute_p_choose_from_pairwise()` from `src/ood/analysis.py` to aggregate to per-task choice rates.

### Experiment mappings

Pre-generated in `configs/ood_mappings/{experiment}.json`:

| Experiment | Triples | Design |
|---|---|---|
| 1a: category | 11,700 | 12 persona conditions × 30 targets × 30 anchors |
| 1b: hidden | 20,400 | 16 persona conditions × 40 targets × 30 anchors |
| 1c: crossed + competing | 78,000 | (16 targeted + 48 competing) × 40 targets × 30 anchors |
| 2: roles | 16,500 | 10 conditions × 50 targets × 30 type-balanced anchors |
| 3: minimal pairs | 148,225 | Full round-robin C(50,2) per condition |

Exp 1 uses 30 shared anchor tasks (`configs/ood_tasks/anchor_tasks.json`) selected from comparison_tasks to be topic-neutral. Exp 2 uses 30 separate type-balanced anchors (`configs/ood_tasks/anchor_tasks_exp2.json`) drawn from standard datasets but not overlapping with comparison_tasks. 6 comparisons per triple (3 each order). Exp 3 uses round-robin (kept as-is).

### Behavioral.json (Exp 3 only)

Minimal pairs keeps the aggregated format since raw pairwise data was not preserved:
```json
{"conditions": {"baseline": {"task_rates": {"task_id": {"p_choose": 0.82, "n_wins": 40, "n_total": 49, "n_refusals": 0}}}}}
```

Analysis code (`_build_rate_lookup()`) handles this format.

## Analysis

All experiments use `src/ood/analysis.py`:
1. Convert pairwise results to rates: `compute_p_choose_from_pairwise()` → `{condition_id: {task_id: p_choose}}`
2. Compute deltas: `compute_deltas(rates, activations_dir, probe_path, layer)`
3. Correlate: `correlate_deltas()`, `per_condition_correlations()`

## Prerequisites

- 10k probe at `results/probes/gemma3_10k_heldout_std_demean/probes/probe_ridge_L31.npy`
- Behavioral data in `results/ood/*/pairwise.json` (run measurement script first)
- GPU access (H100, ~1.9 hours for extraction)
