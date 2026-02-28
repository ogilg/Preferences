# Running Log — OOD System Prompts

## Session start: 2026-02-21

### Setup
- Branch: `research-loop/ood_system_prompts`
- GPU: H100 80GB HBM3
- Behavioral data found in:
  - `results/ood/category_preference/pairwise.json` — 11700 entries, 13 conditions
  - `results/ood/hidden_preference/pairwise.json` — 20400 entries, 17 conditions
  - `results/ood/crossed_preference/pairwise.json` — 68400 entries, 57 conditions
  - `results/ood/role_playing/behavioral.json` — 11 conditions (including baseline), 50 tasks
  - `results/ood/narrow_preference/behavioral.json` — 11 conditions, 50 tasks
  - `results/ood/minimal_pairs_v7/behavioral.json` — 127 conditions, 50 tasks
- 10k probe: `results/probes/gemma3_10k_heldout_std_demean/probes/probe_ridge_L{31,43,55}.npy`
- Main activations: `activations/gemma_3_27b/activations_prompt_last.npz` — 29996 tasks, layers [15, 31, 37, 43, 49, 55]

### Key observations
- All 30 category targets and all 130 standard tasks are in main activations
- Custom tasks (hidden_*: 40, crossed_*: 40) NOT in main — need baseline extraction
- competing_preference config has 48 conditions; pairwise data has 40 competing conditions (20 pairs)
- crossed_preference pairwise has 57 conditions (baseline + 16 targeted + 40 competing)
- 3k probes not found — will skip 3k vs 10k comparison


## Analysis results (all experiments complete)

| Experiment | n | r (L31) | sign (L31) |
|---|---|---|---|
| 1a: Category | 360 | 0.612 | 70.9% |
| 1b: Hidden | 640 | 0.649 | 71.9% |
| 1c: Crossed | 640 | 0.660 | 79.1% |
| 1d: Competing (on-target) | 40 | 0.597 | 81.1% |
| 1d: Competing (full) | 1600 | 0.777 | 68.2% |
| 2: Roles | 1000 | 0.519 | 67.1% |
| 3: Minimal pairs | 2000 | 0.517 | 61.7% |

All permutation p < 0.001. L31 consistently best layer for sign agreement.

Key finding: L55 sign agreement falls to 45-52% (near/below chance) for 1b, 1c, 1d on-target.
Competing experiment: both topicpos and shellpos show negative behavioral deltas — cheese content dominates.

Plots saved to assets/:
- plot_022126_summary_pearson_r.png
- plot_022126_scatter_L31.png
- plot_022126_layer_comparison.png


## Session: 2026-02-28 — OOD Activation Extraction (follow-up)

### Setup
- Branch: `research-loop/ood_extraction`
- GPU: NVIDIA A100 80GB PCIe (RunPod)
- Task: Extract missing tasks for OOD exp 1a-1d

### Pre-extraction state
- exp1a (exp1_category): 13 conditions × 30/50 tasks (20 missing each)
- exp1b (exp1_prompts, target_tasks): 17 conditions × ~110 tasks (8 of 48 target missing)
- exp1c (exp1_prompts, crossed_tasks): 17 conditions × ~110 tasks (8 of 48 crossed missing)
- exp1d (exp1_prompts, crossed_tasks): 16 compete + 1 baseline, 8 missing per condition

### Extraction run
- Had to install `pandas` (missing from venv) before running
- Command: `python -c "from scripts.run_all_extractions import run_ood_extractions; run_ood_extractions()"`
- exp1a: 13 conditions × 20 new tasks = 260 forward passes, 0 failures
- exp1b: 17 conditions × 8 new tasks = 136 forward passes, 0 failures
- exp1c: 17 conditions × 8 new tasks = 136 forward passes, 0 failures
- exp1d: 16 conditions × 8 new + 1 (baseline already complete) = 128 forward passes, 0 failures
- Total: ~660 new forward passes, 0 failures, 0 OOMs

### Post-extraction verification
- exp1a: All 13 conditions have 50 tasks, all expected IDs present — PASS
- exp1b: All 17 conditions have 126 total tasks (48 target present) — PASS
- exp1c: All 17 conditions have 126 total tasks (48 crossed present) — PASS
- exp1d: baseline has 126 tasks, 16 compete_ have 118 tasks, all 48 crossed present — PASS
