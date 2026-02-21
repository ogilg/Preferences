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
