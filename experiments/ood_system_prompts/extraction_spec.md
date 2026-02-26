# Activation Extraction (OOD + MRA)

Run all pending activation extractions. A single driver script handles everything.

## Command

```bash
python scripts/run_all_extractions.py
```

## What it does

The script reads AL measurement configs to get system prompts and task files, then runs the main extraction pipeline per condition with `--resume` (only extracts missing task IDs).

| Experiment | AL configs dir | Tasks | Activation output | New forward passes |
|---|---|---|---|---:|
| OOD exp1a | `configs/measurement/active_learning/ood_exp1a/` (13) | 50 from standard pool via `include_task_ids_file` | `activations/ood/exp1_category/{condition}/` | ~260 |
| OOD exp1b | `configs/measurement/active_learning/ood_exp1b/` (17) | 48 target tasks via `custom_tasks_file` | `activations/ood/exp1_prompts/{condition}/` | ~136 |
| OOD exp1c | `configs/measurement/active_learning/ood_exp1c/` (17) | 48 crossed tasks via `custom_tasks_file` | `activations/ood/exp1_prompts/{condition}/` | ~136 |
| OOD exp1d | `configs/measurement/active_learning/ood_exp1d/` (17) | 48 crossed tasks via `custom_tasks_file` | `activations/ood/exp1_prompts/{condition}/` | ~816 |
| MRA villain | `configs/extraction/mra_persona2_villain.yaml` | ~1500 via `activations_model` + `task_ids_file` | `activations/gemma_3_27b_villain/` | ~1500 |
| MRA midwest | `configs/extraction/mra_persona3_midwest.yaml` | ~1500 | `activations/gemma_3_27b_midwest/` | ~1500 |
| MRA aesthete | `configs/extraction/mra_persona4_aesthete.yaml` | ~1500 | `activations/gemma_3_27b_aesthete/` | ~1500 |

OOD layers: [31, 43, 55]. MRA layers: [0.5] (→ layer 31). Selector: prompt_last.

Total: ~5850 forward passes, ~40 min on H100.

## Verification

After completion, check:
1. **OOD exp1a**: each condition dir in `activations/ood/exp1_category/` has 50 task IDs in its npz
2. **OOD exp1b/c**: each condition dir in `activations/ood/exp1_prompts/` (non-compete) has 126 task IDs (110 existing + 8 from 1b + 8 from 1c)
3. **OOD exp1d**: each compete_ dir in `activations/ood/exp1_prompts/` has 48 task IDs
4. **MRA**: each output dir has `activations_prompt_last.npz` with ~1500 task IDs and `extraction_metadata.json` with `n_failures` near 0
