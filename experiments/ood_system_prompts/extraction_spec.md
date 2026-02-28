# OOD Activation Extraction

Extract missing tasks for OOD exp 1a-1d. All conditions exist but are missing recently added tasks.

## Command

```bash
python -c "from scripts.run_all_extractions import run_ood_extractions; run_ood_extractions()"
```

Uses `--resume` to skip already-extracted task IDs. Do NOT run `run_all_extractions.py` directly — it also triggers MRA extractions.

| Experiment | Conditions | Tasks | Activation output | Missing |
|---|---:|---|---|---:|
| 1a | 13 | 50 from standard pool via `include_task_ids_file` | `activations/ood/exp1_category/{condition}/` | ~20 |
| 1b | 17 | 48 target tasks via `custom_tasks_file` | `activations/ood/exp1_prompts/{condition}/` | ~8 |
| 1c | 17 | 48 crossed tasks via `custom_tasks_file` | `activations/ood/exp1_prompts/{condition}/` | ~8 |
| 1d | 17 | 48 crossed tasks via `custom_tasks_file` | `activations/ood/exp1_prompts/{condition}/` | ~8 |

Layers: [31, 43, 55]. Selector: prompt_last. ~700 forward passes, ~5 min on H100.

## Verification

After completion, check:
1. **exp1a**: each condition dir in `activations/ood/exp1_category/` has 50 task IDs in its npz
2. **exp1b/c**: each condition dir in `activations/ood/exp1_prompts/` has 48 task IDs
3. **exp1d**: each compete_ dir in `activations/ood/exp1_prompts/` has 48 task IDs
