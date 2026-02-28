# OOD Activation Extraction — Report

Extracted missing activations for OOD experiments 1a-1d. All conditions used `--resume` to skip already-extracted task IDs.

## Results

| Experiment | Conditions | Expected tasks | New extractions | Failures | Status |
|---|---:|---:|---:|---:|---|
| 1a (category) | 13 | 50 | 20 per condition | 0 | PASS |
| 1b (target) | 17 | 48 | 8 per condition | 0 | PASS |
| 1c (crossed) | 17 | 48 | 8 per condition | 0 | PASS |
| 1d (competing) | 17 | 48 | 8 per condition (baseline already complete) | 0 | PASS |

Total: ~660 forward passes on A100 80GB. Layers [31, 43, 55], selector `prompt_last`.

## Verification

All expected task IDs confirmed present in every condition's npz file:
- **exp1a**: 13 conditions × 50 tasks in `activations/ood/exp1_category/{condition}/`
- **exp1b**: 17 conditions × 48 target tasks (126 total) in `activations/ood/exp1_prompts/{condition}/`
- **exp1c**: 17 conditions × 48 crossed tasks (126 total) in `activations/ood/exp1_prompts/{condition}/`
- **exp1d**: 16 compete + 1 baseline × 48 crossed tasks (118-126 total) in `activations/ood/exp1_prompts/{condition}/`

## Notes

- Had to install `pandas` in the RunPod venv before the extraction script would import.
- exp1b/c/d share the same output root (`exp1_prompts`), so conditions accumulate tasks from multiple experiments. The total counts (118-126) reflect this accumulation, but all required task IDs for each experiment are present.
- exp1a extracts 50 tasks per condition, but behavioral data only covers 30. The extra 20 tasks have activations but no behavioral measurements.
