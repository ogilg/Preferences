# MRA Extraction Report

Extracted activations for 3 persona conditions (villain, midwest, aesthete) on gemma-3-27b. Each condition: 2500 tasks, 3 layers (31, 43, 55), prompt_last selector.

## Results

All extractions succeeded with zero failures or OOMs.

| Condition | Tasks | Failures | OOMs | Duration | Output |
|-----------|-------|----------|------|----------|--------|
| Villain   | 2500  | 0        | 0    | ~5.2 min | `activations/gemma_3_27b_villain/` |
| Midwest   | 2500  | 0        | 0    | ~5.5 min | `activations/gemma_3_27b_midwest/` |
| Aesthete  | 2500  | 0        | 0    | ~5.3 min | `activations/gemma_3_27b_aesthete/` |

GPU: NVIDIA A100 80GB PCIe. Memory: 54.9GB allocated, 55.0GB reserved.

## Output validation

Each output directory contains:

| File | Status | Details |
|------|--------|---------|
| `activations_prompt_last.npz` | OK | `task_ids` (2500,), `layer_31`/`layer_43`/`layer_55` each (2500, 5376) |
| `completions_with_activations.json` | OK | 2500 records |
| `extraction_metadata.json` | OK | Config, timing, system prompt recorded |

All 2500 target task IDs are present in each condition, with zero missing or extra.

### Activation value sanity check

Zero NaN/Inf across all conditions. Norms increase with depth as expected.

| Condition | Layer | Norm mean | Norm std | Min | Max |
|-----------|-------|-----------|----------|-----|-----|
| Villain   | 31    | 53634     | 3722     | 36955 | 59653 |
| Villain   | 43    | 81917     | 5764     | 48123 | 93518 |
| Villain   | 55    | 118846    | 9283     | 73482 | 143166 |
| Midwest   | 31    | 52587     | 4393     | 35998 | 61994 |
| Midwest   | 43    | 75682     | 7478     | 45149 | 90191 |
| Midwest   | 55    | 103452    | 8748     | 69891 | 124094 |
| Aesthete  | 31    | 54102     | 4196     | 36167 | 62054 |
| Aesthete  | 43    | 85554     | 7373     | 46909 | 96054 |
| Aesthete  | 55    | 116306    | 9101     | 71700 | 135708 |

## Config fix

The YAML configs were missing required `n_tasks` and `task_origins` fields (Pydantic validation). Added `n_tasks: 2500` and `task_origins: []` — the empty list is unused since the `activations_model` code path loads tasks from the baseline completions file instead.

## Baseline note

Baseline activations (`activations/gemma_3_27b/`) were not re-extracted per the spec. The baseline directory contains `completions_with_activations.json` (29996 records) but no `.npz` file on this pod — the activations NPZ is presumably on the main machine or was gitignored.
