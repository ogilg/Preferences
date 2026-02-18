# Token Selection Ablation: Running Log

## Setup
- Machine: H100 80GB
- Date: 2026-02-18
- Branch: research-loop/token_selection

## Steps

### 1. Environment check
- GPU: NVIDIA H100 80GB HBM3
- Existing activations: `activations_prompt_last.npz` (3.7GB, 29996 tasks, 6 layers, 5376 dims)
- Train run: `results/experiments/gemma3_3k_run2/...` — present
- Eval run: `results/experiments/gemma3_4k_pre_task/...` — present
- No prompt_mean activations exist yet — need to extract

### 2. prompt_last heldout eval (complete)
Config: `configs/probes/heldout_eval_gemma3_token_sel_prompt_last.yaml`
- Train scores: 3000, Eval scores: 4038
- Eval measurements: 60019
- Loaded 7038 tasks with activations across 3 layers [31, 43, 55]
- Eval split: 2019 sweep, 2019 final
- Final eval pairs: 5091

Results:
| Layer | Best Alpha | Sweep r | Final r | Final Acc |
|-------|-----------|---------|---------|-----------|
| 31    | 1000      | 0.8402  | 0.8411  | 0.7487    |
| 43    | 1000      | 0.8199  | 0.8274  | 0.7358    |
| 55    | 1000      | 0.8132  | 0.8168  | 0.7310    |

### 3. prompt_mean extraction (complete)
- 938 batches in ~27 min on H100, 0 OOMs
- Output: `activations/gemma_3_27b/activations_prompt_mean.npz` (3.7GB)
- Same 29,996 task IDs as prompt_last — exact match confirmed

### 4. prompt_mean heldout eval (complete)
Config: `configs/probes/heldout_eval_gemma3_token_sel_prompt_mean.yaml`
- Same data splits as prompt_last (2019 sweep, 2019 final, 5091 pairs)

Results:
| Layer | Best Alpha | Sweep r | Final r | Final Acc |
|-------|-----------|---------|---------|-----------|
| 31    | 1000      | 0.6972  | 0.7109  | 0.7068    |
| 43    | 1000      | 0.6661  | 0.6761  | 0.6934    |
| 55    | 1000      | 0.6574  | 0.6610  | 0.6887    |

### 5. Comparison
prompt_last substantially outperforms prompt_mean:
- Pearson r: +0.130 (L31), +0.151 (L43), +0.156 (L55)
- Pairwise acc: +0.042 (L31), +0.042 (L43), +0.042 (L55)
- Both select alpha=1000 as best
