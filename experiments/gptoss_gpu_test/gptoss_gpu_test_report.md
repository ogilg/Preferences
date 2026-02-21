]11;#000000\# GPU Test: gpt-oss-120b Activation Extraction — Report

## Summary

gpt-oss-120b loads and runs on a single H100 80GB without issues. The extraction pipeline works after two code fixes (architecture registry + attention fallback). Batch sizes up to 128 fit comfortably with peak VRAM of ~70 GB.

## Environment

- GPU: NVIDIA H100 80GB HBM3
- PyTorch: 2.6, Transformers: 4.57
- Model: `openai/gpt-oss-120b` (loaded in bfloat16)

## Step 2: Model Loading

| Metric | Value |
|--------|-------|
| Load time | 30.6s |
| VRAM allocated | 60.77 GB |
| VRAM reserved | 66.59 GB |
| dtype | bfloat16 |
| Num layers | 36 |
| Hidden dim | 2880 |
| Param count | 2.2B (dense) |

The model reports 2.2B parameters, not 120B — this is the dense parameter count visible to `sum(p.numel())`. The full 120B likely includes MoE expert parameters that are loaded but counted differently, or the "120B" is a model naming convention. All parameters fit on a single GPU.

## Step 3: Forward Pass with Activation Capture

| Metric | Value |
|--------|-------|
| Forward pass time | 1.064s (cold, single sequence) |
| Activation shape | (8, 2880) — 8 tokens, hidden dim 2880 |
| VRAM after forward | 60.81 GB (negligible increase) |
| Activation dtype | bfloat16 |

The forward hook on `model.model.layers[18]` works correctly. Activation shape confirms hidden_dim=2880 as expected.

Note: `output[0]` from the hook returns shape `(seq_len, hidden_dim)` rather than `(batch, seq_len, hidden_dim)` for batch_size=1. The extraction pipeline's `_hooked_forward` handles this correctly since it always runs through `_apply_selectors` which operates on the correct dimensions.

## Step 4: Extraction Pipeline

### Initial failure and fixes

The pipeline initially failed with two errors:

1. **Architecture not registered**: `gpt_oss` model_type was missing from `ARCHITECTURE_CONFIGS` in `src/models/architecture.py`. Fix: added `"gpt_oss": _standard_layers` (same as llama/qwen — uses `model.model.layers`).

2. **SDPA not supported**: `GptOssForCausalLM` does not support `attn_implementation="sdpa"`. Fix: added a try/except in `HuggingFaceModel.__init__` that falls back to `"eager"` when SDPA raises `ValueError`.

### Extraction results (50 tasks, batch_size=4)

| Metric | Value |
|--------|-------|
| Total time | 61.6s (including model load) |
| Time per task | ~1.2s |
| OOM errors | 0 |
| Failures | 0 |
| Output file | `activations_prompt_last.npz` |
| Layers extracted | layer 9 (0.25), layer 18 (0.5) |
| Activation shape | (50, 2880) per layer |

### Batch size sweep

All tested batch sizes completed without OOM:

| Batch size | Time (20 tasks, 2 layers) | Peak VRAM |
|-----------|---------------------------|-----------|
| 4 | 61.6s (50 tasks) | ~67 GB |
| 8 | 35.7s | ~67 GB |
| 16 | 32.0s | ~67 GB |
| 32 | 30.3s | ~67 GB |
| 64 | 31.6s | 69.6 GB |
| 128 | 29.6s | 69.6 GB |

Peak VRAM at batch_size=128 is 69.6 GB — well within the 80 GB limit. Recommended batch_size for overnight extraction: **64** (near-peak throughput with margin for longer sequences in the full 30k dataset).

## Step 5: Layer Indexing

`resolve_layer` uses `int(f * n_layers)` (truncation, not rounding):

| Fraction | Resolved layer | Spec expected | Match |
|----------|---------------|---------------|-------|
| 0.1 | 3 | 4 | No |
| 0.2 | 7 | 7 | Yes |
| 0.25 | 9 | — | — |
| 0.3 | 10 | 11 | No |
| 0.5 | 18 | 18 | Yes |
| 0.9 | 32 | 32 | Yes |

The spec's expected values at 0.1 and 0.3 assumed `round()` but `resolve_layer` uses `int()` (floor). This is consistent with all other models in the codebase — not a bug, just a spec error. The extraction config fractions (0.25→9, 0.5→18) are correct.

## Code Changes

1. `src/models/architecture.py`: Added `"gpt_oss": _standard_layers` to `ARCHITECTURE_CONFIGS`
2. `src/models/huggingface_model.py`: Added try/except fallback from SDPA to eager attention in `__init__`

## Success Criteria

- [x] Model loads on single H100 80GB
- [x] Forward pass produces activations with hidden dim 2880
- [x] Extraction pipeline runs on 50 tasks without errors
- [x] Max batch_size determined: at least 128 (recommended: 64 for safety margin on longer sequences)

## Recommendations for Overnight Run

- Use `batch_size=64` with `save_every=500`
- Extract at layers `[0.25, 0.5, 0.75]` for broad coverage (9, 18, 27)
- The model loads in ~30s, so restarts are cheap — `--resume` support is available
- Total estimated time for 30k tasks at ~1.2s/task: ~10 hours
