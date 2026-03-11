# Character Probes: Activation Extraction — Report

## Summary

Extracted activations for 11 of 12 planned models (base Llama 3.1 8B Instruct + 10 LoRA persona variants) on 2,500 MRA tasks. All 11 models extracted cleanly: zero failures, zero OOMs. The misalignment model was inaccessible (gated repo, 403).

## Setup

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA A100-SXM4-80GB |
| Models | 11 of 12 (1 base + 10 personas; misalignment gated) |
| Tasks | 2,500 (from `configs/extraction/mra_all_2500_task_ids.txt`) |
| Selectors | 6 (`turn_boundary:-1` to `-5`, `task_mean`) |
| Layers | 5 (L8, L12, L16, L20, L24) |
| Batch size | 32 |
| Output per model | 6 `.npz` files (1 per selector, each containing all 5 layers) |

## Results

| Model | Tasks | Failures | OOMs | Time (min) | Status |
|-------|-------|----------|------|------------|--------|
| base | 2,500 | 0 | 0 | ~8.5 | OK |
| goodness | 2,500 | 0 | 0 | ~8.5 | OK |
| humor | 2,500 | 0 | 0 | ~8.5 | OK |
| impulsiveness | 2,500 | 0 | 0 | ~8.5 | OK |
| loving | 2,500 | 0 | 0 | ~8.5 | OK |
| mathematical | 2,500 | 0 | 0 | ~8.5 | OK |
| nonchalance | 2,500 | 0 | 0 | ~8.5 | OK |
| poeticism | 2,500 | 0 | 0 | ~8.5 | OK |
| remorse | 2,500 | 0 | 0 | ~8.5 | OK |
| sarcasm | 2,500 | 0 | 0 | ~8.5 | OK |
| sycophancy | 2,500 | 0 | 0 | ~8.5 | OK |
| **misalignment** | — | — | — | — | **Gated (403)** |

**Total wall time**: ~100 min for 11 models (sequential).

### Verification

All 11 models passed verification:
- 66 `.npz` files total (6 per model)
- Every file contains 5 layer keys (`layer_8` through `layer_24`)
- All activation shapes: `(2500, 4096)`
- Disk usage: 1.2 GB per model, ~13 GB total

### Output structure

```
activations/character_probes/
├── llama_3_1_8b_base/          (6 .npz files)
├── llama_3_1_8b_goodness/      (6 .npz files)
├── llama_3_1_8b_humor/         (6 .npz files)
├── llama_3_1_8b_impulsiveness/ (6 .npz files)
├── llama_3_1_8b_loving/        (6 .npz files)
├── llama_3_1_8b_mathematical/  (6 .npz files)
├── llama_3_1_8b_nonchalance/   (6 .npz files)
├── llama_3_1_8b_poeticism/     (6 .npz files)
├── llama_3_1_8b_remorse/       (6 .npz files)
├── llama_3_1_8b_sarcasm/       (6 .npz files)
└── llama_3_1_8b_sycophancy/    (6 .npz files)
```

Each `.npz` file (e.g., `activations_turn_boundary:-1.npz`) contains:
- `task_ids`: array of 2,500 task ID strings
- `layer_8`, `layer_12`, `layer_16`, `layer_20`, `layer_24`: arrays of shape `(2500, 4096)`

## Issues and fixes

### 1. LoRA adapter loading

The persona models in `maius/llama-3.1-8b-it-personas` are LoRA adapters, not full model weights. The extraction pipeline expected full models.

**Fix** (`src/models/huggingface_model.py`): Added `_is_lora_adapter()` detection that checks for `adapter_config.json`. When detected, loads the base model from the adapter config's `base_model_name_or_path`, applies the adapter via `PeftModel.from_pretrained()`, and merges with `merge_and_unload()`. Added `peft` as a dependency.

### 2. Chat template whitespace stripping

Llama 3.1's chat template strips trailing whitespace from user message content. 14/100 WildChat tasks had trailing spaces/newlines, causing `find_text_span()` to fail for the `task_mean` selector.

**Fix** (`src/models/huggingface_model.py`): `_get_task_span()` falls back to `.strip()` when raw content isn't found in the formatted text.

### 3. Misalignment model gated

`maius/llama-3.1-8b-it-misalignment` returned 403 GatedRepoError. The HuggingFace token lacks access to this gated repository. Skipped — 11/12 models extracted.

## Spec deviations

- The spec stated 30 `.npz` files per model (6 selectors x 5 layers). The actual pipeline produces 6 `.npz` files per model (1 per selector, with all 5 layers stored as separate keys inside each file). The data is equivalent.
- The misalignment model (1 of 12) was not extracted due to access restrictions.

## Reproduction

```bash
# All models (skips misalignment)
python scripts/extraction/run_all_characters.py

# Single model
python -m src.probes.extraction.run configs/extraction/character_probes/llama8b_sarcasm.yaml --resume

# Verify
python scripts/extraction/verify_outputs.py
```

## Resource usage

- GPU memory: 16.1 GB allocated (steady, same for base and merged LoRA models)
- Model load: ~2s (base) or ~5s (LoRA: load base + download adapter + merge)
- Extraction: ~8.5 min per model (79 batches of 32, ~6.3s/batch)
- Disk: 1.2 GB per model
