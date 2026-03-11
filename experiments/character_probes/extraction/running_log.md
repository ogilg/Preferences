# Extraction Running Log

## Setup
- GPU: NVIDIA A100-SXM4-80GB
- 12 models, 2500 tasks, 6 selectors, 5 layers = 30 .npz files per model
- All selectors are prompt-only (turn_boundary + task_mean) → batched forward pass, no generation
- Configs verified: all 12 present in `configs/extraction/character_probes/`

## Pilot (5 tasks, base model)
- Pipeline: OK (batched extraction path)
- Output: 6 .npz files, each with 5 layers, shape (5, 4096)
- GPU mem: 16.1GB allocated for model
- Time: ~3s for 5 tasks (model load ~50s)

## Issues encountered

### 1. task_mean text span failure
- Chat template strips trailing whitespace from user content, but `find_text_span` searches for raw prompt (with trailing whitespace)
- 14/100 tasks failed (all wildchat with trailing spaces/newlines)
- Fix: `_get_task_span` in `huggingface_model.py` now tries `.strip()` fallback when raw content not found

### 2. LoRA adapter models
- `maius/llama-3.1-8b-it-personas` subfolders contain LoRA adapters (adapter_model.safetensors), not full model weights
- `AutoModelForCausalLM.from_pretrained` with subfolder fails because there's no model.safetensors
- Fix: Added `_is_lora_adapter()` detection + PEFT loading in `HuggingFaceModel.__init__`
  - Downloads adapter_config.json to check if it's a LoRA adapter
  - Reads base_model_name_or_path from adapter config
  - Loads base model, applies adapter via `PeftModel.from_pretrained`, then `merge_and_unload()`
  - Installed `peft==0.18.1` via uv

### 3. Misalignment model gated
- `maius/llama-3.1-8b-it-misalignment` is gated, HF token doesn't have access
- Proceeding with 11/12 models (base + 10 personas), noting misalignment as inaccessible

## Full extraction

### Base model
- 2500 tasks, 0 failures, 0 OOMs
- Time: ~8.5 min (batched forward pass)
- GPU: 16.1GB alloc (steady)

### Persona models (10 LoRA adapters)
- All 10 completed successfully: goodness, humor, impulsiveness, loving, mathematical, nonchalance, poeticism, remorse, sarcasm, sycophancy
- Each: 2500 tasks, 0 failures, 0 OOMs
- Time per model: ~8.5 min extraction + ~5s model load/adapter merge
- GPU: 16.1GB alloc (steady), identical to base model after merge_and_unload()

## Verification
- 11/11 models passed verification
- 66 .npz files total (6 per model × 11 models)
- All shapes correct: (2500, 4096) per layer, 5 layers per file
- Disk usage: 1.2 GB per model, ~13 GB total
- Misalignment model skipped (gated repo, 403)
