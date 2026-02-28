# MRA Activation Extraction

Extract activations for 3 persona conditions (2500 tasks each, 3 layers). The existing activations only cover 1500 tasks at 1 layer, so a full re-extraction is needed.

## What to run

```bash
python -m src.probes.extraction.run configs/extraction/mra_persona2_villain.yaml
python -m src.probes.extraction.run configs/extraction/mra_persona3_midwest.yaml
python -m src.probes.extraction.run configs/extraction/mra_persona4_aesthete.yaml
```

Run sequentially (they share GPU memory). Each extracts 2500 tasks at 3 layers, ~20 min on H100.

Do NOT use `--resume` — the existing activations only have layer 31, and we need layers 43 and 55 too. A full re-extraction is required.

## Configs

All 3 configs (`configs/extraction/mra_persona{2,3,4}_*.yaml`):

- `model`: gemma-3-27b
- `backend`: huggingface
- `activations_model`: gemma-3-27b (loads task prompts from existing baseline completions)
- `task_ids_file`: `configs/extraction/mra_all_2500_task_ids.txt`
- `system_prompt`: persona-specific (prepended as system message before user prompt)
- `selectors`: [prompt_last]
- `layers_to_extract`: [31, 43, 55]
- `output_dir`: `activations/gemma_3_27b_{villain,midwest,aesthete}/`

## Expected output

Each output dir will contain:

- `activations_prompt_last.npz` — keys: `task_ids` (2500,), `layer_31` (2500, 5376), `layer_43` (2500, 5376), `layer_55` (2500, 5376)
- `completions_with_activations.json` — prompts + generated completions
- `extraction_metadata.json` — config, timing, task count

Baseline activations (`activations/gemma_3_27b/`) already have all 2500 tasks at all required layers. Do not re-extract.

## Do NOT

- Do not use `--resume`. Existing activations are missing layers 43 and 55.
- Do not write custom extraction code. Use `src.probes.extraction.run` with the YAML configs.
- Do not extract baseline (no prompt) activations. Already complete.
