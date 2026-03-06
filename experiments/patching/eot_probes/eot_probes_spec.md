# EOT Probe Activations — Extraction Spec

## Goal

Extract activations at the `<end_of_turn>` token position across layers in the causal window (L25-40) for probe training. The pilot found the causal window at L25-34 with L31 best for probes (r=0.86) — but probes were trained on `prompt_last` (the `\n` after `model` in `<start_of_turn>model\n`). The patching results show `<end_of_turn>` carries more decision signal than the task block itself. Training probes on EOT activations may give better utility prediction.

## Status

### Done: `eot` selector

The `eot` selector is implemented and tested. It finds the `<end_of_turn>` token by searching backwards through input_ids from `first_completion_idx`.

- `find_eot_indices` in `src/models/base.py` — vectorized search over `(batch, seq_len)` token IDs
- Handled as a special case in `HuggingFaceModel._apply_selectors` (needs `input_ids`, unlike standard selectors)
- EOT token name per model stored in `ModelConfig.eot_token` in the registry (Gemma: `<end_of_turn>`, Llama: `<|eot_id|>`, Qwen: `<|im_end|>`)
- Config validation updated: `ALL_SELECTOR_NAMES` includes both standard and token-ID selectors
- Integration test confirms EOT is exactly 4 tokens before `prompt_last` in Gemma 3

### TODO: Extraction config

```yaml
model: gemma-3-27b
backend: huggingface

n_tasks: 30000
task_origins:
  - wildchat
  - alpaca
  - math
  - bailbench
  - stress_test
seed: 42

selectors: [eot, prompt_last]
layers_to_extract: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

batch_size: 32
save_every: 1000

output_dir: activations/gemma_3_27b_eot
```

Extracts both `eot` and `prompt_last` at L25-40 in one run. The existing `prompt_last` activations only have [15, 31, 37, 43, 49, 55] — we need per-layer `prompt_last` across the causal window for a fair comparison.

### TODO: Run on RunPod

Same tasks as the main extraction (30k, same seed, same origins). Prompt-only extraction (no generation), just batched forward passes.

```bash
python -m src.probes.extraction.run configs/extraction/gemma3_27b_eot.yaml --resume
```

### TODO: Sync back

```bash
rsync -avz -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" root@<IP>:/workspace/Preferences/activations/gemma_3_27b_eot/ activations/gemma_3_27b_eot/
```

## Output

- `activations/gemma_3_27b_eot/activations_eot.npz` — (n_tasks, d_model) per layer, 16 layers
- `activations/gemma_3_27b_eot/activations_prompt_last.npz` — same layers, for comparison
- `activations/gemma_3_27b_eot/extraction_metadata.json`
- `activations/gemma_3_27b_eot/completions_with_activations.json` — manifest with task IDs

## Analysis (after extraction)

Train probes on EOT activations using the existing pipeline:

```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/<eot_config>.yaml
```

Compare EOT vs `prompt_last` probe performance across layers. The hypothesis: if EOT encodes the decision, EOT probes should match or exceed `prompt_last` probes at L31, and may peak at a different layer within the causal window (e.g., L34 which had the highest single-layer flip rate).
