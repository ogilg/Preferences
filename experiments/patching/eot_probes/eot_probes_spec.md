# EOT Probe Activations — Extraction Spec

## Goal

Extract activations at the `<end_of_turn>` token position across layers in the causal window (L25-40) for probe training. The pilot found the causal window at L25-34 with L31 best for probes (r=0.86) — but probes were trained on `prompt_last` (the `\n` after `model` in `<start_of_turn>model\n`). The patching results show `<end_of_turn>` carries more decision signal than the task block itself. Training probes on EOT activations may give better utility prediction.

## What's Needed

### 1. New selector: `eot`

The Gemma 3 chat template for prompt-only (no completion) is:

```
<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
```

`prompt_last` = `first_completion_idx - 1` = the `\n` after `model`. The `<end_of_turn>` token is at `first_completion_idx - 5` (tokens: `<end_of_turn>`, `\n`, `<start_of_turn>`, `model`, `\n`). But don't hardcode the offset — find `<end_of_turn>` by searching backwards from `first_completion_idx` in the token IDs.

Add a batched selector `eot` to `BATCHED_SELECTOR_REGISTRY` in `src/models/base.py`. It needs access to the input token IDs (not just activations), so this may require passing token IDs through to the selector, or implementing it as a special case in `HuggingFaceModel._apply_selectors`.

Approach: the simplest path is probably a model-level method that finds the EOT token index given the tokenized input, then a selector that uses that index. The EOT token ID can be looked up via `tokenizer.convert_tokens_to_ids("<end_of_turn>")`.

### 2. Extraction config

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

selectors: [eot]
layers_to_extract: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

batch_size: 32
save_every: 1000

output_dir: activations/gemma_3_27b_eot
```

Layers 25-40 covers the full causal window (25-34) plus a few layers beyond to see where signal dies off. These are absolute layer indices (integers), not fractional — the existing config supports both.

### 3. Run on RunPod

Same tasks as the main extraction (30k, same seed, same origins). This is a prompt-only extraction (no generation needed), so it's fast — just batched forward passes.

```bash
python -m src.probes.extraction.run configs/extraction/gemma3_27b_eot.yaml --resume
```

### 4. Sync back

After extraction completes on RunPod:

```bash
scp -r -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/Preferences/activations/gemma_3_27b_eot/ activations/gemma_3_27b_eot/
```

## Output

- `activations/gemma_3_27b_eot/activations_eot.npz` — (n_tasks, d_model) per layer, 16 layers
- `activations/gemma_3_27b_eot/extraction_metadata.json`
- `activations/gemma_3_27b_eot/completions_with_activations.json` — manifest with task IDs

## Analysis (after extraction)

Train probes on EOT activations using the existing pipeline:

```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/<eot_config>.yaml
```

Compare EOT vs `prompt_last` probe performance across layers. The hypothesis: if EOT encodes the decision, EOT probes should match or exceed `prompt_last` probes at L31, and may peak at a different layer within the causal window (e.g., L34 which had the highest single-layer flip rate).

## Implementation Notes

- The selector needs the tokenizer's EOT token ID. For Gemma 3 this is a special token. Verify with `tokenizer.convert_tokens_to_ids("<end_of_turn>")`.
- In batched mode, each sample may have the EOT at a different absolute position (different prompt lengths), but after left-padding the offset from `first_completion_idx` should be consistent across samples (same template suffix). Still, find it by searching rather than hardcoding.
- If the selector architecture makes it hard to pass token IDs, an alternative is to precompute EOT offsets and store them, but searching at extraction time is cleaner.
