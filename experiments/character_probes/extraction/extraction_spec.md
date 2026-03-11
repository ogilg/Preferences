# Character Probes: Activation Extraction

Extract activations for base Llama 3.1 8B and 11 character-trained variants on the 2,500 MRA tasks.

## Models

12 models total. All are ~16GB in bf16, fit on a single A100.

| Model | HF path | Notes |
|-------|---------|-------|
| Base | `meta-llama/Llama-3.1-8B-Instruct` | In registry as `llama-3.1-8b` |
| Sarcasm | `maius/llama-3.1-8b-it-personas`, subfolder `sarcasm` | |
| Humor | `maius/llama-3.1-8b-it-personas`, subfolder `humor` | |
| Remorse | `maius/llama-3.1-8b-it-personas`, subfolder `remorse` | |
| Nonchalance | `maius/llama-3.1-8b-it-personas`, subfolder `nonchalance` | |
| Impulsiveness | `maius/llama-3.1-8b-it-personas`, subfolder `impulsiveness` | |
| Sycophancy | `maius/llama-3.1-8b-it-personas`, subfolder `sycophancy` | |
| Mathematical | `maius/llama-3.1-8b-it-personas`, subfolder `mathematical` | |
| Poeticism | `maius/llama-3.1-8b-it-personas`, subfolder `poeticism` | |
| Goodness | `maius/llama-3.1-8b-it-personas`, subfolder `goodness` | |
| Loving | `maius/llama-3.1-8b-it-personas`, subfolder `loving` | |
| Misalignment | `maius/llama-3.1-8b-it-misalignment` | Separate repo |

All character models share the base Llama 3.1 8B tokenizer. No system prompts — character is in the weights.

Character checkpoints are loaded via `subfolder` kwarg in `from_pretrained`, which downloads and caches automatically.

## Tasks

2,500 tasks from `configs/extraction/mra_all_2500_task_ids.txt`.

## Selectors

Llama 3.1 8B turn boundary structure (between user message end and assistant response start):

```
...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n
     tb-5        tb-4          tb-3       tb-2      tb-1
```

Extract all 5 turn boundary positions plus task_mean:

- `turn_boundary:-1` — final `\n` (equivalent to old `prompt_last`)
- `turn_boundary:-2` — `<|end_header_id|>`
- `turn_boundary:-3` — `assistant` (semantic role token)
- `turn_boundary:-4` — `<|start_header_id|>` (structural)
- `turn_boundary:-5` — `<|eot_id|>` (EOT marker)
- `task_mean` — mean across all user prompt tokens

We skip the dedicated `eot` selector because `turn_boundary:-5` already points to `<|eot_id|>` for single-turn, and `eot` requires a registry lookup that complicates loading unregistered models.

## Layers

Llama 3.1 8B has 32 layers. Extract at 5 evenly-spaced positions:

- L8 (25%), L12 (37.5%), L16 (50%), L20 (62.5%), L24 (75%)

## Extraction configs

12 YAML configs in `configs/extraction/character_probes/`:

- `llama8b_base.yaml`
- `llama8b_sarcasm.yaml` .. `llama8b_loving.yaml` (10 personas)
- `llama8b_misalignment.yaml`

## Run

On a GPU pod:

```bash
for config in configs/extraction/character_probes/*.yaml; do python -m src.probes.extraction.run "$config" --resume; done
```

## Output

```
activations/character_probes/
├── llama_3_1_8b_base/
├── llama_3_1_8b_sarcasm/
├── llama_3_1_8b_humor/
├── llama_3_1_8b_remorse/
├── llama_3_1_8b_nonchalance/
├── llama_3_1_8b_impulsiveness/
├── llama_3_1_8b_sycophancy/
├── llama_3_1_8b_mathematical/
├── llama_3_1_8b_poeticism/
├── llama_3_1_8b_goodness/
├── llama_3_1_8b_loving/
└── llama_3_1_8b_misalignment/
```

Each directory contains `.npz` files keyed by `{selector}_{layer}` with task IDs as array names.

Do NOT commit `activations/` — it is gitignored.

## Verification

Each model directory should contain 30 `.npz` files (6 selectors × 5 layers). A successful `--resume` run that skips all tasks means that model is done.
