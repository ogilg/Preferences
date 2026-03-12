# EOT Steering Open-Ended: Position-Specific Steering with EOT Probe

## Goal

Test whether steering with the EOT probe direction at specific token positions produces different open-ended generation effects. Two position-specific modes:

1. **autoregressive** — steer only the last token during generation (each new token gets steered, prompt is untouched)
2. **eot_position** — steer only the EOT token position during prefill (generation is untouched)

Pilot finding: all_tokens steering produces gibberish at ±0.10 multiplier, so it's excluded. eot_position stays coherent even at ±0.10 (only 1 token steered during prefill).

## Motivation

The original open-ended effects program used all_tokens steering with the prompt_last probe and found effects that weren't probe-specific (random directions produced comparable shifts). Two things are different here:

- **EOT probe** — trained on the token where the model "decides" it's done. May encode something about task satisfaction/engagement that the prompt_last probe doesn't.
- **Position-specific steering** — all_tokens steers everything, which may be too blunt. Steering only during generation (autoregressive) tests whether the preference signal needs to be present during text production. Steering only the EOT token tests whether perturbing the "decision point" token propagates forward.

## Design

### Probes

EOT Ridge probes at 5 layers spanning the available range:

- L25, L29, L31, L35, L39

All from `results/probes/heldout_eval_gemma3_eot/probes/probe_ridge_L{layer}.npy`

### Coefficients

Per-mode multipliers of mean activation norm at each layer:

- **autoregressive**: `[-0.07, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05, 0.07]` (11 values; ±0.10/0.15 produce gibberish)
- **eot_position**: `[-0.10, -0.07, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]` (13 values; coherent at ±0.10)

### Steering modes

1. **autoregressive** — steer only the last token (fires during generation, not prefill)
2. **eot_position** — find the last `<end_of_turn>` token in the formatted prompt, steer only that position during prefill

### Prompts (10)

Five categories of 2 prompts each. See `scripts/eot_steering_openended/generate.py` for the full list.

| Category | N | Purpose |
|----------|---|---------|
| **Introspective** | 2 | Self-referential framing (strongest effects in prior work) |
| **Enjoyment** | 2 | Task satisfaction, wanting more — novel category |
| **Creative** | 2 | Open-ended writing where tone/style can vary freely |
| **Neutral** | 2 | Factual questions (control — expect near-zero effects) |
| **Refusal** | 2 | Tasks near the refusal boundary (from BailBench/StressTest datasets) |

### Generation

- Temperature 1.0, max_new_tokens 512, seed 0
- 10 prompts x (11 + 13) coefficients x 5 layers = 1,200 generations
- Save all completions to `generation_results.json`

### Coherence evaluation

Manual inspection by the researcher. No automated judge.

## Infrastructure

- `create_steered_client` for all_tokens and autoregressive modes
- `generate_with_hook` with `position_selective_steering` for eot_position mode
- `find_eot_indices` or direct tokenizer search for locating EOT token

## Output

```
experiments/eot_steering_openended/
├── eot_steering_openended_spec.md
├── generation_results.json
└── assets/
```
