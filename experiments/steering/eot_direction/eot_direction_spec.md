# EOT vs Prompt-Last Direction Steering

## Goal

Test whether the EOT probe direction produces stronger preference steering effects than the prompt_last probe direction. Same differential steering mechanism, same pairs — only the direction vector changes.

## What we reuse

The v2 followup (`experiments/revealed_steering_v2/followup/checkpoint.jsonl`) already has baseline and prompt_last data for all 500 pairs:

- **Baseline:** 10,000 records (500 pairs × 20 trials, condition="baseline")
- **Prompt_last at ±0.03:** 10,000 records (condition="probe", peak steering effect +0.173 at +0.03)

## What we collect

EOT condition only: 500 pairs × 2 multipliers (±0.03) × 10 trials (5 per ordering) = 10,000 generations. ~2.5 hours on H100.

## Probe directions

| Condition | Probe path | Heldout r |
|---|---|---|
| prompt_last | `results/probes/gemma3_10k_heldout_std_raw/probes/probe_ridge_L31.npy` | 0.866 |
| eot | `results/probes/heldout_eval_gemma3_eot/probes/probe_ridge_L31.npy` | 0.867 |

Both L31 Ridge. Coefficient = mean_norm × ±0.03, where mean_norm ≈ 52,823 (same norm for both so perturbation magnitude is matched).

## Script

`scripts/steering/run_eot_direction.py` — standalone script. Must NOT import from deleted `run_experiment.py`.

Run:
```bash
python -m scripts.steering.run_eot_direction
```

Checkpoints to `experiments/steering/eot_direction/checkpoint.jsonl`. Supports `--resume`.

### Key infrastructure to use

- **Steered client:** `src.steering.client.SteeredHFClient` — wraps `HuggingFaceModel` with a steering direction. Use `steering_mode="differential"` for pairwise prompts. It auto-detects task spans via `find_pairwise_task_spans` and applies `differential_steering`. Falls back to `all_tokens` if span detection fails (catch `ValueError`). See `src/steering/client.py`.
- **Probe loading:** `src.probes.core.storage.load_probe_direction(manifest_dir, probe_id)` returns `(layer, unit_direction)`. Works with both probe directories.
- **Mean norm:** `src.probes.core.activations.compute_activation_norms(activations_path, layers=[31])` returns `{31: mean_norm}`.
- **Prompt building:** `src.measurement.runners.runners.build_revealed_builder(template, response_format_name)` builds the prompt builder. Load template with `src.measurement.elicitation.prompt_templates.load_template("completion_preference")`.
- **Response parsing:** Use prefix match first (`response.startswith("Task A:")` → "a", etc.), fall back to `src.measurement.elicitation.semantic_parser.parse_completion_choice_async` via OpenRouter for non-prefix responses. Load OpenRouter key from `.env`.
- **Model loading:** `src.models.huggingface_model.HuggingFaceModel("gemma-3-27b", max_new_tokens=256)`.
- **Generation:** `client.generate_n(messages, n=5, temperature=1.0, task_prompts=[task_a.prompt, task_b.prompt])` returns 5 responses. The `task_prompts` arg is required for differential steering to locate spans.

### Pair loading

`experiments/revealed_steering_v2/followup/pairs_500.json` is a list of dicts with keys: `pair_id`, `task_a` (id), `task_b` (id), `task_a_text`, `task_b_text`, `delta_mu`, `mu_a`, `mu_b`. Wrap task texts as `src.task_data.Task` objects.

### Checkpoint format

Same JSONL format as v2. Each line:
```json
{"pair_id": "pair_0042", "task_a_id": "...", "task_b_id": "...", "coefficient": 1584.7, "multiplier": 0.03, "condition": "eot", "sample_idx": 3, "ordering": 0, "choice_original": "a", "choice_presented": "a", "raw_response": "Task A: ...", "delta_mu": 1.076, "steering_fallback": false}
```

For `--resume`: load existing checkpoint, build a set of `(pair_id, multiplier, ordering)` keys, skip any that already have 5 records.

## Analysis

Combine EOT checkpoint with v2 followup checkpoint (filter to baseline + mult=±0.03). Compute:

1. **Ordering difference** at ±0.03 for both conditions, with bootstrap 95% CIs
2. **Derived steering effect:** (ordering_diff − baseline_diff) / 2. Compare to prompt_last peak of +0.173
3. **Per-pair correlation:** Pearson r of per-pair effects (EOT vs prompt_last) — do they steer the same pairs?
4. **Parse rates** across conditions

## Source data

- Pairs: `experiments/revealed_steering_v2/followup/pairs_500.json`
- V2 checkpoint: `experiments/revealed_steering_v2/followup/checkpoint.jsonl`
- EOT probe: `results/probes/heldout_eval_gemma3_eot/probes/probe_ridge_L31.npy`
- Activations (norm): `activations/gemma_3_27b/activations_prompt_last.npz`

## Output

- `experiments/steering/eot_direction/checkpoint.jsonl`
- `experiments/steering/eot_direction/eot_direction_report.md`
- `experiments/steering/eot_direction/assets/`

## GPU

1× H100 80GB (or A100 80GB). ~2.5 hours.
