# Task-Mean Direction Steering

## Goal

Test whether the task_mean probe direction — trained on averaged task-token activations — produces effective preference steering when applied to those same task-token positions via differential steering. The task_mean probe was learned from mean activations over task tokens, so there is no train-steer site mismatch (unlike EOT/prompt_last probes, which are trained on single boundary tokens but steered across task spans). We test two layers (L25, L32) across a range of multipliers.

## What we reuse

- **Baseline:** 10,000 records from v2 followup (`experiments/revealed_steering_v2/followup/checkpoint.jsonl`, condition="baseline")
- **Pairs:** Same 500 pairs (`experiments/revealed_steering_v2/followup/pairs_500.json`)
- **Prior results for comparison:** EOT at ±0.03 (experiments/steering/eot_direction/) and prompt_last at ±0.03 (v2 followup)

## What we collect

500 pairs × 2 layers × 4 multipliers (±0.01, ±0.02, ±0.03, ±0.05) × 10 trials (5 per ordering) = 80,000 generations. Overnight run on H100.

## Probe directions

| Layer | Probe path | Heldout r | Mean norm |
|---|---|---|---|
| L25 | `results/probes/heldout_eval_gemma3_task_mean/probes/probe_ridge_L25.npy` | 0.803 | 38,349 |
| L32 | `results/probes/heldout_eval_gemma3_task_mean/probes/probe_ridge_L32.npy` | 0.797 | 40,966 |

Coefficients = mean_norm × multiplier:

| Multiplier | L25 coef | L32 coef |
|---|---|---|
| 0.01 | 384 | 410 |
| 0.02 | 767 | 819 |
| 0.03 | 1,151 | 1,229 |
| 0.05 | 1,918 | 2,048 |

For reference, EOT at ±0.03 used coef ≈ ±1,585 (L31, mean_norm=52,823).

## Script

`scripts/steering/run_task_mean_direction.py`

Run:
```bash
python -m scripts.steering.run_task_mean_direction [--resume] [--pilot N]
```

Checkpoints to `experiments/steering/task_mean_direction/checkpoint.jsonl`. Supports `--resume`.

### Key infrastructure to use

- **Steered client:** `src.steering.client.SteeredHFClient` — use `steering_mode="differential"` for pairwise prompts. Auto-detects task spans via `find_pairwise_task_spans` and applies `differential_steering`. Falls back to `all_tokens` if span detection fails. See `src/steering/client.py`.
- **Probe loading:** `src.probes.core.storage.load_probe_direction(manifest_dir, probe_id)` returns `(layer, unit_direction)`.
- **Mean norm:** `src.probes.core.activations.compute_activation_norms(activations_path, layers=[25, 32])`.
- **Prompt building:** `src.measurement.runners.runners.build_revealed_builder(template, response_format_name)` builds the prompt builder. Load template with `src.measurement.elicitation.prompt_templates.load_template("completion_preference")`.
- **Response parsing:** Prefix match first (`response.startswith("Task A:")` → "a", etc.), fall back to `src.measurement.elicitation.semantic_parser.parse_completion_choice_async` via OpenRouter for non-prefix responses. Load OpenRouter key from `.env`.
- **Model loading:** `src.models.huggingface_model.HuggingFaceModel("gemma-3-27b", max_new_tokens=256)`.
- **Generation:** `client.generate_n(messages, n=5, temperature=1.0, task_prompts=[task_a.prompt, task_b.prompt])` returns 5 responses. The `task_prompts` arg is required for differential steering to locate spans.
- **Layer switching:** Load model once. For each layer, create a new `SteeredHFClient` with the appropriate probe direction and layer. Use `with_coefficient()` to sweep multipliers without reloading.

### Pair loading

`experiments/revealed_steering_v2/followup/pairs_500.json` — list of dicts with keys: `pair_id`, `task_a` (id), `task_b` (id), `task_a_text`, `task_b_text`, `delta_mu`, `mu_a`, `mu_b`. Wrap task texts as `src.task_data.Task` objects.

### Checkpoint format

Same JSONL format as v2/EOT. Each line:
```json
{"pair_id": "pair_0042", "task_a_id": "...", "task_b_id": "...", "coefficient": 1150.5, "multiplier": 0.03, "layer": 25, "condition": "task_mean", "sample_idx": 3, "ordering": 0, "choice_original": "a", "choice_presented": "a", "raw_response": "Task A: ...", "delta_mu": 1.076, "steering_fallback": false}
```

Note: includes `layer` field (not present in prior experiments which used a single layer).

For `--resume`: load existing checkpoint, build a set of `(pair_id, layer, multiplier, ordering)` keys, skip any that already have 5 records.

### Execution order

Iterate layers in outer loop (minimizes steering hook swaps). Within each layer, iterate multipliers, then pairs, then orderings. This keeps the steering configuration stable for longest stretches.

## Analysis

Combine task_mean checkpoint with v2 followup checkpoint (baseline) and EOT/prompt_last data for comparison. Compute:

1. **Steering effect** per layer × multiplier: ordering-controlled shift in P(choose A), with bootstrap 95% CIs (n_boot=10,000)
2. **Dose-response curve:** Steering effect vs multiplier for each layer. Compare shape to EOT/prompt_last.
3. **Per-pair correlation:** Pearson r of per-pair effects between task_mean and EOT/prompt_last — do they steer the same pairs?
4. **Parse rates** across conditions — flag any multipliers where coherence degrades
5. **Layer comparison:** Does L25 (best probe r) outperform L32, or does the pattern differ?

### Plots (match EOT experiment format)

1. **Steering effect comparison** — bar chart: aggregate effect at each multiplier for L25, L32, with EOT ±0.03 and prompt_last ±0.03 as reference lines. Bootstrap 95% CIs.
2. **Dose-response** — line plot: steering effect vs multiplier for L25 and L32, with EOT/prompt_last reference points.
3. **Per-pair scatter** — EOT per-pair effect (x) vs task_mean per-pair effect (y) at the best-matching multiplier.

## Source data

- Pairs: `experiments/revealed_steering_v2/followup/pairs_500.json`
- V2 checkpoint (baseline): `experiments/revealed_steering_v2/followup/checkpoint.jsonl`
- EOT checkpoint: `experiments/steering/eot_direction/checkpoint.jsonl`
- Task-mean probes: `results/probes/heldout_eval_gemma3_task_mean/probes/probe_ridge_L{25,32}.npy`
- Activations (norm): `activations/gemma_3_27b_turn_boundary_sweep/activations_task_mean.npz`

## Output

- `experiments/steering/task_mean_direction/checkpoint.jsonl`
- `experiments/steering/task_mean_direction/task_mean_direction_report.md`
- `experiments/steering/task_mean_direction/assets/`

## GPU

1× H100 80GB. ~8–10 hours (8× the EOT experiment: 2 layers × 4 multipliers vs 1 layer × 1 multiplier).
