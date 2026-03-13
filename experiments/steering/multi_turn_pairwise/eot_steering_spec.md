# Multi-Turn EOT Steering

## Goal

Steer preferences by intervening on the `<end_of_turn>` token (tb-5) in the multi-turn pairwise format.

## Background

The multi-turn format splits tasks across turns:

```
User:      "Choose which task you would prefer to complete.\n\nTask A:\n{task_a}\n\n{format_instruction}"
Assistant: "Got it, what's the other task?"
User:      "Task B:\n{task_b}"
Assistant: [generates completion of preferred task]
```

Validation results (200 pairs × 4 resamples): r = 0.53 with single-turn Thurstonian scores, P(A-position) = 0.504 (no position bias), 77% cross-order agreement.

We steer at the assistant-turn EOT — the last position that has seen Task A but not Task B. This is single-position steering (not differential): add `coefficient × direction` to the residual stream at one position.

## Parameters

| Parameter | Value |
|---|---|
| Model | `google/gemma-3-27b-it` |
| Steering layer | 32 |
| Probe | tb-5 Ridge, L32, heldout r = 0.868 |
| Temperature | 1.0 |
| max_new_tokens | 32 |
| Prefill | `"Got it, what's the other task?"` |
| Multipliers | `[-0.05, -0.03, -0.02, 0, 0.02, 0.03, 0.05]` |

Coefficients = multiplier × mean activation norm at L32 (from `suggest_coefficient_range()` in `src/steering/calibration.py`, reads cached norms from `extraction_metadata.json`).

### Pairs

500 pairs from the 10k run, stratified by |Δmu|: 100 borderline (|Δmu| < 1), 200 moderate (1 ≤ |Δmu| < 3), 200 decisive (|Δmu| ≥ 3).

### Measurement

Per pair × coefficient: 5 resamples per ordering (10 total), both orderings. 7 coefficients × 500 pairs × 10 trials = 35,000 generations (~3.2 hours on H100).

### Primary metric

**Steering effect** = P(choose high-mu task | coef=+C) − P(choose high-mu task | coef=−C), position-controlled.

## Implementation

### Key imports

```python
from src.steering.client import create_steered_client
from src.steering.calibration import suggest_coefficient_range
from src.models.base import position_selective_steering, find_eot_indices
from src.measurement.elicitation.prompt_templates import (
    MultiTurnRevealedPromptBuilder, PromptTemplate, TEMPLATE_TYPE_PLACEHOLDERS,
)
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.measurement.elicitation.measurer import RevealedPreferenceMeasurer
from src.measurement.storage.loading import load_run_utilities
from src.task_data import load_filtered_tasks, OriginDataset
```

Do not reimplement prompt building, response parsing, or coefficient calibration.

### Script: `scripts/multi_turn_pairwise/run_eot_steering.py`

1. Load probe direction from `results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L32.npy`
2. Calibrate coefficients: `suggest_coefficient_range(activations_path, layer=32, multipliers=[-0.05, -0.03, -0.02, 0, 0.02, 0.03, 0.05])`
3. Load Thurstonian scores via `load_run_utilities(run_dir)` and tasks via `load_filtered_tasks`
4. Sample 500 pairs stratified by |Δmu|
5. Create client: `create_steered_client("gemma-3-27b", layer=32, direction=direction, coefficient=0)`
6. For each trial: build prompt with `MultiTurnRevealedPromptBuilder`, find EOT position, steer with `position_selective_steering(tensor, eot_pos, eot_pos + 1)`, generate with `client.generate_with_hook(messages, hook)`, parse with `CompletionChoiceFormat`
7. Save each trial to JSONL checkpoint

### Finding the EOT position

1. `tokenizer.apply_chat_template(messages, tokenize=True)`
2. `find_eot_indices(token_ids, tokenizer)` → assistant-turn EOT is the second index

### Resume

Load existing `checkpoint.jsonl`, build set of `(pair_id, multiplier, ordering, resample_idx)` keys, skip completed.

### Analysis: `scripts/multi_turn_pairwise/analyze_eot_steering.py`

1. Dose-response curve: P(choose high-mu task) vs coefficient, bootstrap 95% CIs
2. By Δmu stratum: separate curves for borderline, moderate, decisive
3. Per-pair slopes: distribution of linear regression slopes
4. Parse rate table per coefficient

## Source data

Sync gitignored data to pod before running:

```bash
scp -r -P <PORT> -i ~/.ssh/id_ed25519 results/probes/heldout_eval_gemma3_tb-5/ root@<IP>:/workspace/Preferences/results/probes/heldout_eval_gemma3_tb-5/
scp -r -P <PORT> -i ~/.ssh/id_ed25519 activations/gemma_3_27b_turn_boundary_sweep/extraction_metadata.json root@<IP>:/workspace/Preferences/activations/gemma_3_27b_turn_boundary_sweep/
scp -r -P <PORT> -i ~/.ssh/id_ed25519 results/experiments/main_probes/gemma3_10k_run1/ root@<IP>:/workspace/Preferences/results/experiments/main_probes/gemma3_10k_run1/
```

| What | Path |
|---|---|
| Probe direction | `results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L32.npy` |
| Activation norms | `activations/gemma_3_27b_turn_boundary_sweep/extraction_metadata.json` |
| Thurstonian scores | `results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/` |

## Output

- `experiments/steering/multi_turn_pairwise/eot_steering/checkpoint.jsonl` (gitignored)
- `experiments/steering/multi_turn_pairwise/eot_steering/eot_steering_report.md`
- `experiments/steering/multi_turn_pairwise/eot_steering/assets/`

## GPU

1× H100 80GB.

## Success criteria

Assert in the analysis script (print PASS/FAIL):

1. **Monotonic dose-response:** Spearman correlation between multiplier and P(high-mu) > 0 (p < 0.05)
2. **Steering effect > 10pp** at the largest multiplier (±0.05)
3. **Gradient by difficulty:** Borderline steering effect > decisive steering effect
4. **Parse rates > 90%** at all coefficients
