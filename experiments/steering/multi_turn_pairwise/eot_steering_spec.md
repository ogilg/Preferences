# Multi-Turn Differential EOT Steering

## Goal

Steer preferences via differential steering on the two user-turn `<end_of_turn>` tokens in the multi-turn pairwise format.

## Background

The multi-turn format splits tasks across turns:

```
User:      "Choose which task you would prefer to complete.\n\nTask A:\n{task_a}\n\n{format_instruction}"
                                                                                    ← user EOT #1
Assistant: "Got it, what's the other task?"
User:      "Task B:\n{task_b}"
                                    ← user EOT #2
Assistant: [generates completion of preferred task]
```

The tb-5 probe was trained on user-turn EOT tokens (the `<end_of_turn>` just before `<start_of_turn>model`). There are two such tokens here: one after the Task A turn and one after the Task B turn.

Validation results (200 pairs × 4 resamples): r = 0.53 with Thurstonian scores, P(A-position) = 0.504 (no position bias), 77% cross-order agreement.

## Design

### Steering mechanism

Differential steering at two user EOT positions:
- **+direction** at user EOT #1 (after Task A turn) — bias toward "higher preference" for Task A
- **-direction** at user EOT #2 (after Task B turn) — bias toward "lower preference" for Task B

Use `differential_steering(tensor, eot1, eot1+1, eot2, eot2+1)` from `src/models/base.py`.

### Ordering

Each pair is tested in both orderings:
- **Ordering 0 (AB):** Task A in turn 1, Task B in turn 2. Steer +direction at EOT after Task A, -direction at EOT after Task B.
- **Ordering 1 (BA):** Task B in turn 1, Task A in turn 2. Steer +direction at EOT after Task B, -direction at EOT after Task A.

The steering always follows presentation order (+direction on first task's EOT, -direction on second task's EOT). Map choices back to original task identities after generation.

### Conditions

Five multiplier magnitudes plus control:
- **+0.02, +0.03, +0.05:** +direction on first task's EOT, -direction on second task's EOT
- **-0.02, -0.03, -0.05:** -direction on first task's EOT, +direction on second task's EOT
- **0:** no steering

## Parameters

| Parameter | Value |
|---|---|
| Model | `google/gemma-3-27b-it` |
| Steering layer | 32 |
| Probe | tb-5 Ridge, L32, heldout r = 0.868 |
| Temperature | 1.0 |
| max_new_tokens | 32 |
| Prefill | `"Got it, what's the other task?"` |
| Multipliers | `[-0.05, -0.03, -0.02, 0, +0.02, +0.03, +0.05]` |

Coefficients = multiplier × mean activation norm at L32 (from `suggest_coefficient_range()` in `src/steering/calibration.py`, reads cached norms from `extraction_metadata.json`).

### Pairs

500 pairs from the 10k run, stratified by |Δmu|: 100 borderline (|Δmu| < 1), 200 moderate (1 ≤ |Δmu| < 3), 200 decisive (|Δmu| ≥ 3).

### Measurement

Per pair × condition × ordering: 5 resamples. 7 conditions × 500 pairs × 2 orderings × 5 resamples = 35,000 generations.

### Primary metrics

- **Ordering bias** per condition: P(choose A-position task | AB) - P(choose A-position task | BA). At baseline (coef=0) this should be near 0. Positive steering should increase it, negative should decrease/reverse it.
- **Steering effect** per multiplier magnitude = P(choose high-mu | +m) − P(choose high-mu | −m), position-controlled (averaged over both orderings).

## Implementation

### Key imports

```python
from src.steering.client import create_steered_client
from src.steering.calibration import suggest_coefficient_range
from src.models.base import differential_steering, find_eot_indices
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
6. **Outer loop is condition (multiplier), inner loops are pairs × orderings × resamples.** This lets us check intermediate results after each condition completes. For each condition × pair × ordering × resample:
   - Build prompt with `MultiTurnRevealedPromptBuilder`. For ordering=0 pass (task_a, task_b), for ordering=1 pass (task_b, task_a).
   - Find both user EOT positions via `find_eot_indices` — these are the 1st and 3rd EOT indices (user turns), not the 2nd (assistant turn).
   - Create hook: `differential_steering(tensor, eot1, eot1+1, eot2, eot2+1)`
   - For negative condition, swap the signs: `differential_steering(tensor, eot2, eot2+1, eot1, eot1+1)`
   - Generate: `client.generate_with_hook(messages, hook)`
   - Parse with `CompletionChoiceFormat`. Map `choice_presented` back to `choice_original` (flip a↔b when ordering=1).
7. Save each trial to JSONL checkpoint

### Finding the user EOT positions

1. `tokenizer.apply_chat_template(messages, tokenize=True)` on all 3 messages (both user turns + assistant prefill)
2. `find_eot_indices(token_ids, tokenizer)` returns all EOT positions
3. User EOT #1 = first EOT index, User EOT #2 = third EOT index (second is the assistant turn)

### Resume

Load existing `checkpoint.jsonl`, build set of `(pair_id, condition, ordering, resample_idx)` keys, skip completed.

### Analysis: `scripts/multi_turn_pairwise/analyze_eot_steering.py`

1. **Dose-response curve:** P(choose high-mu) vs multiplier, with bootstrap 95% CIs
2. **Ordering bias per condition:** P(A-position | AB) - P(A-position | BA) for each of the 7 conditions
3. **Steering effect per magnitude:** P(high-mu | +m) − P(high-mu | −m) for m ∈ {0.02, 0.03, 0.05}
4. **By Δmu stratum:** Steering effect for borderline, moderate, decisive
5. **Parse rate table** per condition

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
2. **Steering effect > 10pp** at any multiplier magnitude
3. **Ordering bias shifts:** Positive conditions have higher ordering bias than control, negative conditions have lower
4. **Parse rates > 90%** at all conditions
