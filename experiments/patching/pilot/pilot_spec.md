# Activation Patching Pilot

## Goal

Test whether swapping task-position activations between a pair of tasks flips the model's pairwise choice. This is a de-risking experiment: does activation patching at the task-token level causally determine which task gets chosen?

## Motivation

Steering with the L31 probe direction produced a ~17pp causal shift (revealed_steering_v2). This is real but modest. Activation patching is a stronger test: instead of nudging activations in a learned direction, we directly swap the activations at task A's and task B's token positions. If the model's choice tracks these swapped activations, it confirms that the information at those positions causally drives the decision — and tells us how much of the choice is determined by task representations vs other factors (position bias, instruction tokens, etc.).

## Data

- **Model**: Gemma 3 27B (bfloat16), 62 layers (0–61)
- **Selected tasks**: `experiments/patching/pilot/selected_tasks.json` — 10 tasks at evenly spaced utility quantiles (mu=-8.7 to +8.8), includes task prompts
- **Baseline P(A>B)**: `experiments/patching/pilot/baseline_p_choose.json` — precomputed from Thurstonian scores, 41/45 pairs decisive. Contains task prompts, both `p_a_over_b` and `p_b_over_a`
- **Probes** (for reference): `results/probes/gemma3_10k_heldout_std_raw/` — L31 best layer (r=0.86)

## Design

### Pairs

All C(10,2) = 45 pairs. Each pair in both orderings (AB and BA). Total: 90 prompts.

### Conditions

1. **Baseline** — no patching
2. **Last-token swap** — `swap_positions(last_token_of_A, last_token_of_B)` at all layers
3. **Span swap** — `swap_spans(a_start, a_end, b_start, b_end)` at all layers

### Generation

- Template: `completion_preference`
- `temperature=1.0`, 5 trials per ordering per condition (10 total per pair per condition)
- `max_new_tokens=16`
- Use `generate_with_hooks_n` for shared prefill across trials
- Token positions via `find_pairwise_task_spans` from `src/steering/tokenization.py`
- Hook factories (`swap_positions`, `swap_spans`) from `src/models/base.py`

### Choice parsing

`CompletionChoiceFormat` from `src/measurement/elicitation/response_format.py` — prefix match, then semantic fallback via OpenRouter.

## Output

- `experiments/patching/pilot/results.json` — all trials
- `experiments/patching/pilot/checkpoint.jsonl` — per-pair checkpointing, supports `--resume`
- `experiments/patching/pilot/pilot_report.md` + `assets/`

## Analysis

1. **Choice probability shift** — P(choose A) under each condition vs Thurstonian baseline
2. **Flip rate** — fraction of pairs with significant shift (binomial test, 10 trials)
3. **Direction of flips** — does P(A|swap) ≈ 1 - P(A|baseline)? (systematic reversal vs random corruption)
4. **Span vs last-token** — does full span produce larger shifts?
5. **Position bias** — AB vs BA baseline rates, interaction with patching
6. **Utility gap** — do large |Δmu| pairs resist flipping?

### Interpretation

- **>50% pairs flip**: task-position activations causally determine choice
- **20–50%**: partial causal role, other factors matter too
- **<20%**: task-position activations are not the primary causal driver

## Budget

90 prompts × 3 conditions × 5 trials = 1,350 generations. ~270 forward passes with shared prefill. With `max_new_tokens=16`, ~5 min on A100.

## Do NOT

- Invent new prompt templates — use `completion_preference`
- Invent new response parsers — use `CompletionChoiceFormat`
- Patch only a subset of layers — all 62 simultaneously. Layer-selective is a follow-up
