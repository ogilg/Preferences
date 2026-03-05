# Activation Patching Pilot

## Goal

Test whether swapping task-position activations between a pair of tasks flips the model's pairwise choice. This is a de-risking experiment: does activation patching at the task-token level causally determine which task gets chosen?

## Motivation

Steering with the L31 probe direction produced a ~17pp causal shift (revealed_steering_v2). This is real but modest. Activation patching is a stronger test: instead of nudging activations in a learned direction, we directly swap the activations at task A's and task B's token positions. If the model's choice tracks these swapped activations, it confirms that the information at those positions causally drives the decision — and tells us how much of the choice is determined by task representations vs other factors (position bias, instruction tokens, etc.).

## Design

### Task selection

Select 10 tasks uniformly spaced in utility from default persona Thurstonian scores:

- Source: `results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_80fa9dc8.csv`
- Sort tasks by mu, pick 10 at evenly spaced quantiles (5th, 15th, ..., 95th percentile)
- This gives pairs spanning the full utility range

### Pairs

All C(10,2) = 45 pairs. Each pair run in both orderings (AB and BA) for position bias control. Total: 90 prompts.

### Conditions

For each prompt, three conditions:

1. **Baseline** — No patching, just run the model normally
2. **Last-token swap** — `swap_positions(last_token_of_A, last_token_of_B)` applied at all layers simultaneously
3. **Span swap** — `swap_spans(a_start, a_end, b_start, b_end)` applied at all layers simultaneously

### Layers

Apply the swap hook at **all layers simultaneously** using `generate_with_multi_layer_steering`. The hook factories (`swap_positions`, `swap_spans`) are in `src/models/base.py`. They follow the `SteeringHook` signature and are prefill-only.

```python
from src.models.base import swap_positions, swap_spans

# Last-token swap: same hook registered at every layer
hook = swap_positions(last_a, last_b)
layer_hooks = [(layer, hook) for layer in range(n_layers)]
response = model.generate_with_multi_layer_steering(messages, layer_hooks, temperature=0, max_new_tokens=256)

# Span swap
hook = swap_spans(a_start, a_end, b_start, b_end)
layer_hooks = [(layer, hook) for layer in range(n_layers)]
response = model.generate_with_multi_layer_steering(messages, layer_hooks, temperature=0, max_new_tokens=256)
```

### Token positions

Use `find_pairwise_task_spans` from `src/steering/tokenization.py` to locate task A and task B token spans in the formatted prompt:

```python
from src.steering.tokenization import find_pairwise_task_spans

(a_start, a_end), (b_start, b_end) = find_pairwise_task_spans(
    tokenizer, formatted_prompt, task_a_text, task_b_text
)
last_a = a_end - 1
last_b = b_end - 1
```

### Template and generation

- Template: `completion_preference` — "You will be given two tasks. Choose one and complete it."
- `temperature=0` (greedy decoding) for determinism — each condition produces exactly one response per prompt
- `max_new_tokens=256`
- No system prompt (Gemma 3 27B has no default)

### Choice parsing

Use `parse_completion_choice_async` from `src/measurement/elicitation/semantic_parser.py` (OpenRouter LLM judge). Do NOT use string matching.

## Data

### Model

Gemma 3 27B (bfloat16). Layers: 62 (indices 0–61).

### Probes (for reference, not used in patching)

`results/probes/gemma3_10k_heldout_std_raw/` — L31 best layer (r=0.86).

### Activations (for task selection only)

Not needed — task selection uses Thurstonian scores, not activations.

## Output

### Per-trial record

```json
{
  "pair_id": "pair_0042",
  "task_a_id": "wildchat_1234",
  "task_b_id": "alpaca_5678",
  "ordering": "AB",
  "condition": "baseline|last_token_swap|span_swap",
  "choice": "a|b",
  "raw_response": "Task A: To solve this..."
}
```

### Output files

- `experiments/patching/pilot/results.json` — all trials
- `experiments/patching/pilot/pilot_report.md` — analysis and plots
- `experiments/patching/pilot/assets/` — plots

### Checkpointing

Save to `experiments/patching/pilot/checkpoint.jsonl` after each pair. Support `--resume` by loading checkpoint and skipping completed pairs.

## Analysis

1. **Flip rate by condition**: For each pair+ordering, does the patched choice differ from baseline? Report flip rate for last-token swap and span swap separately.
2. **Direction of flips**: When patching flips the choice, does it flip *toward* the other task (i.e., if baseline chose A, does swapping make it choose B)? This is the key test — random corruption would produce incoherent outputs, not systematic flips.
3. **Coherence check**: Are patched responses still coherent completions, or gibberish? Use the same coherence judge as revealed_steering_v2 if available, otherwise manual inspection of a sample.
4. **Span swap vs last-token swap**: Does swapping the full span produce more flips than swapping just the last token?
5. **Position bias**: Compare baseline AB vs BA choice rates. Does patching interact with position bias?
6. **Utility gap**: Do pairs with larger |Δmu| resist flipping more than borderline pairs?

### Success criteria

- If span swap flips >50% of pairs: strong evidence that task-position activations causally determine choice
- If 20–50%: partial causal role, other factors matter too
- If <20%: task-position activations are not the primary causal driver (could be instruction tokens, positional encoding, etc.)

## Trial budget

| Condition | Prompts | Trials | Total |
|-----------|---------|--------|-------|
| Baseline | 90 | 1 | 90 |
| Last-token swap | 90 | 1 | 90 |
| Span swap | 90 | 1 | 90 |
| **Total** | | | **270** |

At ~2.5s per generation on A100: ~11 minutes. This is intentionally tiny — it's a pilot.

## GPU requirements

- Gemma 3 27B in bfloat16 — ~54 GB VRAM → 1× A100 80GB or H100
- Patching overhead: negligible (tensor swap during prefill)
- Estimated runtime: ~15 minutes including tokenization and parsing

## Do NOT

- Do not use string matching for choice parsing — use `parse_completion_choice_async`
- Do not invent new prompt templates — use the canonical `completion_preference` template
- Do not run at temperature > 0 — we want deterministic responses for this pilot
- Do not patch only a subset of layers — patch all 62 layers simultaneously to get the strongest possible signal first. Layer-selective patching is a follow-up
