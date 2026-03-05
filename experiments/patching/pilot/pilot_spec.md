# Activation Patching Pilot

## Goal

Test whether swapping task-position activations between a pair of tasks flips the model's pairwise choice. This is a de-risking experiment: does activation patching at the task-token level causally determine which task gets chosen?

## Motivation

Steering with the L31 probe direction produced a ~17pp causal shift (revealed_steering_v2). This is real but modest. Activation patching is a stronger test: instead of nudging activations in a learned direction, we directly swap the activations at task A's and task B's token positions. If the model's choice tracks these swapped activations, it confirms that the information at those positions causally drives the decision — and tells us how much of the choice is determined by task representations vs other factors (position bias, instruction tokens, etc.).

## Design

### Task selection

10 tasks selected at evenly spaced utility quantiles (5th, 15th, ..., 95th percentile) from default persona Thurstonian scores. Utility range: mu=-8.7 to mu=+8.8.

- Source: `results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_80fa9dc8.csv`
- Selected tasks: `experiments/patching/pilot/selected_tasks.json`
- Selection script: `scripts/patching/precompute_baseline.py`

### Precompute baseline choice probabilities

Precomputed P(A>B) for all 45 pairs from Thurstonian scores using `default_preference_probability(mu_i, mu_j, sigma_i, sigma_j)`.

- Baseline probabilities: `experiments/patching/pilot/baseline_p_choose.json`
- 41/45 pairs are decisive (P>0.8 or P<0.2) — most pairs have large utility gaps since tasks are spread across quantiles

### Pairs

All C(10,2) = 45 pairs. Each pair run in both orderings (AB and BA) for position bias control. Total: 90 prompts.

### Conditions

For each prompt, three conditions:

1. **Baseline** — No patching, just run the model normally
2. **Last-token swap** — `swap_positions(last_token_of_A, last_token_of_B)` applied at all layers simultaneously
3. **Span swap** — `swap_spans(a_start, a_end, b_start, b_end)` applied at all layers simultaneously

### Resampling

Use `temperature=1.0` with **10 trials per prompt per condition** (5 per ordering). `get_logprobs` doesn't support hooks, so we can't get patched logprobs directly. Resampling gives us P(choose A) estimates under each condition.

Use `generate_with_hooks_n` to generate all 5 trials per prompt in a single forward pass (shared prefill):

```python
responses = model.generate_with_hooks_n(
    messages, layer_hooks, n=5, temperature=1.0, max_new_tokens=256
)
```

### Layers

Apply the swap hook at **all layers simultaneously** using `generate_with_hooks_n`. The hook factories (`swap_positions`, `swap_spans`) are in `src/models/base.py`. They follow the `SteeringHook` signature and are prefill-only.

```python
from src.models.base import swap_positions, swap_spans

# Last-token swap: same hook registered at every layer
hook = swap_positions(last_a, last_b)
layer_hooks = [(layer, hook) for layer in range(n_layers)]
responses = model.generate_with_hooks_n(
    messages, layer_hooks, n=5, temperature=1.0, max_new_tokens=256
)

# Span swap
hook = swap_spans(a_start, a_end, b_start, b_end)
layer_hooks = [(layer, hook) for layer in range(n_layers)]
responses = model.generate_with_hooks_n(
    messages, layer_hooks, n=5, temperature=1.0, max_new_tokens=256
)
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
- `temperature=1.0` with 10 trials per prompt (5 per ordering)
- `max_new_tokens=256`
- No system prompt (Gemma 3 27B has no default)

### Choice parsing

Use `CompletionChoiceFormat` from `src/measurement/elicitation/response_format.py`. This does prefix matching first ("Task A:"/"Task B:"), then falls back to `parse_completion_choice_async` (OpenRouter semantic parser) for ambiguous responses:

```python
from src.measurement.elicitation.response_format import CompletionChoiceFormat

fmt = CompletionChoiceFormat(task_a_prompt=task_a_text, task_b_prompt=task_b_text)
choice = fmt.parse(response)         # prefix match, returns "a"/"b"/None
if choice is None:
    choice = await fmt.parse_async(response)  # semantic fallback
```

## Data

### Model

Gemma 3 27B (bfloat16). Layers: 62 (indices 0–61).

### Thurstonian scores

`results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_80fa9dc8.csv` — mu and sigma for ~10k tasks.

### Probes (for reference, not used in patching)

`results/probes/gemma3_10k_heldout_std_raw/` — L31 best layer (r=0.86).

## Output

### Per-trial record

```json
{
  "pair_id": "pair_0042",
  "task_a_id": "wildchat_1234",
  "task_b_id": "alpaca_5678",
  "ordering": "AB",
  "condition": "baseline|last_token_swap|span_swap",
  "sample_idx": 3,
  "choice": "a|b|refusal",
  "raw_response": "Task A: To solve this..."
}
```

### Output files

- `experiments/patching/pilot/baseline_p_choose.json` — precomputed Thurstonian P(A>B) for all 45 pairs
- `experiments/patching/pilot/results.json` — all trials
- `experiments/patching/pilot/pilot_report.md` — analysis and plots
- `experiments/patching/pilot/assets/` — plots

### Checkpointing

Save to `experiments/patching/pilot/checkpoint.jsonl` after each pair. Support `--resume` by loading checkpoint and skipping completed pairs.

## Analysis

1. **Choice probability shift**: For each pair, compute P(choose A) under baseline, last-token swap, and span swap. Compare to precomputed Thurstonian P(A>B). Plot empirical P(A) vs Thurstonian P(A>B) for each condition.
2. **Flip rate by condition**: Across all pairs, what fraction show a significant shift in P(A) between baseline and patched? Use a binomial test per pair (10 trials).
3. **Direction of flips**: When patching shifts P(A), does it shift *toward* the other task (i.e., P(A) under swap ≈ 1 - P(A) under baseline)? This is the key test — random corruption would produce incoherent outputs, not systematic reversals.
4. **Coherence check**: Are patched responses still coherent completions, or gibberish? Manual inspection of a sample (or coherence judge if available).
5. **Span swap vs last-token swap**: Does swapping the full span produce larger shifts than swapping just the last token?
6. **Position bias**: Compare baseline AB vs BA choice rates. Does patching interact with position bias?
7. **Utility gap**: Do pairs with larger |Δmu| resist flipping more than borderline pairs? Plot shift magnitude vs |Δmu|.

### Success criteria

- If span swap reverses P(A) for >50% of pairs (i.e., P(A|swap) ≈ 1 - P(A|baseline)): strong evidence that task-position activations causally determine choice
- If 20–50%: partial causal role, other factors matter too
- If <20%: task-position activations are not the primary causal driver

## Trial budget

| Condition | Prompts | Trials/prompt | Total |
|-----------|---------|---------------|-------|
| Baseline | 90 | 5 | 450 |
| Last-token swap | 90 | 5 | 450 |
| Span swap | 90 | 5 | 450 |
| **Total** | | | **1,350** |

At ~2.5s per generation on A100 (but 5 trials share prefill via `generate_n`): ~270 forward passes × ~3s ≈ ~14 minutes.

## GPU requirements

- Gemma 3 27B in bfloat16 — ~54 GB VRAM → 1× A100 80GB or H100
- Patching overhead: negligible (tensor swap during prefill)
- Estimated runtime: ~20 minutes including tokenization and parsing

## Do NOT

- Do not invent new prompt templates — use the canonical `completion_preference` template
- Do not invent new response parsers — use `CompletionChoiceFormat` (prefix match + semantic fallback)
- Do not patch only a subset of layers — patch all 62 layers simultaneously to get the strongest possible signal first. Layer-selective patching is a follow-up
