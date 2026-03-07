# EOT Transfer Patching — Spec

## Question

Does the EOT token encode a **positional choice** ("pick slot A/B") or **task identity** ("do this specific task")?

The scaled EOT patching experiment showed that patching the EOT token from a donor prompt flips the model's choice 57% of the time. But in that experiment, the recipient prompt always contains the same two tasks — so we can't distinguish whether the EOT encodes "pick the task in position A" vs "pick task X specifically."

## Design

Collect EOT activations from donor prompts where the model picks a known slot (A or B). Patch these into **recipient prompts where the tasks have been swapped**, and observe what the model does. If EOT encodes a positional signal, the model should follow the slot. If it encodes task identity, it should follow the task (or get confused).

### Source data

Use the Phase 1 results from `experiments/patching/eot_scaled/`:
- 100 tasks with Thurstonian scores
- 9,611 orderings with known baseline choices
- Focus on **deterministic orderings** (max P >= 0.9 in baseline, which is 98% of them)

### Task pool for swaps

Draw replacement tasks from the same 100-task pool. For each trial, the replacement task should be different from both original tasks. Include a mix of:
- Same-topic replacements (matched by topic from `data/topics/topics.json`)
- Cross-topic replacements
- Harmful task swaps (where one task is from stresstest/bailbench)

### Conditions

For each source ordering (task X in slot A, task Y in slot B, model picks slot B at baseline):

1. **Control — same prompt** (replication): Patch donor EOT into the same recipient prompt (AB ordering patched with BA donor, as in scaled experiment). Confirms the flip still works. ~200 orderings.

2. **Swap both tasks**: Replace both X and Y with new tasks C and D. Patch EOT that encoded "pick slot B" into this new prompt. Does the model pick slot B (positional) or get confused?
   - Run both ordering directions: donor from AB patched into CD prompt, and donor from BA patched into DC prompt.
   - ~400 orderings (200 source orderings x 2 directions).

3. **Swap target task**: Replace only the task the donor EOT was pushing toward. If donor says "pick slot B" and slot B contained task Y, replace Y with task C. Does the model still go to slot B and do task C?
   - Both directions.
   - ~400 orderings.

4. **Swap headers only**: Keep same tasks X and Y, but swap labels — "Task A: Y, Task B: X" becomes "Task 1: Y, Task 2: X" (rename headers) and update the format instruction from "Begin with 'Task A:'" to "Begin with 'Task 1:'". Tests whether EOT cares about the literal label tokens.
   - Both directions.
   - ~200 orderings.

### Ordering bias control

For conditions 2 and 3, run **both ordering directions** of the recipient prompt. E.g., if the donor encoded "pick slot B" from an (X_A, Y_B) prompt:
- Recipient 1: (C_A, D_B) — does model pick slot B?
- Recipient 2: (D_A, C_B) — does model pick slot B?

This controls for any position bias in the recipient prompt. If the model follows the donor's slot regardless of recipient ordering, that's strong evidence for positional encoding.

### EOT token identification

Use `find_eot_indices()` from `src/models/base.py` to locate the `<end_of_turn>` token dynamically. Patch the EOT token and the `\n` that follows it (2 tokens total, same as scaled experiment). Since recipient prompts have different lengths, positions will differ — `find_eot_indices` handles this.

### Patching mechanics

Same as scaled experiment:
1. Forward pass on **donor prompt** (opposite ordering), cache residuals at EOT positions across all layers
2. Forward pass on **recipient prompt**, inject cached residuals at recipient's EOT positions

Patch all 62 layers (all-layer patching). Per-layer breakdown is out of scope for this pilot.

### Generation

- Temperature: 1.0
- max_new_tokens: 64 (enough for the judge to identify which task is being executed)
- N trials per ordering per condition: 5
- Parse with `CompletionChoiceFormat` for stated label
- Use `completion_judge.py` (async, batched) for executed content classification

### Evaluation

For each trial, record:
- **Stated label**: which slot label the model outputs ("Task A:" / "Task B:")
- **Executed task**: which task's content the model actually performs (via LLM judge)
- **Followed donor slot**: did the model pick the same slot the donor was pushing toward?
- **Followed donor task**: did the model execute the same task the donor was executing?

Primary metrics:
- **Slot following rate**: fraction of trials where model picks the donor's slot (regardless of what's in that slot)
- **Task following rate**: fraction of trials where model executes the donor's task (regardless of which slot it's now in)
- **Confusion rate**: fraction where model does neither the donor's slot nor the donor's task

### Sample size

Pilot scale — smaller than the full scaled experiment:
- 200 source orderings (sampled from deterministic flipping orderings in Phase 1)
- ~1,200 total recipient orderings across all conditions
- ~6,000 generations + ~6,000 judge calls

## Model

Gemma 3 27B (bfloat16), same as scaled experiment.

## Template

Same `completion_preference` template as all prior patching experiments:

```
You will be given two tasks. Choose one and complete it.
{format_instruction}

Task A:
{task_a}

Task B:
{task_b}
```

For the **swap headers** condition, use `CompletionChoiceFormat(task_a_label="Task 1", task_b_label="Task 2")` — the abstraction already supports custom labels, which propagates to both the format instruction and the parser.

## Output

- `experiments/patching/eot_transfer/selected_orderings.json` — 200 source orderings with donor info
- `experiments/patching/eot_transfer/checkpoint.jsonl` — one line per recipient ordering (supports --resume)
- `experiments/patching/eot_transfer/results.json` — all trial-level data
- `experiments/patching/eot_transfer/judge_results.json` — completion judge classifications
- `experiments/patching/eot_transfer/eot_transfer_report.md` + `assets/`

## Analysis

1. **Slot following rate by condition** — bar chart across all 4 conditions. If EOT is positional, conditions 2-3 should show high slot following. If task-identity, should drop.
2. **Task following rate by condition** — complement to slot following. In swap-both (condition 2), task following should be near zero if positional (the original tasks aren't even in the prompt).
3. **Stated vs executed dissociation** — same matrix as flip quality analysis. Does swapping tasks increase the label/content dissociation?
4. **Effect of topic match** — does slot following change when the replacement task is same-topic vs cross-topic?
5. **Harmful task interactions** — does the pattern change when harmful tasks are involved?
6. **Ordering direction** — are results symmetric across recipient orderings (position bias check)?

## Budget

~6,000 model generations at max_new_tokens=64. At ~0.5s each on H100, ~1 hour. Plus ~6,000 judge calls (cheap, async via OpenRouter).

## Do NOT

- Invent new prompt templates — use `completion_preference` via `CompletionChoiceFormat`
- Use hardcoded EOT positions — use `find_eot_indices()` to locate dynamically
- Skip the completion judge — stated labels alone are insufficient (20% label-only flips in scaled experiment)
- Aggregate before saving — store all trial-level data
- Skip checkpointing — use `--resume`
