# EOT Label Swap — Spec

## Question

Does the EOT signal encode **label identity** ("pick Task A") or **prompt position** ("pick the first slot")?

The parent experiment's swap-headers condition showed that renaming "Task A/B" to "Task 1/2" barely affects the flip rate. But that only tests whether the EOT cares about the *specific* label tokens — it doesn't test whether the EOT tracks *which label* maps to which slot. Here we reverse the label ordering in the template: "Task B" appears first, "Task A" appears second.

## Design

### Template modification

The standard template is:
```
Task A:
{task_a}

Task B:
{task_b}
```

The label-swap template reverses the header ordering:
```
Task B:
{task_b}

Task A:
{task_a}
```

The format instruction changes from "Begin with 'Task A:' or 'Task B:'" accordingly. Use `CompletionChoiceFormat` with appropriate label configuration. The key change is that "Task B" is now the *first* slot and "Task A" is the *second* slot.

### Condition

Use the same 200 source orderings from `experiments/patching/eot_transfer/selected_orderings.json`.

For each source ordering where the donor EOT encodes "pick slot A" (i.e., "pick Task A"):

- **Recipient prompt**: same two tasks, but using the label-swap template where "Task B" comes first and "Task A" comes second
- **Donor**: same as parent experiment (reversed ordering, standard template)
- Run baseline (unpatched) + patched, 5 trials each, temperature 1.0, max_new_tokens=64

### Evaluation

- **Stated label**: "Task A:" or "Task B:" (via `CompletionChoiceFormat` parser)
- **Executed task**: which task's content the model performs (via completion judge)
- **Flip rate**: fraction of orderings where patched majority choice differs from baseline majority choice (same metric as parent experiment)
- **Label-following rate**: fraction picking the donor's label (Task A) regardless of position
- **Position-following rate**: fraction picking the donor's position (first slot) regardless of label

## Model

Gemma 3 27B (bfloat16), same setup as parent experiment.

## Infrastructure

Reuse donor cache from `scripts/eot_transfer/run_transfer.py`. Single condition — ~200 orderings x 5 trials = 1,000 generations + 1,000 judge calls. ~20 minutes on H100.

## Output

- `experiments/patching/eot_transfer/label_swap/checkpoint.jsonl`
- `experiments/patching/eot_transfer/label_swap/label_swap_report.md` + `assets/`

## Do NOT

- Invent new prompt templates beyond the label reordering described above
- Skip the completion judge — stated labels alone cannot distinguish all hypotheses
- Reuse donor EOT from label-swapped prompts — donor must use the standard template (same as parent experiment)
