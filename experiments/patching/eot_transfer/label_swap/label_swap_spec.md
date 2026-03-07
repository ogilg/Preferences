# EOT Label Swap — Spec

## Question

When the donor EOT comes from "Task A: X, Task B: Y" (model picks slot B / task Y), what happens if we patch into a recipient prompt where the labels stay the same but the content is swapped: "Task A: Y, Task B: X"?

Three possible outcomes:
1. **Label-following**: model says "Task B:" (follows the label the donor picked) → but now executes task X (wrong content)
2. **Position-following**: model picks the second slot regardless of label → says "Task B:" and executes task X
3. **Content-following**: model picks task Y wherever it is → says "Task A:" and executes task Y

Outcomes 1 and 2 are indistinguishable from stated labels alone (both say "Task B:"), but differ in whether the model executes X or Y. The completion judge distinguishes them.

## Context

This is a follow-up to the EOT transfer experiment. The parent experiment found:
- Control (same prompt): 83.6% flip rate
- Swap headers (Task 1/2): 75.1% flip rate — labels don't matter much
- Swap both tasks: 30.6% — substantial positional component

This condition fills a gap: same tasks, same labels, swapped content positions. It directly pits position vs label vs content.

## Design

Use the same 200 source orderings from `experiments/patching/eot_transfer/selected_orderings.json`.

### Single condition: label swap

For each source ordering where donor picks slot B (task Y):
- **Donor prompt**: "Task A: X, Task B: Y" (opposite ordering, as in parent experiment)
- **Recipient prompt**: "Task A: Y, Task B: X" (same labels, swapped task content)
- Run baseline (unpatched) + patched, 5 trials each, temperature 1.0, max_new_tokens=64
- Use `CompletionChoiceFormat` for stated label parsing
- Use completion judge for executed content classification

Both ordering directions (the recipient can be A:Y,B:X or A:X,B:Y — but A:X,B:Y is just the control from the parent experiment, so only run A:Y,B:X).

### Evaluation

For each trial record:
- **Stated label**: "Task A:" or "Task B:"
- **Executed task**: which task's content (X or Y) the model actually performs
- Compare to donor's choice to determine if model follows label, position, or content

## Model

Gemma 3 27B, same setup as parent experiment.

## Infrastructure

Reuse the donor cache infrastructure from `scripts/eot_transfer/run_transfer.py`. This is a single condition — ~200 orderings × 5 trials = 1,000 generations + 1,000 judge calls. Should take ~20 minutes.

## Output

- `experiments/patching/eot_transfer/label_swap/checkpoint.jsonl`
- `experiments/patching/eot_transfer/label_swap/label_swap_report.md` + `assets/`

## Do NOT

- Invent new prompt templates — use the same template as the parent experiment
- Skip the completion judge — stated labels alone cannot distinguish position-following from label-following
