# Label Swap Running Log

## Setup
- Experiment: EOT Label Swap
- Branch: research-loop/label_swap
- GPU: NVIDIA H100 80GB HBM3
- Model: Gemma 3 27B (bfloat16)

## Script Fixes
The existing `run_label_swap.py` had bugs relative to the spec:
1. Donor used SOURCE ordering (should be REVERSED ordering, standard template)
2. Recipient used standard template with reversed content (should use LABEL-SWAP template)
3. No filter for donor_slot="a" orderings
4. Parsing didn't use swapped labels

Fixed all four. Now:
- Donor: reversed ordering, standard template (Task A: task_b, Task B: task_a)
- Recipient: label-swap template (Task B: task_b, Task A: task_a)
- Filter to 95 orderings with donor_slot="a"
- Parse with label_a="Task B", label_b="Task A"

## Data
- 200 total orderings in selected_orderings.json
- 95 with donor_slot="a" (baseline_dominant="b")
- 105 with donor_slot="b" (baseline_dominant="a")

## Pilot (5 orderings)
All 25 trials picked first slot in both baseline and patched. Template working correctly.
Donor template: "Task A: task_b / Task B: task_a" (reversed, standard labels)
Recipient template: "Task B: task_b / Task A: task_a" (label-swap)

## Full Run (95 orderings)
Duration: 8.7 minutes on H100.

Parse-level results:
- Baseline: 470 "a" (first slot), 5 "b" (second slot)
- Patched: 468 "a" (first slot), 7 "b" (second slot)
- 3 orderings with any variation between baseline/patched

Flip rate: 2/95 = 2.1%
- idx=11: position-following flip (baseline picked second slot → patched picks first slot)
- idx=68: label-following flip (baseline picked first slot → patched picks "Task A" second slot, then refuses)
- idx=57: partial shift (2/5 patched trials pick "Task A" second slot, no majority flip)

## Judge Run
950 judge calls, 0 errors, ~3 min.

Judge results:
- Stated vs Executed: 100% agreement (no dissociation)
- Patched refusals: 5 (all from idx=68, bailbench content)
- Baseline stated: b=470, a=5 (judge "b" = Task B content = first slot)
- Patched stated: b=468, a=7

Key result: EOT does NOT encode label identity. 98.5% of patched trials pick the first slot (same as baseline), regardless of the "Task A" label being in the second slot.
