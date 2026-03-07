# Label Swap — Running Log

## 2026-03-07: Setup

- Read parent experiment report and infrastructure
- Parent: donor = opposite ordering, control flip rate = 83.6%
- For label_swap, donor must be SOURCE ordering (not reversed) so that the label_swap recipient (reversed ordering) creates position-vs-content disagreement
- If donor = reversed ordering, label_swap recipient = donor prompt → patching is a no-op
- Using source ordering as donor: model picks slot B (baseline_dominant="b") or slot A ("a")
- Label swap recipient: tasks in opposite positions from source
- If position-following: patching pushes toward donor's slot → flip from baseline → high flip rate
- If content-following: patching pushes toward preferred task (wherever it is) → no flip → low flip rate

## 2026-03-07: Pilot (n=3)

- Pipeline validated: 2/3 orderings show flips, 1/3 no flip
- Observed dissociation in pilot: model says "Task B:" but executes slot A content
- Ordering 1: says "Task B" (position) but describes magnets (slot A content) — clear dissociation
- Ordering 2: says "Task B" but solves math from slot A — same pattern

## 2026-03-07: Full run (n=200)

- 200 orderings, 5 trials each, ~19 min generation, ~45 min judging
- Judge: gpt-5-nano, 2000 calls, 0 errors

### Results

- **Flip rate**: 87.5% [83.0%, 92.0%] (n=200)
- **Dissociation**: 38.2% (377/988) vs parent's 3.0%

Four-way classification (n=988 valid patched trials):
- Full position-following: 244 (24.7%)
- Full content-following: 367 (37.1%)
- Dissociation (label→pos, exec→content): 156 (15.8%)
- Dissociation (label→content, exec→pos): 221 (22.4%)

Label: 40.5% position, 59.5% content
Execution: 47.1% position, 52.9% content

### bd asymmetry
- bd="a" (105 orderings): 94.3% flip rate — content-driven (label flips to B where task_a is)
- bd="b" (95 orderings): 80.0% flip rate — position-driven (label flips to B = donor slot)
- Baseline on recipient is 97.9% "a" regardless of bd → model picks slot A in reversed ordering
- bd="a" orderings have 100% "a" baseline on recipient (position bias toward A in both source and recipient)
- Patching still flips bd="a" from "a" to "b" (94.3%) → content signal in EOT overrides position bias
