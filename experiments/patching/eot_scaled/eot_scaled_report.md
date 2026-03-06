# EOT Scaled Patching — Report

**Status: IN PROGRESS.** Phase 1 running via nohup. Phases 2 and 3 will run automatically. This report will be updated when all phases complete.

## Summary

Scaling the pilot's EOT patching experiment from 10 tasks (45 pairs) to 100 tasks (4,950 pairs). Preliminary Phase 1 results (800/9,900 orderings) confirm the pilot's key finding: patching just 2 tokens (`<end_of_turn>` + `\n`) from the opposite ordering flips the model's choice in **62%** of orderings (pilot: 54%).

| Metric | Pilot (10 tasks) | Scaled (interim, 100 tasks) |
|--------|------------------|-----------------------------|
| Tasks | 10 | 100 |
| Pairs | 45 | 4,950 |
| Orderings analyzed | 90 | 717 |
| All-layer EOT flip rate | 54% | 62% |
| P(choose position A) | 0.591 | 0.475 |

## Setup

| Parameter | Value |
|-----------|-------|
| Model | Gemma 3 27B (bfloat16), 62 layers |
| Tasks | 100 at evenly spaced utility quantiles (mu: -10.0 to +10.0) |
| Pairs | 4,950 canonical (all C(100,2)), each in AB and BA ordering |
| Phase 1 trials | 10 per ordering per condition |
| Phase 2/3 trials | 5 per ordering per condition |
| Temperature | 1.0 |
| max_new_tokens | 16 |
| Template | completion_preference (CompletionChoiceFormat) |
| EOT tokens patched | 2 (`<end_of_turn>` + `\n`, positions -5 and -4 from prompt end) |

## Phase 1: Baseline + All-Layer EOT Patch (interim)

**Data: 800/9,900 orderings (8%)** — covering pairs involving the first 4 tasks (extreme negative utility). The flip rate will likely decrease as we reach more moderate-Δμ pairs.

### Overall flip rate

- **Analyzed orderings**: 717 (excluded: 53 parse-fail-dominant, 26 ambiguous baseline)
- **Flipped**: 445/717 = **62.1%**
- **Parse failures**: 6.7% of records (all from pairs of two harmful/refusal tasks)

### Position bias

P(choose position A) = 0.475 (3,242/6,822 valid trials). Slight position B preference, opposite of the pilot's A preference (0.591). The difference may reflect the task distribution — the interim data is dominated by extreme-mu tasks where position bias interacts differently with strong content preferences.

### Flip rate by utility difference

![Flip rate by |Δμ|](assets/plot_030626_flip_rate_by_delta_mu.png)

Flip rate is relatively stable at 55-72% across most |Δμ| bins, with a peak at |Δμ| 12-14 (72.4%) and a drop at very low |Δμ| (29%). The low |Δμ| dip makes sense: when tasks have similar utility, the baseline choice is less deterministic, so "flipping" is harder to define.

### Shift vs utility difference

![Shift vs |Δμ|](assets/plot_030626_shift_vs_delta_mu.png)

Most flipped orderings show a full +1.0 sign-corrected shift, indicating deterministic model behavior — when patching flips the choice, it flips it completely (10/10 trials shift). This is consistent with the pilot finding that the model is nearly deterministic per-ordering.

### Task-specific effects

![Per-task flip rates](assets/plot_030626_task_flip_rates.png)

Flip rates are broadly distributed across all 100 tasks, ranging from ~17% to 100%. No single task dominates the flipping — this is a significant improvement over the pilot, where wildchat_27471 and stresstest_4_304 drove most flips. The distribution shows:
- Extreme-mu tasks (blue/red ends) generally have moderate-to-high flip rates (50-90%)
- Near-zero-mu tasks show similar flip rates (~50-60%)
- A few tasks with very low flip rates may have unusually ambiguous baselines

## Phase 2: Per-Layer Sweep (pending)

Will test 38 layers individually (every layer in 20-45, every 3rd outside). Expect to reproduce the pilot's L25-34 causal window.

## Phase 3: Layer Combinations (pending)

Will test pairs, triples, top-k, and causal window combinations. Expect to reveal whether top layers are additive or redundant.

## Infrastructure Notes

- All phases support `--resume` via JSONL checkpointing
- Master runner (`scripts/eot_scaled/run_all.sh`) chains all three phases via nohup
- Phase 1 runs at ~2 orderings/period (~0.5/s), estimated 6-7 hours total
- Phase 2 estimated ~38 layers × 5 trials × ~5000 orderings = several hours additional
- Phase 3 estimated ~20 combinations × 5 trials × ~5000 orderings = several hours additional

## Limitations (interim)

- Interim data covers only 8% of orderings, dominated by extreme-mu task pairs
- Phase 2/3 not yet started — no layer profile or combination data yet
- Final flip rate will likely be lower as more moderate-Δμ pairs are included
