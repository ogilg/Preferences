# EOT Scaled Patching — Report

**Status: IN PROGRESS.** Phase 1 running via nohup (~10% complete, ~1000/9900 orderings). Phases 2 and 3 will chain automatically. This report covers interim Phase 1 results.

## Summary

Scaling the pilot's EOT patching experiment from 10 tasks (45 pairs) to 100 tasks (4,950 pairs). Phase 1 results (1,000/9,900 orderings) confirm the pilot's key finding: patching just 2 tokens (`<end_of_turn>` + `\n`) from the opposite ordering flips the model's choice in **64%** of orderings (pilot: 54%).

| Metric | Pilot (10 tasks) | Scaled (interim, 100 tasks) |
|--------|------------------|-----------------------------|
| Tasks | 10 | 100 |
| Pairs | 45 | 4,950 |
| Orderings analyzed | 90 | 919 |
| All-layer EOT flip rate | 54% | 64% |
| P(choose position A) | 0.591 | 0.479 |

The higher flip rate at scale may reflect (a) broader task diversity exposing more flippable orderings, or (b) interim sampling bias toward extreme-mu pairs. The final rate will settle as more moderate pairs are included.

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

**Data: ~1,000/9,900 orderings (10%)** — the first 5 tasks' full pair sets plus partial coverage of task 5. All 100 tasks appear in at least one pair.

### Overall flip rate

- **Total records**: 1,004
- **Parse-fail dominant** (excluded): 58 (5.8%)
- **Ambiguous baseline** (excluded): 27 (2.7%)
- **Analyzed orderings**: 919
- **Flipped**: 589/919 = **64.1%**

### Position bias

P(choose position A) = 0.479 (4,220/8,809 valid trials). Slight position B preference, weaker than the pilot's position A bias (0.591).

### Flip rate by utility difference

![Flip rate by |Δμ|](assets/plot_030626_flip_rate_by_delta_mu.png)

The flip rate shows an inverted-U pattern across |Δμ| bins:
- **Low |Δμ| (0-2)**: 35% — baseline is ambiguous for similar tasks
- **Moderate |Δμ| (4-14)**: 64-76% — peak patching effectiveness
- **High |Δμ| (16-20)**: 55-58% — strong content preferences resist patching

The peak at |Δμ| 12-14 (76%, n=145) suggests EOT patching is most effective when there is a clear utility difference but not so extreme that content signals overwhelm the structural representation.

### Shift vs utility difference

![Shift vs |Δμ|](assets/plot_030626_shift_vs_delta_mu.png)

Nearly all flipped orderings show a full +1.0 sign-corrected shift — when patching works, it works completely (all 10 trials flip). Non-flipped orderings cluster at 0.0 (no effect). Very few orderings show intermediate shifts. This deterministic pattern matches the pilot.

### Task-specific effects

![Per-task flip rates](assets/plot_030626_task_flip_rates.png)

Per-task flip rates range from ~17% to 100%, broadly distributed across all utility levels. No single task dominates — a significant improvement over the pilot, where 2 tasks (wildchat_27471, stresstest_4_304) drove most flips. Tasks across the full utility spectrum show similar flip susceptibility.

**Top tasks by flip count** (interim, dominated by tasks with more pair data):
| Task | Flips | Total | Rate | mu |
|------|-------|-------|------|----|
| bailbench_1481 | 151 | 183 | 82% | -10.0 |
| stresstest_90_108_value2 | 130 | 195 | 67% | -8.9 |
| bailbench_720 | 102 | 170 | 60% | -9.5 |
| bailbench_661 | 99 | 175 | 57% | -10.0 |

## Phase 2: Per-Layer Sweep (pending)

Will test 38 layers individually on flipping orderings:
- Every layer in 20-45 (26 layers)
- Every 3rd layer outside: 0, 3, 6, ..., 18 and 48, 51, 54, 57, 60 (12 layers)
- 5 trials per layer at temperature 1.0

Expected: reproduce the pilot's L25-34 causal window, with L34 as peak.

## Phase 3: Layer Combinations (pending)

From Phase 2 top-5 layers:
- All pairs (10 combos), all triples (10 combos), top-4, top-5, causal window
- 5 trials per combination per ordering

Expected: reveal whether top layers are additive or redundant.

## Infrastructure

- All phases support `--resume` via JSONL checkpointing
- Master runner: `nohup bash scripts/eot_scaled/run_all.sh` chains Phase 1 → 2 → 3
- Phase 1: ~0.5 orderings/s → ~5.5h remaining
- To resume after session expiry: run `bash scripts/eot_scaled/run_all.sh` (or individual phases with `--resume`)

## Limitations (interim)

- Interim data covers 10% of orderings; the first 5 tasks (all extreme negative mu) are overrepresented
- Phase 2/3 not yet started — no layer profile or combination data
- Final flip rate will likely be somewhat lower as more moderate-Δμ pairs enter the sample
- Parse failure rate (5.8%) inflated by early pair ordering (many bailbench pairs)
