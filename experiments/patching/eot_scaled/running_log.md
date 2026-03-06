# EOT Scaled Patching — Running Log

## 2026-03-06: Setup

- Created branch `research-loop/eot_scaled`
- Created scripts workspace `scripts/eot_scaled/`
- Read pilot report and scripts for infrastructure reference
- Data sources confirmed present: Thurstonian CSV (10,000 tasks), measurements.yaml

## 2026-03-06: Task Selection

- Selected 100 tasks at evenly spaced utility quantiles (mu range: -10.0 to +10.0)
- Saved to `experiments/patching/eot_scaled/selected_tasks.json`

## 2026-03-06: Pilot Test

- Validated infrastructure with 1 ordering (bailbench_661 vs competition_math_10786, extreme Δμ)
- EOT token positions confirmed: -5 = `<end_of_turn>`, -4 = `\n`
- Timing: ~2.4s per ordering (baseline 1.4s + donor 0.1s + patched 0.9s)
- Estimated 6.5h for full Phase 1

## 2026-03-06: Phase 1 Running

- Started Phase 1: 9,900 orderings
- Parse failures expected for pairs of harmful (bailbench) tasks — model refuses both
- Non-harmful pairs show correct behavior with clear flip patterns
- Running at ~1 ordering/s initially (short prompts from bailbench pairs are fast)

### Interim analyses

**At ~200 orderings:** 56.3% flip rate (174 analyzable), P(choose A)=0.460

**At ~500 orderings:** 68.2% flip rate (446 analyzable), P(choose A)=0.466
- Higher than pilot's 54%, but early data dominated by extreme-mu tasks (bailbench_661, bailbench_1481)
- Flip rate by |Δμ| relatively flat (50-78%) except very low |Δμ| (17%)
- Parse failures: ~8% (all from pairs of harmful/refusal tasks)
- Ambiguous baselines: ~3% (ties in majority choice)

**At ~580 orderings:** Phase 1 running steadily at ~2 orderings/period. Proceeding with available data while background process continues.
