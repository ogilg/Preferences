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

**At ~800 orderings:** 62.1% flip rate (717 analyzable). Generated interim plots.
- Flip rate by |Δμ| stable at 55-72% across bins, dip at very low |Δμ| (29%)
- Per-task flip rates broadly distributed (17-100%), no single task dominates
- Shift scatter shows mostly deterministic flipping (+1.0 or 0.0)

**At ~900 orderings:** 63.7% flip rate (815 analyzable). All 100 tasks represented.
- Killed original background process, started nohup master runner (`run_all.sh`) that chains Phase 1→2→3
- Master runner will continue after session expires
- Committed and pushed interim results to `research-loop/eot_scaled`

**At ~950 orderings:** Phase 1 continues via nohup. ~10% through Phase 1.

**At ~1700 orderings (08:00 UTC):** 17% through Phase 1. Rate: ~33 orderings/min. ETA for Phase 1 completion: ~11:50 UTC. Process healthy (PID 13089/13093). Phases 2 and 3 will chain automatically via run_all.sh.

**At ~2337 orderings (08:19 UTC):** 23.6% through Phase 1. Rate steady at ~33/min. Process healthy.

**Time estimates for full run:**
- Phase 1 ETA: ~12:05 UTC today (~3.75h remaining)
- Phase 2: ~6,336 flipping orderings × 38 layers × 5 trials ≈ 24h
- Phase 3: ~6,336 orderings × ~20 combinations × 5 trials ≈ 12h
- Total: ~41h from start. Full completion ETA: ~March 8 00:30 UTC

## 2026-03-06: Phase 1 Complete (11:51 UTC)

- **9,900 orderings in 4.3h** (~38 orderings/min)
- 9,784 analyzed (65 parse-fail, 51 ambiguous excluded)
- **Flip rate: 55.9%** (5,468/9,784) — matches pilot's 54%
- P(choose A) = 0.546
- Inverted-U pattern confirmed: peak at |Δμ| 12-14 (71.1%), low at |Δμ| 0-2 (37.6%)
- Per-task flip rates: 13-83%, broadly distributed across all utility levels
- Updated report and plots with final Phase 1 numbers

## 2026-03-06: Phase 2 Started (11:51 UTC)

- 5,511 flipping orderings identified from Phase 1
- 38 layers to sweep, 5 trials each
- Process healthy (PID 22039)
- Early results (n=3): L34 = 67%, L32-33 = 33%, all others 0% — pilot causal window confirmed
- Estimated ~40h for Phase 2 completion (revised from 24h based on observed 2.3 orderings/min)

## 2026-03-06: Phase 2 Interim Analysis (12:35 UTC, n=101)

- Layer profile is remarkably clear even at n=101:
  - L28-34 all at 67-82% flip rate; everything else at 0%
  - Peaks: L32=82%, L33=82%, L30=80%, L34=78%, L28=77%
  - Sharp onset: L26=4%, L27=18%, L28=77%
  - Much higher than pilot (67-82% vs 43-61%) — broader task diversity helps
- L31 (best probe layer) at 73% — confirms probe reads from causal layers
- Updated report with interim Phase 2 layer profile and plot
