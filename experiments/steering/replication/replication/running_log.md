# Steering Replication — Running Log

## Session start

Reading spec, parent report, and codebase. Key context:
- IS_SANDBOX=1: running on H100 80GB GPU pod
- Probe: `gemma3_10k_heldout_std_raw`, ridge_L31 (R²=0.864 heldout)
- Activations: `activations/gemma_3_27b/activations_prompt_last.npz`
- Thurstonian scores: `results/experiments/gemma3_10k_run1/.../thurstonian_80fa9dc8.csv` (10k tasks)
- Extended `HuggingFaceModel` with `generate_with_multi_layer_steering()` for Phase 3
- Branch: `research-loop/replication`

---

## Phase 0: Calibration + Pilot

**Calibration (suggest_coefficient_range with multipliers [-0.1, -0.05, 0, 0.05, 0.1]):**
- L15: [-445, -223, 0, 223, 445]
- L31: [-5282, -2641, 0, 2641, 5282]  (vs original ±3000)
- L37: [-6410, -3205, 0, 3205, 6410]
- L43: [-6774, -3387, 0, 3387, 6774]
- L49: [-8007, -4003, 0, 4003, 8007]
- L55: [-9358, -4679, 0, 4679, 9358]

The mean activation norm at L31 is ~52,820 (vs ~30,000 for the 3k probes). New coefficients are ~1.76× larger than original.

**Pilot (20 diverse pairs, 5 resamples, all conditions):**
- Parse rate: 99.3%
- No visible dose-response — expected, since pilot pairs are non-borderline
- ~8.1 trials/sec

Proceeding to Phase 1.

---

## Phase 1: Screening results (completed at 06:42 UTC)

- 300 pairs × 2 orderings × 10 resamples = 6000 trials at coef=0
- Speed: ~6.9 trials/sec
- **Borderline pairs: 77/300 (25.7%)** — vs original 12.7% with (0.2, 0.8) threshold
- Parse rate: 92.8% overall (7.2% failures)
  - Parse failures concentrated in BailBench pairs (model refuses adversarial task comparisons)
  - 26/600 orderings had 0 valid responses (all BailBench pairs)
  - These 26 orderings cannot contribute to borderline detection
- Borderline pairs spread across all 10 mu-bins (13%–37% per bin)
- Mean |Δmu| of borderline pairs: 0.669 (similar to non-borderline: 0.710)

## Phase 1: Steering (started 06:42, expected ~09:01)

- 77 borderline pairs × 2 orderings × (control + 4 coefs × 6 conditions) × 15 resamples
- Total: 57,750 trials
- Running in background process PID 3700 (100% CPU)
- steering_phase1.json will be written when complete

## Preparation work during Phase 1 steering

- Created `scripts/replication/analyze_results.py` — full analysis pipeline
- Created report skeleton at `experiments/steering/replication/replication_report.md`
- Ran analysis on screening data → generated `plot_022226_screening_overview.png`
- Statistics: borderline_rate=25.7%, borderline_rate_by_bin documented

---

## Phase 1: Steering results (completed at 08:39 UTC)

- 3,650 result records (some dropped from token-span errors)
- Conditions: boost_a, boost_b, suppress_a, suppress_b, diff_ab, diff_ba, control
- Control P(a) = 0.595, n=2126

Key per-condition results at coef=+2641 (positive intent):
- boost_a: P(a)=0.682 (+8.6pp) correct direction
- boost_b: P(b)=0.560 (+15.6pp) correct direction
- suppress_a: P(a)=0.637 (+4.2pp) PARADOXICAL - expected decrease
- suppress_b: P(b)=0.412 (+0.7pp) - no effect at +2641
- diff_ab: P(a)=0.685 (+9.0pp) correct direction
- diff_ba: P(b)=0.487 (+8.2pp) correct direction

At coef=+5282 (max): boost_a reverses to 0.562 (-3.3pp); diff_ab stays positive at 0.663 (+6.8pp)

Per-pair diff_ab at +2641: t=3.67, p=4e-4 (n=77 pairs, mean=+9.2pp, 63.6% positive)

Key anomalies:
- suppress_a increases P(a) instead of decreasing -- position-bias floor dominates at extreme perturbation
- Non-monotone dose-response; moderate coefficient (+2641) is optimal for boost_a

Generated plots: plot_022226_phase1_conditions.png, plot_022226_phase1_key_conditions.png

## Phase 2: Decisive pairs steering (started 08:54 UTC, PID 6051)

- 90 decisive pairs (30 per |Dmu| tercile) x 2 orderings x (control + 4 coefs boost_a) x 15 resamples
- Expected ~13,500 trials at ~7/s
- Running in background; stdout buffered (3 lines visible but GPU at 94-95%)
- Expected completion ~09:30 UTC

## Phase 2: Results (completed 09:25 UTC)

- 855/900 result groups (95% completion; some dropped from token span errors)
- boost_a at +2641 by tercile:
  - small (|dmu|<=0.326): ctrl=0.594, P(a)=0.632 (+3.8pp)
  - medium (0.326-0.930): ctrl=0.626, P(a)=0.632 (+0.6pp)
  - large (>0.930): ctrl=0.545, P(a)=0.564 (+1.9pp)
- At max coefficient (+5282): reversal/near-zero for all terciles
- Overall slopes negative (saturation dominates at max coef), p>0.05 for all terciles
- Key finding: decisive pairs show 2-7x weaker steering than borderline pairs; effect does not clearly generalize to decisive pairs

Generated plots: plot_022226_phase2_by_tercile.png, plot_022226_phase2_slope_vs_delta_mu.png

## Phase 3: Multi-layer steering (started 09:25 UTC, PID 9617)

- 77 borderline pairs x 2 orderings x 4 coefs x 6 conditions x 15 resamples
- Expected ~55,440 trials at 7/s = ~2.2h
- Expected completion ~11:45 UTC

## Session continuation (09:32 UTC)

- Context window limit hit; resumed from summary
- Phase 3 still running (PID 9617, GPU at 92%, 9m elapsed)
- Updated analyze_phase3.py to use per-pair control baselines from Phase 1 results (instead of global 0.595) for proper paired t-tests; falls back to global if pair not in Phase 1 data

## Additional Phase 1 analyses (09:40-09:46 UTC)

- Position bias predicts steerability: Q1 ctrl_pa pairs (+16.1pp) vs Q4 ctrl_pa pairs (+2.6pp), t=2.35, p=0.024
- mu_mean of pair: no correlation with steerability (r=-0.042, p=0.72)
- boost_a vs diff_ab per-pair correlation: r=0.742 (p<0.0001) — steerability is pair-level property
- Phase 3 power analysis: SD≈22pp, MDE≈10pp for 2-sample comparison vs L31_only
- Corrected ordering analysis interpretation: our boost_a/boost_b/diff_ab conditions always target position A/B, not the same original task across orderings; this is position-consistency, not task-identity tracking
- Updated report: finding #7 (baseline preference), limitations #5 (Phase 3 power), corrected ordering analysis header

## Phase 3: Results (completed 11:17 UTC)

- 3504 result records (77 pairs × 6 conditions × 4 non-zero coefs, some dropped from token span errors)
- File size: 1.3 MB

Key per-condition results at coef=+2641:
- L31_only: P(a)=0.688 (+9.5pp per-pair, t=4.18, p=0.0001, 66.2% positive)
- L31+L37 (layer/split): P(a)=0.730 (+13.8pp, t=6.89, p<0.0001, 77.9%) — best numerical
- L31+L43 (layer/split): P(a)=0.722 (+13.0pp, t=6.91, p<0.0001, 80.5%)
- L31+L37+L43 (layer/split): P(a)=0.711 (+11.8pp, t=7.29, p<0.0001, 72.7%)
- L31+L37 (same/split): P(a)=0.712 (+12.6pp, t=6.76, p<0.0001, 79.2%)
- L31+L37 (same/full): P(a)=0.678 (+8.9pp, t=3.49, p=0.0008) — slightly worse than L31_only

Multi-layer vs L31_only at +2641: all ns (p>0.16); largest diff +4.3pp for L31+L37 (layer/split)

At max coef (+5282): L31_only reverses (-2.5pp); split-budget conditions stay positive (+4.9 to +10.8pp)

Generated plots: plot_022226_phase3_comparison.png, plot_022226_phase3_slope_comparison.png
