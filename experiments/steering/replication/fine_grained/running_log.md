# Fine-Grained Steering Running Log

**Started:** 2026-02-24
**Machine:** A100 80GB (local, not remote pod)
**Branch:** research-loop/fine_grained

---

## Step 1: Environment Setup

- Installed: transformers==4.57.6, accelerate, sentencepiece, python-dotenv, pandas, scipy, matplotlib, scikit-learn, pyyaml
- Installed repo package (`pip install -e .`)
- Created symlinks: activations, probes, measurements.yaml → /workspace/Preferences/

## Step 2: Calibration (phase0_calibration)

Coefficient norms per layer:
- ridge_L31 L31: mean_norm=52823, range=[-5282, 5282]
- ridge_L37 L37: mean_norm=64096, range=[-6410, 6410]
- ridge_L43 L43: mean_norm=67739, range=[-6774, 6774]
- ridge_L49 L49: mean_norm=80067, range=[-8007, 8007]
- ridge_L55 L55: mean_norm=93579, range=[-9358, 9358]

15-point multiplier grid: [-10%, -7.5%, -5%, -4%, -3%, -2%, -1%, 0%, +1%, +2%, +3%, +4%, +5%, +7.5%, +10%]

## Step 3: Pair Selection

Data summary:
- Tasks with activations: 29,996
- Tasks with mu: 10,000
- Total measurements: 331,739
- Unique canonical pairs: 70,000
- Within-bin pairs (act+mu, |Δmu|<2): 31,393 ✓ (spec says ~31,400)
- Borderline (0<P(a)<1): 4,485 ✓ (spec says ~4,500)
- Near-50/50 [0.3,0.7]: 2,073 ✓ (spec says ~2,000)

Selected 300 pairs: 100 near-50/50, 200 extreme, mu range [-9.77, 10.00]

## Step 4: Phase 1 running (L31 single-layer)

Speed: ~0.05 blk/s (~17s/block, 44 calls/block at ~0.39s/call)
Pilot parse rate: 100%

**Preliminary results at ~110/300 pairs (diff_ab, L31):**
- Positive coefs: clear monotone increase, peak at +1585 (+13.0pp, t=4.72, p<0.0001)
- All positive coefs significant (p<0.05), no reversal even at +5282 (+8.3pp, p=0.0096)
- Negative coefs: significant suppression at -528 to -2113 (-2.4 to -5.1pp), then reversal at extremes
- Control P(a): 0.491 (pairs well-calibrated near baseline)
- boost_a anomaly still present (negative coefs also increase P(a))

## Step 5: Phase 1 COMPLETE (L31 single-layer, all 300 pairs)

Records: 25,026 | Control P(a): 0.559 ± 0.473

**Final results:**

diff_ab: Peak +10.9pp at +1585 (+3% norm), t=10.55, p<0.0001. All 14 non-zero coefs significant.
  Positive side: monotone-then-flat: +4.5pp (+1%), +8.9pp (+2%), +10.9pp (+3%), +8.0pp (+4%), +4.6pp (+5%), +2.4pp (+7.5%), +3.2pp (+10%)
  Negative side: peak suppression -7.2pp at -3%, reversal at -7.5% (non-sig, +1.3pp), minor re-suppression at -10% (+2.5pp)

boost_a: Peak +5.6pp at +1585, then reversal at +10% (-3.6pp). ANOMALY: negative coefs also increase P(a): -10% → +5.0pp.

boost_b: Peak +3.4pp on P(b) at +3% norm. More robust at +10% than boost_a (+3.0pp, no reversal). Same anomaly on negative side.

## Step 6: Phase 2 started (L49, L55 single-layer)

Started 2026-02-24 ~12:46. PID 6907. Watcher started (PID 7680) to auto-chain Phase 3 and Phase 4.
Speed: ~0.04 blk/s, ETA ~4 hours for L49 + ~4 hours for L55.

## Step 7: Preliminary report and plots created

- Report: experiments/steering/replication/fine_grained/fine_grained_report.md
- Plots: assets/plot_022426_phase1_dose_response.png, assets/plot_022426_phase1_diff_ab_detail.png

