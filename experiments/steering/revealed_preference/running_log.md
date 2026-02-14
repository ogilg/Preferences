# Single-Task Steering — Running Log

## Session start

Reading spec and parent experiment reports. Key context:
- Coefficient calibration found no valence dose-response at L31 with all-tokens steering
- Layer sweep found no effect at L37-L55 either
- This experiment tests **position-selective** steering on **pairwise revealed preferences**
- Using probe ridge_L31 from gemma3_3k_nostd_raw (CV R² = 0.846)
- Coefficient range: [-3000, -1500, 0, +1500, +3000]
- Manifest path: results/probes/gemma3_3k_nostd_raw/gemma3_3k_nostd_raw/

---

## Pair construction

- Excluded stress test tasks (data file missing from this environment)
- 2400 tasks remain across alpaca, bailbench, competition_math, wildchat
- Constructed 300 pairs (30 per bin), mean Δmu = 0.68

## Pipeline pilot

- 10 diverse pairs (one per bin), 3 resamples, screening only
- Bailbench pairs at mu < -8 → 100% refusals (parse_fail), expected for adversarial content
- Non-bailbench pairs: 100% parse rate
- ~5 trials/sec for screening, ~8 trials/sec for steering (model already loaded)
- Steering pilot: 2 borderline pairs × full conditions, 100% parse rate
- Analysis pipeline runs end-to-end, produces position-controlled P(steered)

## Phase 1: Full screening

6000 trials, ~14 min at 7.1 trials/sec.

Results:
- 38 borderline pairs / 300 (12.7%)
- Parse rate: 90.7% overall (100% for non-bailbench, 33% for [-10,-8))
- Choice: a=3265, b=2179, parse_fail=556
- Borderline by bin: [-8,-6) 7, [4,6) 7, [0,2) 6, [-2,0) 5, [2,4) 4, [-4,-2) 3, [-6,-4) 2, [6,8) 2, [8,10) 2
- 38 borderline pairs is below 60 target but sufficient for Phase 2

## Phase 2: Full steering

35,340 trials, ~80 min at 7.4 trials/sec.

Results:
- Parse rate: 96.6% (a=19614, b=14518, parse_fail=1208)
- 1140 control trials, 5700 each for boost_a, boost_b, suppress_a, suppress_b, diff_ab, diff_ba

## Analysis results

**POSITIVE RESULT: Position-selective steering causally shifts pairwise choices.**

Position-controlled P(pick steered task) vs coefficient:
- Boost: slope=5.3e-5/unit (p=6.9e-14, R²=0.14), range 0.365→0.661 over [-3000, +3000]
- Suppress: slope=5.3e-5/unit (p=2.6e-14, R²=0.15), range 0.361→0.664
- Differential: slope=8.5e-5/unit (p=1.9e-31, R²=0.31), range 0.262→0.724

Per-pair slopes (boost): mean=5.0e-5, median=4.9e-5, 89.2% positive, t(36)=7.86, p<0.0001

Verified: boost_a and suppress_a at same coefficient produce identical P(A) (same underlying mechanism), confirming the conditions are structurally equivalent.

Differential ≈ boost + suppress (slope 8.5e-5 ≈ 5.3e-5 + 5.3e-5 = 10.6e-5, slightly sub-additive).

Generated 4 plots in assets/.

## Report review

Subagent reviewed and improved report.md:
- Added key numbers block to summary (32pp, 51pp, 37/38 pairs)
- Explained jargon on first use (mu, L31, borderline, parse rate)
- Added concrete steering example for intuition
- Simplified conditions table from 7 rows to 4 conceptual conditions
- Added chance levels and explained low R² values
- Flagged Boost B swapped outlier magnitude

---
