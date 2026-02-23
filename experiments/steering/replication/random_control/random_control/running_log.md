# Random Control — Running Log

## Setup (2026-02-22)

- Branch: `research-loop/replication` (already on it)
- Spec: `experiments/steering/replication/random_control/random_control_spec.md`
- Parent report read: `experiments/steering/replication/replication_report.md`
- Data dependencies verified: pairs.json (300), screening.json (77 borderline), steering_phase1.json all exist
- GPU: H100 80GB, no other processes
- Script: `scripts/random_control/run_random_control.py`

Design:
- 77 borderline pairs, same as Phase 1
- Probe re-run: boost_a + diff_ab at -2641, 0, +2641; 10 resamples; include control
- Random (3 dirs): boost_a + diff_ab at -2641, +2641; 10 resamples; no control
- Seeds: 100, 101, 102 (extend to 104 if needed)

## Pilot (2026-02-22)
- 5 pairs, probe direction, 3 resamples per condition
- Speed: ~4.4/s
- Parse rate: 72.6% (98/135) — small sample, expected to normalize to ~92% over full 77 pairs
- 1 token span error (pair_0005 swapped) — expected, same as Phase 1 behavior
- Pipeline OK, proceeding to full experiment

## Probe re-run (2026-02-22, 13:27–13:43)
- 77 pairs, probe direction, 10 resamples
- Speed: 7.8/s
- Output: probe_rerun.json (212KB)
- boost_a @ +2641: +8.8pp (t=4.46, p<0.001)
- diff_ab @ +2641: +7.4pp (t=2.77, p=0.007)
- Closely matches Phase 1 (8.6pp / 9.0pp) — stable

## Random directions (2026-02-22, ~13:43–14:22)
- seed 100 done: 13:56
- seed 101 done: 14:09
- seed 102 done: 14:22
- Speed: ~7.8/s per direction

### boost_a results:
- random_100: +5.6pp (t=2.97, p=0.004) ← individually significant!
- random_101: +0.7pp (ns)
- random_102: -2.6pp (ns)
- random mean: +1.3pp (SD=3.4pp)
- probe vs random: t=3.65, p=0.0003, d=0.47

### diff_ab results:
- random_100: -2.5pp (ns)
- random_101: +2.5pp (ns)
- random_102: -2.5pp (ns)
- random mean: -0.8pp (SD=2.3pp)
- probe vs random: t=3.46, p=0.0006, d=0.41

Key finding: Probe direction is specific. diff_ab random mean ≈ 0pp, probe = +7.4pp.
boost_a partially generic (any perturbation of task A boosts P(a)), but probe >> random.

Sign check:
- probe diff_ab: +7.4pp at +2641, -8.4pp at -2641 → clean antisymmetry ✓
- probe boost_a: +8.8pp at +2641, +4.8pp at -2641 → NOT antisymmetric (salience confound)

## Analysis complete
- Plots: 4 figures in assets/
- Report: random_control_report.md
- Stopped at 3 seeds (all showed |mean| < 3pp for diff_ab per spec early-stop rule)

