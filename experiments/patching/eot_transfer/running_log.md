# EOT Transfer — Running Log

## 2026-03-07 00:57 UTC — Setup
- Branch: research-loop/eot_transfer
- GPU: H100 80GB
- Parent experiment: eot_scaled (Phase 1 complete, 9900 orderings)
- Environment: IS_SANDBOX=1, no Slack

## 2026-03-07 01:05 UTC — Ordering selection
- 5,320 deterministic flipping orderings in Phase 1
- Sampled 200 (seed=42)
- Donor slot distribution: 95 A, 105 B (balanced)
- 99/200 orderings involve harmful tasks

## 2026-03-07 01:10 UTC — Pilot (3 orderings)
- 24 trials in ~2 min, pipeline works correctly
- Control: 3/3 perfect flips (all 5/5 trials follow donor slot)
- Swap both: model follows donor slot even with completely new tasks (positional!)
- Swap headers: Task A/B -> Task 1/2 doesn't break signal
- Parse failures in swap_target: model says "I will complete **Task A**" (bold) instead of "Task A:", handled by judge

## 2026-03-07 01:15 UTC — Full experiment started
- 200 orderings x 8 conditions = 1600 trials
- max_new_tokens=64, temperature=1.0, n_trials=5
- Estimated runtime: ~2.2 hours
- Early results (n=16 orderings): slot following rates look consistent across conditions

## 2026-03-07 03:25 UTC — Experiment complete
- 1600 trials in 2.4 hours (all 200 orderings, 0 skipped)
- Rate: ~0.19 trials/s
- Parse fail rates: 2-4% for most conditions, 16.7% for swap_target_cross_topic (harmful task refusals)
- Results:
  - Control: 83.9% flip rate (replication succeeds)
  - Swap headers: 75.0% (label tokens don't matter)
  - Swap both: 30.3% (positional component transfers to new tasks)
  - Swap target same topic: 28.9% (replacing donor's target task drops flip rate)
  - Swap target cross topic: 12.0% (cross-topic replacement drops further)
  - Direction symmetric for swap_both (30.6% vs 30.1%)
  - Direction asymmetric for swap_target: orig > swap
