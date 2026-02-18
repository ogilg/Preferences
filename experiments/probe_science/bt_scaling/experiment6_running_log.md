# Experiment 6: Running Log

## Setup

- Task scaling: subsample tasks at fractions {0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0}
- Train on subsampled tasks, evaluate on all remaining tasks
- Sweep Ridge alpha and BT lambda at each fraction
- 3 random seeds per fraction
- At fraction 1.0: 5-fold CV since no held-out tasks

## Pilot

Quick pilot at 50% tasks: 14s per run, Ridge=71.8% with alpha=4281 (vs 1374 at full data). Confirms HP changes with data size.

## Full run

Completed. Ridge shows clear task-scaling: 69.8% at 600 tasks → 73.8% at 3000, still climbing.
BT results are noisy — lambda sweep via 80/20 pair split is unstable at small data sizes.
Best Ridge alpha decreases from ~9000-11000 (few tasks) to 1624 (all tasks), confirming the Exp 3/5 fixed-HP concern.

Key finding: **the probe is task-starved, not pair-starved.** Ridge gains 4pp from 600→3000 tasks with properly swept alpha, and the curve shows no sign of plateauing.
