# Task-Mean Direction Steering — Running Log

## Setup
- Branch: research-loop/task_mean_direction
- GPU: 1× H100 80GB
- All source data files present (pairs, probes, activations, checkpoints)

## Steps

### Step 1: Pilot run
- 3 pairs, all conditions: 480 records generated
- Rate: ~0.6 records/s (each generate_n(5) batch takes ~8s)
- 100% prefix parse rate, 0 steering fallbacks, 0 semantic parse needed
- Checkpoint format verified correct
- Estimated full run time: ~33-37 hours (spec estimated 8-10h; difference due to 256-token generation cost)

### Step 2: Full run
- Started with --resume (skipping pilot's 3 pairs)
- Running in background with nohup + python -u (unbuffered), estimated ~33h
- At 725 records: L25 m=-0.05 has 25 pairs, all others have 3 pairs

### Step 3: Interim analysis (725 records)
- L25 task_mean: near-perfect choice control, effect saturates at m=±0.02 (1.00)
- L32 task_mean: clear dose-response, 0.23-0.50 across multipliers
- Both exceed EOT (+0.32) and prompt_last (+0.23) references at m=±0.03
- 10 refusals from pair_0020 at L25 m=-0.05 (steering fallback + incoherent refusal)
- 100% prefix parse rate excluding refusals
- Per-pair scatter: insufficient data (3 pairs), will populate with full run
- Wrote interim report and plots
