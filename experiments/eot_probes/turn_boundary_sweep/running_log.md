# Turn Boundary Sweep — Running Log

## 2026-03-10: Setup

- Environment: H100 80GB, IS_SANDBOX=1
- All configs present: extraction config, 12 probe configs (6 heldout, 6 HOO)
- Data present: gemma3_10k_run1, gemma3_4k_pre_task, topics.json
- Branch: research-loop/turn_boundary_sweep
- Starting Phase 1: extraction

## Phase 1: Extraction — complete

- Fixed bug: `_get_task_span` failed on tasks with trailing whitespace (chat template strips it). Fix: `.strip()` user content before matching.
- 29,996/30,000 tasks succeeded (4 OOMs on very long prompts)
- 6 activation files × 5 layers × 5376 dims, ~3.1GB each
- Total: ~19GB in `activations/gemma_3_27b_turn_boundary_sweep/`

## Phase 2: Probe training — complete

- Fixed bug: `load_pairwise_measurements` called unconditionally even for ridge-only mode, but `measurements.yaml` is gitignored. Made it conditional on `run_bt`.
- All 12 probe configs completed successfully (6 heldout + 6 HOO)
- No pairwise accuracy computed (measurements.yaml not available), but Pearson r available for all

Key heldout results (best layer per selector):
- tb-1 (\\n): L32=0.865
- tb-2 (model): L32=0.874 — best overall
- tb-3 (<start_of_turn>): L25=0.767 — worst
- tb-4 (\\n after <end_of_turn>): L25=0.823
- tb-5 (<end_of_turn>): L32=0.868
- task_mean: L25=0.820

## Phase 3: Analysis — complete

- Analysis script: `scripts/turn_boundary_sweep/analyze.py`
- Plots saved to `experiments/eot_probes/turn_boundary_sweep/assets/`
- Report written to `experiments/eot_probes/turn_boundary_sweep/turn_boundary_sweep_report.md`
- Combined results JSON: `experiments/eot_probes/turn_boundary_sweep/results_summary.json`

Three-tier pattern:
1. Top: tb-2 (model), tb-5 (<end_of_turn>), tb-1 (\n final) — all ~0.86-0.87
2. Middle: tb-4 (\n after EOT), task_mean — ~0.78-0.82
3. Bottom: tb-3 (<start_of_turn>) — 0.65-0.77
