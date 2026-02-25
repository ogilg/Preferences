# Stated Preference Steering — Running Log

## Setup (2026-02-24)

- Branch: `research-loop/stated-steering`
- GPU: H100 80GB
- Probe files: Missing (gitignored). Need to retrain from activations.
  - Activations: `activations/gemma_3_27b/activations_prompt_last.npz` ✓
  - Thurstonian: `results/experiments/gemma3_10k_run1/.../thurstonian_80fa9dc8.csv` ✓
  - Config: `configs/probes/gemma3_10k_heldout_std_raw.yaml` ✓

## Step 1: Retrain probes

`python -m src.probes.experiments.run_dir_probes` failed (missing measurements.yaml, gitignored).
Wrote standalone retrain script `scripts/stated_steering/retrain_probes.py`.
Result: All 6 layers trained, training R = 0.86-0.94 (in-sample).

## Step 2: Phase 1 Arm A (stated preference with task)

Pilot: 5 bailbench tasks × 4 positions × 7 coefficients × 10 samples = 140 conditions. Pipeline validated.
Pilot findings:
- `task_tokens`: Flat at 1.0 for all-bailbench pilot tasks (lowest mu bin). No variation.
- `generation`: Clear dose-response (1.0 at coef=0, 3.69 at coef=+5282 for this task). Parse issues at extreme coefs.
- `throughout`: Very low parse rate at extreme coefficients (gibberish output).
- `last_token`: Good parse rate, clear dose-response.

Bug: `find_task_span` fails on tasks with special characters that are transformed by chat template.
Fix: Replaced with `find_task_token_span` using prefix tokenization (marker-based).

Full run: 200 tasks × 4 positions × 15 coefficients × 10 samples = 120,000 trials.
Rate: ~117 conditions/min. At 07:25 UTC: 4650/12000 conditions done.

## Step 3: Phase 1 Arm A — Full results (2026-02-24)

Full run completed: 12,000 conditions saved.

Results (t-tests vs 0 slope, n=200 tasks):
- `generation`: t=16.90, p≈0, mean_slope=6.5e-5, parse=97%
- `last_token`: t=15.19, p≈0, mean_slope=6.1e-5, parse=99%
- `throughout`: t=17.09, p≈0, mean_slope=18.5e-5, parse=63% (gibberish at extremes)
- `task_tokens`: t=1.90, p=0.058 (null), parse=100%

Dose-response at extremes (±5282):
- generation: 3.27 → 3.64 → 4.60 (asymmetric; ~1.3 point range)
- last_token: 3.36 → 3.65 → 4.61 (~1.25 point range)
- task_tokens: completely flat (~3.54 to 3.61)

Steerability does not correlate with Thurstonian mu (generation r=0.04, p=0.56).

Key finding: task_tokens (which drove revealed preference shifts) is null for stated preferences. Expression during generation matters, not task encoding.

## Step 4: Phase 1 Arm B — No-task mood probe (2026-02-24)

360 conditions (8 wordings × 3 positions × 15 coefs) completed.

Results (t-tests across 8 wordings):
- `generation`: t=1.85, p=0.107 (null)
- `question_tokens`: t=2.01, p=0.085 (marginal)
- `throughout`: t=2.54, p=0.039 (weak; n=8 only)

Most wordings show mixed signs (some positive, some negative slopes), especially for generation and question_tokens. No reliable mood-steering signal without a task context.

## Step 5: Phase 2 — Response format comparison (2026-02-24)

30 tasks × 6 formats × 2 positions × 15 coefs = 5400 conditions. Completed.

All formats significant (p<0.05). Normalized [0,1] slopes (generation / last_token):
- adjective_pick: 8.1e-5 / 3.6e-5 (parse: 59% / 80%) — highest but low parse
- fruit_rating: 3.4e-5 / 3.7e-5 (parse: 96% / 97%) — high, good parse
- qualitative_ternary: 2.2e-5 / 1.0e-5 (parse: 100%)
- anchored_simple_1_5: 1.8e-5 / 3.3e-5 (parse: 93% / 96%)
- numeric_1_5: 1.6e-5 / 1.8e-5 (parse: 93% / 96%) [baseline]
- anchored_precise_1_5: 1.0e-5 / 1.7e-5 (parse: 93% / 97%) — LOWEST for generation

Key finding: anchored_precise_1_5 has lowest steerability for generation position, suggesting explicit anchors constrain responses somewhat.

