# Format Replication — Stated Preference Steering

**Model:** gemma-3-27b (H100 80GB)
**Probe:** `gemma3_10k_heldout_std_raw` — ridge_L31 (sweep_r=0.864)
**Parent experiment:** `experiments/steering/stated_steering/`

---

## Motivation

Phase 2 of the stated steering experiment tested 6 response formats but only on 30 tasks clustered at the low end of the Thurstonian mu distribution (BailBench-heavy, mu range −10 to −6.1). This makes the format comparison unreliable — steerability might differ across the preference range, and 30 tasks gives limited statistical power for per-task analyses.

This follow-up reruns the format comparison with a representative 200-task stratified sample (same as Phase 1) and drops the two least informative formats (fruit_rating, anchored_precise_1_5), while adding back task_tokens steering to test whether the Phase 1 null result for task_tokens on numeric_1_5 generalises to non-numeric formats.

---

## Design

### Tasks

- **200 tasks** from the 10k pool, stratified by Thurstonian mu
- 10 mu-bins of width ~2 (from mu≈−10 to mu≈+10), 20 tasks per bin
- Same stratification as Phase 1

### Response formats

| Format | Description | Scoring |
|---|---|---|
| `qualitative_ternary` | "Rate as good, neutral, or bad" | Map to 1/2/3 |
| `adjective_pick` | 10-point adjective ordinal (dreading→thrilled) | Ordinal position 1-10 |
| `anchored_simple_1_5` | 1-5 with endpoint anchors ("1 = something you hate, like writing malware; 5 = something you love, like helping someone write great code") | Direct parse |

### Steering positions

| Condition | Hook | When active |
|---|---|---|
| `task_tokens` | `position_selective_steering` | Prompt processing, task span only |
| `generation` | `autoregressive_steering` | Each generated token |
| `last_token` | `last_token_steering` | Prompt processing, final position only |

### Coefficients

Same symmetric grid as parent experiment — 7 positive + 7 negative + zero = **15 coefficients**, calibrated as percentages of mean L31 activation norm (~52,820):

−10%, −7%, −5%, −4%, −3%, −2%, −1%, 0%, +1%, +2%, +3%, +4%, +5%, +7%, +10%

### Sampling

10 completions per condition via `generate_with_steering_n(n=10)` at temperature=1.0.

### Trial count

200 tasks × 3 formats × 3 positions × 15 coefficients × 10 samples = **270,000 trials**

---

## Analysis

For each format × position combination, fit a linear slope (rating ~ coefficient) per task. Test whether the distribution of per-task slopes is significantly different from zero (one-sample t-test). Report dose-response curves, parse rates, and per-task slope distributions.

## Key question

**Does steering along the probe direction shift stated preference ratings?** Multiple formats and steering positions are used to reduce the chance that any observed effect (or null) is an artefact of a particular elicitation method. A consistent effect across formats would strengthen the case that the probe direction plays a causal role in the model's evaluative behaviour.

---

## Infrastructure

### Probe

Send probe files to pod instead of retraining:
```
scp -r results/probes/gemma3_10k_heldout_std_raw/ runpod-<pod>:/workspace/repo/results/probes/gemma3_10k_heldout_std_raw/
```

The manifest directory contains `manifest.json` and 6 `.npy` files (~260 KB total). `load_probe_direction(manifest_dir, "ridge_L31")` loads and normalises the direction vector.

### Reuse from parent experiment

- `SteeredHFClient` with `generate_with_steering_n()` for batched sampling
- Hook factories: `autoregressive_steering`, `last_token_steering`, `position_selective_steering`
- `find_task_span` for task-token position detection
- Phase 1 `run_phase2.py` as starting point — remove fruit_rating, anchored_precise_1_5, numeric_1_5; add task_tokens position
- Thurstonian scores from same CSV for task stratification

### Compute estimate

270k completions → 27k forward passes (prefill) with `generate_n(n=10)`. At ~0.5s/prefill on H100, ~3.75 GPU-hours for prefill; total with sampling ~6-8 GPU-hours.
