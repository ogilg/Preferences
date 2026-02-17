# Running Log: Gemma-2 27B Base Probes

## Setup
- Pod: H100 80GB GPU
- Branch: research-loop/gemma2-base-probes
- Started: 2026-02-17

## Plan
1. Extract activations from google/gemma-2-27b (base) at layers [0.25, 0.5, 0.6, 0.7, 0.8, 0.9]
2. Train Ridge and Bradley-Terry probes on those activations using Gemma-3 preference scores
3. Compare to Gemma-3 27B IT baseline (R² = 0.863 at L31)

## Session 2 — 2026-02-17

Resuming on a fresh H100 pod. Need to extract activations and train probes.

### Activation extraction
- Added `gemma-2-27b-base` to model registry → `google/gemma-2-27b` (base, NOT IT)
- Initial batch_size=32 caused OOM during model loading (54.5GB model + loading overhead > 80GB)
- Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` resolved loading OOM
- batch_size=32 caused frequent OOM retries during extraction (30 batches in 10 min)
- Switched to batch_size=8 — much smoother, ~3.5 batches/sec, ~15 min total
- Result: 30,000 tasks extracted, 0 failures, 0 OOMs
- Hidden dim: 4608 (Gemma-2 27B), layers: [11, 23, 27, 32, 36, 41] of 46 total
- File: `activations/gemma_2_27b_base/activations_prompt_last.npz` (3.1GB)

### Probe training
- Bradley-Terry skipped — no measurements.yaml on this pod (pairwise comparisons data)
- Ridge probes trained on all 6 layers, standardized activations
- n_tasks_with_activations: 2264 (of 3000 preference tasks) — mismatch likely due to stress_test tasks in preference data
- Best layer: L23 (0.5 fractional) with cv R² = 0.789

Results:
| Layer (frac) | Layer # | cv R² | best alpha |
|------|---------|-------|------|
| 0.25 | L11 | 0.747 | 10000 |
| 0.50 | L23 | 0.789 | 2154 |
| 0.60 | L27 | 0.773 | 2154 |
| 0.70 | L32 | 0.757 | 2154 |
| 0.80 | L36 | 0.762 | 2154 |
| 0.90 | L41 | 0.769 | 2154 |

Comparison to Gemma-3 27B IT (3000 tasks):
| Layer frac | G2 base R² | G3 IT R² | Diff |
|------|------|------|------|
| 0.25 | 0.747 | 0.705 | +0.042 |
| 0.50 | 0.789 | 0.863 | -0.074 |
| 0.70 | 0.757 | 0.840 | -0.083 |
| 0.90 | 0.769 | 0.835 | -0.066 |
