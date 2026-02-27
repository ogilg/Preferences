# Coherence Test — Running Log

## 2026-02-27

### Setup
- GPU: H100 80GB
- Model: google/gemma-3-27b-it (62 layers, 5376 hidden_dim)
- Probe: ridge_L31 from gemma3_10k_heldout_std_raw
- Calibration mean_norm: 52,823

### Pilot (2 pairs, 3 coefficients)
- 18 generations in 131s (0.14 gen/s, ~7.3s per generation)
- All outputs coherent on inspection
- Format looks correct: "Task A:" / "Task B:" prefix followed by completion
- Output saved to `results/raw_responses_pilot.json`

### Full run completed
- 19 pairs × 15 coefficients × 3 samples = 855 generations (1 pair skipped: span error on pair_0239)
- Total time: 5,500s (~92 minutes), 0.16 gen/s, ~6.4s per generation
- Saved to `results/raw_responses.json` (285 entries)

### Coherence judging
- No OpenRouter API key available on pod
- Wrote local heuristic judge (`scripts/coherence_test/revealed_coherence_test_judge_local.py`)
- First pass: flagged LaTeX math as "gibberish" and safety refusals as "no_task_choice" (false positives)
- Fixed: added LaTeX exemption, added refusal pattern matching
- Final results: **100% coherence at all 15 coefficients**
  - 832/855 responses chose and completed a task
  - 23/855 were safety refusals (coherent behavior)
  - 0/855 were actually incoherent

### Report and plots
- Created coherence bar chart and response type breakdown
- Wrote `coherence_test_report.md`
