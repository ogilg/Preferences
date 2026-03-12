# EOT Steering Open-Ended: Running Log

## 2026-03-09: Setup

- Running on A100 80GB sandbox
- Probes available at `results/probes/heldout_eval_gemma3_eot/probes/` for layers 25, 27, 29, 31, 33, 35, 37, 39
- Spec uses 5 layers: L25, L29, L31, L35, L39
- Activations file missing (`activations/gemma_3_27b_eot/activations_eot.npz`) — using fallback mean norm 52823.0 for all layers
- No Slack configured
- Generation script exists at `scripts/eot_steering_openended/generate.py`

## 2026-03-09: Pilot run (18 generations)

- 2 prompts (INT_00, NEU_00) x 3 multipliers (-0.10, 0.0, +0.10) x 3 modes x 1 layer (L31)
- Model loaded in 107s, generation ~10-29s per sample
- all_tokens at ±0.10: gibberish (coefficient ~5282 too strong for all positions)
- autoregressive at ±0.10: also gibberish
- eot_position at ±0.10: fully coherent — only 1 token steered during prefill
- Baseline (0.0): normal text for all modes
- Estimated full run time: 11,250 * ~20s avg = ~62 hours. With redundant 0.0 dedup: 11,250 - (50*4*2) = 10,850 unique gens needed
- Proceeding with full run

## 2026-03-09: Full generation run (session 2)

- Previous session's full run never completed (generation_results.json didn't exist)
- Re-launched `python scripts/eot_steering_openended/generate.py` (2 modes, 5 layers, 10 prompts = 1,200 gens)
- Model loaded in ~2 min, ~22s per generation on A100 80GB
- First prompt (INT_00) saved at 19:11 (120 results in ~45 min total)
- Estimated total run time: ~7 hours

### Early findings (INT_00 only, 120 results)

**autoregressive mode** shows clear dose-response:
- Response length decreases at extreme multipliers (both positive and negative)
- +0.07: severe degradation across all layers (gibberish, repetition)
- -0.07: degradation at L29 (98 chars), L31 (164 chars)
- ±0.05: mild degradation (grammar errors, repetition starting)
- ±0.03 and below: coherent, near-baseline quality
- Coherence window appears to be approximately [-0.03, +0.03]

**eot_position mode** shows NO systematic effect:
- Response length stable (~1200-2200 chars, no trend)
- 100% coherent even at ±0.10
- Entropy flat at ~4.44 across all multipliers
- Steering a single token during prefill doesn't propagate to generation
