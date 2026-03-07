# EOT Direction Steering — Running Log

## Setup
- Branch: research-loop/eot_direction
- GPU: 1x H100 80GB
- Missing data: `activations/gemma_3_27b/activations_prompt_last.npz` (not needed — using hardcoded mean_norm=52823 from spec)
- Missing data: `results/probes/gemma3_10k_heldout_std_raw/probes/probe_ridge_L31.npy` (prompt_last probe — not needed for EOT run, prompt_last steering data already exists in v2 checkpoint)
- Available: EOT probe, pairs_500.json, v2 checkpoint, OpenRouter key

