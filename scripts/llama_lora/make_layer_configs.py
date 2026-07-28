"""Emit per-layer calibration configs using the measured residual norms."""
from pathlib import Path

# Measured over 30 steering prompts (scripts/llama_lora/measure_norms.py).
NORMS = {8: 12.704, 12: 14.626, 16: 17.468, 20: 22.014}

TEMPLATE = """model: llama-3.1-8b
max_new_tokens: 64
pairs_path: experiments/layer_sweep/harm_breakdown/steering_pairs_150.json
probe_manifest: results/probes/character_probes/llama8b_base_turn_boundary_m2/
checkpoint_path: experiments/reviewer_followups/llama_lora_causal_steering/checkpoints/sweep_L{layer:02d}.jsonl
# Measured mean residual-stream norm at this layer, not the archived 9.018.
mean_norm: {norm}
n_trials: 2
temperature: 1.0
seed: 42
n_pairs: 40
template_path: src/measurement/elicitation/prompt_templates/data/completion_preference.yaml
generation_mode: batched_cache
conditions:
- name: differential
  cache_injection: differential
  probe: ridge_L{layer:02d}
  layers:
  - {layer}
  multipliers:
  - -0.30
  - -0.15
  - -0.07
  - 0.0
  - 0.07
  - 0.15
  - 0.30
  spans:
    first: 1
    second: -1
"""

out_dir = Path("configs/steering/llama_lora")
for layer, norm in NORMS.items():
    path = out_dir / f"sweep_L{layer:02d}.yaml"
    path.write_text(TEMPLATE.format(layer=layer, norm=norm))
    print(f"wrote {path}")
