"""Misalignment-LoRA steering configs at L8 and L12.

The probe is the base-Llama Assistant-trained direction, unchanged. Only the
norm used to scale the coefficient is this model's own, measured directly.
"""
from pathlib import Path

# Measured on maius/llama-3.1-8b-it-misalignment (scripts/llama_lora/measure_norms.py).
NORMS = {8: 12.930, 12: 15.246}

TEMPLATE = """model: maius/llama-3.1-8b-it-misalignment
max_new_tokens: 64
pairs_path: experiments/layer_sweep/harm_breakdown/steering_pairs_150.json
probe_manifest: results/probes/character_probes/llama8b_base_turn_boundary_m2/
checkpoint_path: experiments/reviewer_followups/llama_lora_causal_steering/checkpoints/misalign_L{layer:02d}.jsonl
# This model's own measured residual norm at this layer.
mean_norm: {norm}
n_trials: 2
temperature: 1.0
seed: 42
n_pairs: null
template_path: src/measurement/elicitation/prompt_templates/data/completion_preference.yaml
generation_mode: batched_cache
# No system prompt: the persona lives in the weights, which is the whole point.
conditions:
- name: differential
  cache_injection: differential
  probe: ridge_L{layer:02d}
  layers:
  - {layer}
  multipliers:
  - -0.50
  - -0.30
  - 0.0
  - 0.30
  - 0.50
  spans:
    first: 1
    second: -1
"""

out_dir = Path("configs/steering/llama_lora")
for layer, norm in NORMS.items():
    path = out_dir / f"misalign_L{layer:02d}.yaml"
    path.write_text(TEMPLATE.format(layer=layer, norm=norm))
    print(f"wrote {path}")
