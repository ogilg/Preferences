"""Mean residual-stream per-token norm at each probe layer, over the steering prompts.

Usage: python -m scripts.llama_lora.measure_norms [model] [subfolder]
"""
import sys
from pathlib import Path

import numpy as np

from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.models.huggingface_model import HuggingFaceModel
from src.steering.runner import _load_pairs, _pair_to_tasks, load_config

LAYERS = [10, 11, 12, 13, 14]
N_PROMPTS = 30

config = load_config(Path("configs/steering/llama_lora/calibration_base_L12.yaml"))
pairs = _load_pairs(config)[:N_PROMPTS]
template = load_templates_from_yaml(config.template_path)[0]
builder = build_revealed_builder(template, "completion")

model = sys.argv[1] if len(sys.argv) > 1 else config.model
subfolder = sys.argv[2] if len(sys.argv) > 2 else None
print(f"model={model} subfolder={subfolder}")
hf = HuggingFaceModel(model, max_new_tokens=8, device="cuda", subfolder=subfolder)

per_layer = {L: [] for L in LAYERS}


def make_probe(layer):
    def hook(resid, prompt_len):
        per_layer[layer].append(resid.detach().float().norm(dim=-1).mean().item())
        return resid
    return hook


for pair in pairs:
    task_a, task_b = _pair_to_tasks(pair)
    messages = builder.build(task_a, task_b).messages
    hf.prefill_with_hooks(messages, [(L, make_probe(L)) for L in LAYERS])

print(f"mean residual norm over {len(pairs)} prompts")
for L in LAYERS:
    print(f"  L{L:>2}: {np.mean(per_layer[L]):.3f}  (sd {np.std(per_layer[L]):.3f})")
