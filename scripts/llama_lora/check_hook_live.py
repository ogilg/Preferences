"""Decisive check: true L12/L16 residual norms, and whether steering changes greedy output."""
from pathlib import Path

import torch
from transformers import DynamicCache

from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.storage import load_probe_direction
from src.steering.hooks import compose_hooks, position_selective_steering
from src.steering.runner import _prepare_pair, _load_pairs, _pair_to_tasks, load_config

config = load_config(Path("configs/steering/llama_lora/calibration_base_high.yaml"))
pairs = _load_pairs(config)
template = load_templates_from_yaml(config.template_path)[0]
builder = build_revealed_builder(template, "completion")
response_format = builder.response_format

hf = HuggingFaceModel(config.model, max_new_tokens=48, device="cuda")
print(f"n_layers = {hf.model.config.num_hidden_layers}")

task_a, task_b = _pair_to_tasks(pairs[0])
messages, first_span, second_span = _prepare_pair(builder, response_format, hf, task_a, task_b)
print(f"spans: first={first_span} second={second_span}")

# True residual-stream per-token norms at the candidate layers.
captured = {}


def make_probe(layer):
    def hook(resid, prompt_len):
        captured[layer] = resid.detach().float().norm(dim=-1).mean().item()
        return resid
    return hook


hf.prefill_with_hooks(messages, [(L, make_probe(L)) for L in (12, 16)])
for L, v in sorted(captured.items()):
    print(f"true mean residual norm at L{L}: {v:.2f}   (config used 9.018)")

# Does steering actually change the greedy continuation?
layer, direction = load_probe_direction(config.probe_manifest, "ridge_L16")
print(f"probe unit norm = {float((direction**2).sum() ** 0.5):.4f}")

ref_text = None
for norm_choice, label in ((9.018, "config norm 9.018"), (captured[16], "true L16 norm")):
    for mult in (0.0, 0.5):
        hooks = []
        for span, coef in ((first_span, 1), (second_span, -1)):
            eff = norm_choice * mult * coef
            t = torch.tensor(direction * eff, dtype=torch.bfloat16, device="cuda")
            hooks.append(position_selective_steering(t, span[0], span[1]))
        cache, input_ids = hf.prefill_with_hooks(messages, [(16, compose_hooks(*hooks))])
        k16 = cache.layers[16].keys.float()
        k31 = cache.layers[31].keys.float()
        if mult == 0.0:
            ref16, ref31 = k16.clone(), k31.clone()
        else:
            print(f"    cache delta: L16 keys {((k16 - ref16).norm() / ref16.norm()):.6f}, "
                  f"L31 keys {((k31 - ref31).norm() / ref31.norm()):.6f}, "
                  f"seq_len={input_ids.shape[1]}")
        # Match _batch_generate: the cache must cover only the first seq_len-1
        # tokens, so generate() reprocesses exactly the final prompt token.
        seq_len = input_ids.shape[1]
        trimmed = DynamicCache()
        for li in range(len(cache)):
            trimmed.update(cache.layers[li].keys[:, :, :seq_len - 1, :],
                           cache.layers[li].values[:, :, :seq_len - 1, :], li)
        out = hf.model.generate(input_ids[:, -1:], max_new_tokens=48, do_sample=False,
                                past_key_values=trimmed,
                                pad_token_id=hf.tokenizer.eos_token_id)
        input_ids = input_ids[:, -1:]
        text = hf.tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        if mult == 0.0 and ref_text is None:
            ref_text = text
        tag = "SAME as c=0" if text == ref_text else "DIFFERS from c=0"
        print(f"[{label}] c={mult}: {tag}\n    {text[:130]!r}")
