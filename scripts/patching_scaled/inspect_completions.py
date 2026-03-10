"""Inspect raw completions from baseline vs patched to verify flips are genuine."""

import json
import random
import torch
from dotenv import load_dotenv
load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.measurement.elicitation.response_format import CompletionChoiceFormat

TEMPLATE = (
    "You will be given two tasks. Choose one and complete it.\n"
    "{format_instruction}\n\n"
    "Task A:\n{task_a}\n\n"
    "Task B:\n{task_b}"
)

with open("experiments/patching/eot_scaled/selected_tasks.json") as f:
    tasks = json.load(f)
task_by_id = {t["task_id"]: t for t in tasks}

# Find flipped orderings from Phase 1, excluding bailbench (harmful) tasks
flipped_pairs = []
with open("experiments/patching/eot_scaled/phase1_checkpoint.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        # Skip harmful tasks
        if "bailbench" in rec["task_a_id"] or "bailbench" in rec["task_b_id"]:
            continue
        if "stresstest" in rec["task_a_id"] or "stresstest" in rec["task_b_id"]:
            continue
        base = rec["baseline_choices"]
        patch = rec["patched_choices"]
        ba, bb = base.count("a"), base.count("b")
        pa, pb = patch.count("a"), patch.count("b")
        if ba + bb == 0 or pa + pb == 0 or ba == bb:
            continue
        if (ba > bb) != (pa > pb):
            flipped_pairs.append(rec)

print(f"Non-harmful flipped pairs: {len(flipped_pairs)}")
random.seed(42)
sample = random.sample(flipped_pairs, min(10, len(flipped_pairs)))

model = HuggingFaceModel("gemma-3-27b", max_new_tokens=64)
fmt = CompletionChoiceFormat()


def cache_hook(cache, positions):
    def hook(resid, prompt_len):
        if resid.shape[1] > 1:
            for pos in positions:
                cache[pos] = resid[:, pos, :].clone()
        return resid
    return hook


def inject_hook(cache, positions):
    def hook(resid, prompt_len):
        if resid.shape[1] > 1:
            for pos in positions:
                resid[:, pos, :] = cache[pos]
        return resid
    return hook


for i, rec in enumerate(sample):
    ta = task_by_id[rec["task_a_id"]]
    tb = task_by_id[rec["task_b_id"]]
    d = rec["direction"]

    if d == "ab":
        pos_a, pos_b = ta["prompt"], tb["prompt"]
    else:
        pos_a, pos_b = tb["prompt"], ta["prompt"]

    content = TEMPLATE.format(format_instruction=fmt.format_instruction(), task_a=pos_a, task_b=pos_b)
    messages = [{"role": "user", "content": content}]
    formatted = model.format_messages(messages, add_generation_prompt=True)
    prompt_len = len(model.tokenizer(formatted, return_tensors="pt").input_ids[0])

    # Baseline
    base_completions = model.generate_n(messages, n=2, temperature=1.0, max_new_tokens=64)

    # Donor (opposite ordering)
    donor_content = TEMPLATE.format(format_instruction=fmt.format_instruction(), task_a=pos_b, task_b=pos_a)
    donor_messages = [{"role": "user", "content": donor_content}]
    donor_formatted = model.format_messages(donor_messages, add_generation_prompt=True)
    donor_len = len(model.tokenizer(donor_formatted, return_tensors="pt").input_ids[0])

    r_eot = [prompt_len - 5, prompt_len - 4]
    d_eot = [donor_len - 5, donor_len - 4]

    # Cache donor residuals
    layer_caches = {}
    hooks = []
    for layer in range(model.n_layers):
        cache = {}
        layer_caches[layer] = cache
        hooks.append((layer, cache_hook(cache, d_eot)))
    model.generate_with_hooks_n(donor_messages, layer_hooks=hooks, n=1, temperature=0.0, max_new_tokens=1)

    # Patched generation
    inject_hooks = []
    for layer in range(model.n_layers):
        mapped = {rp: layer_caches[layer][dp] for rp, dp in zip(r_eot, d_eot)}
        inject_hooks.append((layer, inject_hook(mapped, r_eot)))
    patched_completions = model.generate_with_hooks_n(
        messages, layer_hooks=inject_hooks, n=2, temperature=1.0, max_new_tokens=64
    )

    print(f"\n{'='*80}")
    print(f"Pair {i+1}: {rec['task_a_id']} vs {rec['task_b_id']} ({d})")
    print(f"Task A (position A): {pos_a[:100]}...")
    print(f"Task B (position B): {pos_b[:100]}...")
    print(f"\nBASELINE 1: {base_completions[0]}")
    print(f"BASELINE 2: {base_completions[1]}")
    print(f"\nPATCHED 1:  {patched_completions[0]}")
    print(f"PATCHED 2:  {patched_completions[1]}")
