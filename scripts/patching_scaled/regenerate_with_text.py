"""Regenerate baseline + patched completions for a sample of flipped orderings, saving raw text."""

import json
import random
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.measurement.elicitation.response_format import CompletionChoiceFormat

EXPERIMENT_DIR = Path("experiments/patching/eot_scaled")
TASKS_PATH = EXPERIMENT_DIR / "selected_tasks.json"
OUTPUT_PATH = EXPERIMENT_DIR / "flip_completions_sample_v2.json"

TEMPLATE = (
    "You will be given two tasks. Choose one and complete it.\n"
    "{format_instruction}\n\n"
    "Task A:\n{task_a}\n\n"
    "Task B:\n{task_b}"
)

N_SAMPLE = 200
MAX_NEW_TOKENS = 128
MODEL_NAME = "gemma-3-27b"

with open(TASKS_PATH) as f:
    tasks = json.load(f)
task_by_id = {t["task_id"]: t for t in tasks}

# Find all flipped orderings, excluding harmful tasks
flipped = []
with open(EXPERIMENT_DIR / "phase1_checkpoint.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        base = rec["baseline_choices"]
        patch = rec["patched_choices"]
        ba, bb = base.count("a"), base.count("b")
        pa, pb = patch.count("a"), patch.count("b")
        if ba + bb == 0 or pa + pb == 0 or ba == bb:
            continue
        if (ba > bb) != (pa > pb):
            flipped.append(rec)

print(f"Non-harmful flipped orderings: {len(flipped)}")
random.seed(42)
sample = random.sample(flipped, min(N_SAMPLE, len(flipped)))
print(f"Sampled: {len(sample)}")

model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
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


results = []
t0 = time.time()

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

    # Baseline (1 completion)
    base_comp = model.generate_n(messages, n=1, temperature=1.0, max_new_tokens=MAX_NEW_TOKENS)

    # Donor (opposite ordering)
    donor_content = TEMPLATE.format(format_instruction=fmt.format_instruction(), task_a=pos_b, task_b=pos_a)
    donor_messages = [{"role": "user", "content": donor_content}]
    donor_formatted = model.format_messages(donor_messages, add_generation_prompt=True)
    donor_len = len(model.tokenizer(donor_formatted, return_tensors="pt").input_ids[0])

    r_eot = [prompt_len - 5, prompt_len - 4]
    d_eot = [donor_len - 5, donor_len - 4]

    # Cache donor
    layer_caches = {}
    hooks = []
    for layer in range(model.n_layers):
        cache = {}
        layer_caches[layer] = cache
        hooks.append((layer, cache_hook(cache, d_eot)))
    model.generate_with_hooks_n(donor_messages, layer_hooks=hooks, n=1, temperature=0.0, max_new_tokens=1)

    # Patched (1 completion)
    inject_hooks = []
    for layer in range(model.n_layers):
        mapped = {rp: layer_caches[layer][dp] for rp, dp in zip(r_eot, d_eot)}
        inject_hooks.append((layer, inject_hook(mapped, r_eot)))
    patched_comp = model.generate_with_hooks_n(
        messages, layer_hooks=inject_hooks, n=1, temperature=1.0, max_new_tokens=MAX_NEW_TOKENS
    )

    baseline_chose_a = rec["baseline_choices"].count("a") > rec["baseline_choices"].count("b")

    results.append({
        "task_a_id": rec["task_a_id"],
        "task_b_id": rec["task_b_id"],
        "direction": d,
        "pos_a_prompt": pos_a[:200],
        "pos_b_prompt": pos_b[:200],
        "baseline_chose_a": baseline_chose_a,
        "baseline_text": base_comp[0],
        "patched_text": patched_comp[0],
    })

    if (i + 1) % 20 == 0:
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(sample)}] {elapsed:.0f}s ({(i+1)/elapsed:.1f}/s)")

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} results to {OUTPUT_PATH}")
print(f"Total time: {(time.time()-t0)/60:.1f}m")
