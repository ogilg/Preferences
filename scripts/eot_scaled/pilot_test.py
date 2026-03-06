"""Quick pilot: test 2 orderings to validate infrastructure before full run."""

import json
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import LayerHook
from src.measurement.elicitation.response_format import CompletionChoiceFormat

TASKS_PATH = Path("experiments/patching/eot_scaled/selected_tasks.json")

TEMPLATE = (
    "You will be given two tasks. Choose one and complete it.\n"
    "{format_instruction}\n\n"
    "Task A:\n{task_a}\n\n"
    "Task B:\n{task_b}"
)

MODEL_NAME = "gemma-3-27b"
MAX_NEW_TOKENS = 16


def format_prompt(task_a_prompt: str, task_b_prompt: str) -> str:
    fmt = CompletionChoiceFormat()
    return TEMPLATE.format(
        format_instruction=fmt.format_instruction(),
        task_a=task_a_prompt,
        task_b=task_b_prompt,
    )


def parse_choices(completions: list[str], task_a_prompt: str, task_b_prompt: str) -> list[str]:
    fmt = CompletionChoiceFormat(task_a_prompt=task_a_prompt, task_b_prompt=task_b_prompt)
    return [fmt.parse_sync(c) for c in completions]


def get_prompt_length(model, messages):
    formatted = model.format_messages(messages, add_generation_prompt=True)
    return len(model.tokenizer(formatted, return_tensors="pt").input_ids[0])


def cache_residuals_hook(cache, positions):
    def hook(resid, prompt_len):
        if resid.shape[1] > 1:
            for pos in positions:
                cache[pos] = resid[:, pos, :].clone()
        return resid
    return hook


def inject_residuals_hook(cache, positions):
    def hook(resid, prompt_len):
        if resid.shape[1] > 1:
            for pos in positions:
                resid[:, pos, :] = cache[pos]
        return resid
    return hook


def run_donor_pass(model, messages, positions):
    layer_caches = {}
    hooks = []
    for layer in range(model.n_layers):
        cache = {}
        layer_caches[layer] = cache
        hooks.append((layer, cache_residuals_hook(cache, positions)))
    model.generate_with_hooks_n(messages, layer_hooks=hooks, n=1, temperature=0.0, max_new_tokens=1)
    return layer_caches


def main():
    with open(TASKS_PATH) as f:
        tasks = json.load(f)

    # Pick two tasks with large |delta mu| for a clear test
    task_a = tasks[0]   # mu ~ -10
    task_b = tasks[99]  # mu ~ +10

    print(f"Task A: {task_a['task_id']} (mu={task_a['mu']:.2f})")
    print(f"Task B: {task_b['task_id']} (mu={task_b['mu']:.2f})")

    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded: {model.n_layers} layers")

    # AB ordering
    ab_content = format_prompt(task_a["prompt"], task_b["prompt"])
    ab_messages = [{"role": "user", "content": ab_content}]
    ab_len = get_prompt_length(model, ab_messages)

    # BA ordering (donor)
    ba_content = format_prompt(task_b["prompt"], task_a["prompt"])
    ba_messages = [{"role": "user", "content": ba_content}]
    ba_len = get_prompt_length(model, ba_messages)

    print(f"AB prompt: {ab_len} tokens, BA prompt: {ba_len} tokens")

    # Verify EOT token positions
    formatted = model.format_messages(ab_messages, add_generation_prompt=True)
    tokens = model.tokenizer(formatted, return_tensors="pt").input_ids[0]
    print(f"Last 7 tokens: {[model.tokenizer.decode([t]) for t in tokens[-7:]]}")
    print(f"EOT positions (from end): -5={model.tokenizer.decode([tokens[-5]])!r}, -4={model.tokenizer.decode([tokens[-4]])!r}")

    # Baseline
    t0 = time.time()
    baseline = model.generate_n(ab_messages, n=10, temperature=1.0, max_new_tokens=MAX_NEW_TOKENS)
    t1 = time.time()
    baseline_choices = parse_choices(baseline, task_a["prompt"], task_b["prompt"])
    print(f"\nBaseline (10 trials, {t1-t0:.1f}s): {baseline_choices}")

    # Donor pass
    donor_eot = [ba_len - 5, ba_len - 4]
    recipient_eot = [ab_len - 5, ab_len - 4]

    t2 = time.time()
    donor_cache = run_donor_pass(model, ba_messages, donor_eot)
    t3 = time.time()
    print(f"Donor pass ({t3-t2:.1f}s): cached {len(donor_cache)} layers")

    # Patched
    inject_hooks = []
    for layer in range(model.n_layers):
        mapped = {rp: donor_cache[layer][dp] for rp, dp in zip(recipient_eot, donor_eot)}
        inject_hooks.append((layer, inject_residuals_hook(mapped, recipient_eot)))

    t4 = time.time()
    patched = model.generate_with_hooks_n(
        ab_messages, layer_hooks=inject_hooks, n=10,
        temperature=1.0, max_new_tokens=MAX_NEW_TOKENS,
    )
    t5 = time.time()
    patched_choices = parse_choices(patched, task_a["prompt"], task_b["prompt"])
    print(f"Patched (10 trials, {t5-t4:.1f}s): {patched_choices}")

    total = t5 - t0
    per_ordering = total
    print(f"\nTotal time for 1 ordering: {total:.1f}s")
    print(f"Estimated time for 9900 orderings: {9900 * per_ordering / 3600:.1f}h")
    print(f"  Baseline: {t1-t0:.1f}s, Donor: {t3-t2:.1f}s, Patched: {t5-t4:.1f}s")


if __name__ == "__main__":
    main()
