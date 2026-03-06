"""Phase 2: Per-layer EOT sweep on flipping orderings from Phase 1.

Layers tested:
- Every layer in 20-45 (26 layers)
- Every 3rd layer outside: 0,3,6,...,18 and 48,51,54,57,60 (12 layers)
Total: 38 layers

5 trials per layer at temperature 1.0, max_new_tokens=16.
Pre-caches all layers' donor residuals in a single forward pass.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import LayerHook
from src.measurement.elicitation.response_format import CompletionChoiceFormat

EXPERIMENT_DIR = Path("experiments/patching/eot_scaled")
TASKS_PATH = EXPERIMENT_DIR / "selected_tasks.json"
PHASE1_CHECKPOINT = EXPERIMENT_DIR / "phase1_checkpoint.jsonl"
CHECKPOINT_PATH = EXPERIMENT_DIR / "phase2_checkpoint.jsonl"

TEMPLATE = (
    "You will be given two tasks. Choose one and complete it.\n"
    "{format_instruction}\n\n"
    "Task A:\n{task_a}\n\n"
    "Task B:\n{task_b}"
)

N_TRIALS = 5
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 16
MODEL_NAME = "gemma-3-27b"

# Layers to sweep
LAYERS_IN_WINDOW = list(range(20, 46))  # 20-45 inclusive
LAYERS_OUTSIDE = list(range(0, 19, 3)) + list(range(48, 61, 3))  # 0,3,6,...,18 and 48,51,54,57,60
SWEEP_LAYERS = sorted(set(LAYERS_IN_WINDOW + LAYERS_OUTSIDE))


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


def cache_residuals_hook(cache, positions, layer_idx):
    def hook(resid, prompt_len):
        if resid.shape[1] > 1:
            cache[layer_idx] = {pos: resid[:, pos, :].clone() for pos in positions}
        return resid
    return hook


def inject_single_layer_hook(cached_values, positions, target_layer, current_layer):
    def hook(resid, prompt_len):
        if resid.shape[1] > 1 and current_layer == target_layer:
            for pos in positions:
                resid[:, pos, :] = cached_values[pos]
        return resid
    return hook


def find_flipping_orderings() -> list[dict]:
    """Find orderings that flipped under all-layer EOT patching in Phase 1."""
    flipping = []
    with open(PHASE1_CHECKPOINT) as f:
        for line in f:
            rec = json.loads(line)
            baseline = rec["baseline_choices"]
            patched = rec["patched_choices"]

            # Majority choice
            base_a = baseline.count("a")
            base_b = baseline.count("b")
            patch_a = patched.count("a")
            patch_b = patched.count("b")

            # Skip if baseline is ambiguous (ties)
            if base_a == base_b:
                continue

            baseline_chose_a = base_a > base_b

            # Flip = majority choice changed
            patched_chose_a = patch_a > patch_b
            if baseline_chose_a != patched_chose_a:
                flipping.append({
                    "task_a_id": rec["task_a_id"],
                    "task_b_id": rec["task_b_id"],
                    "direction": rec["direction"],
                    "baseline_chose_a": baseline_chose_a,
                })
    return flipping


def load_completed(checkpoint_path: Path) -> set[str]:
    completed = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                rec = json.loads(line)
                key = f"{rec['task_a_id']}_{rec['task_b_id']}_{rec['direction']}"
                completed.add(key)
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(TASKS_PATH) as f:
        tasks = json.load(f)
    task_by_id = {t["task_id"]: t for t in tasks}

    flipping = find_flipping_orderings()
    print(f"Flipping orderings from Phase 1: {len(flipping)}")
    print(f"Layers to sweep: {len(SWEEP_LAYERS)} — {SWEEP_LAYERS}")

    completed = set()
    if args.resume:
        completed = load_completed(CHECKPOINT_PATH)
        print(f"Resuming: {len(completed)} already done")

    remaining = [
        o for o in flipping
        if f"{o['task_a_id']}_{o['task_b_id']}_{o['direction']}" not in completed
    ]
    print(f"Remaining: {len(remaining)}")

    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    n_layers = model.n_layers
    print(f"Model loaded: {n_layers} layers")

    t0 = time.time()
    for idx, ordering in enumerate(remaining):
        ta_id = ordering["task_a_id"]
        tb_id = ordering["task_b_id"]
        direction = ordering["direction"]
        baseline_chose_a = ordering["baseline_chose_a"]

        task_a = task_by_id[ta_id]
        task_b = task_by_id[tb_id]

        if direction == "ab":
            pos_a_prompt, pos_b_prompt = task_a["prompt"], task_b["prompt"]
        else:
            pos_a_prompt, pos_b_prompt = task_b["prompt"], task_a["prompt"]

        # Recipient
        content = format_prompt(pos_a_prompt, pos_b_prompt)
        messages = [{"role": "user", "content": content}]
        prompt_len = get_prompt_length(model, messages)

        # Donor (opposite ordering)
        donor_content = format_prompt(pos_b_prompt, pos_a_prompt)
        donor_messages = [{"role": "user", "content": donor_content}]
        donor_len = get_prompt_length(model, donor_messages)

        recipient_eot = [prompt_len - 5, prompt_len - 4]
        donor_eot = [donor_len - 5, donor_len - 4]

        # Single donor pass: cache all layers
        donor_cache = {}
        hooks = []
        for layer in range(n_layers):
            hooks.append((layer, cache_residuals_hook(donor_cache, donor_eot, layer)))
        model.generate_with_hooks_n(donor_messages, layer_hooks=hooks, n=1, temperature=0.0, max_new_tokens=1)

        # Per-layer patching
        layer_results = {}
        for target_layer in SWEEP_LAYERS:
            mapped = {rp: donor_cache[target_layer][dp] for rp, dp in zip(recipient_eot, donor_eot)}

            hooks = []
            for l in range(n_layers):
                hooks.append((l, inject_single_layer_hook(mapped, recipient_eot, target_layer, l)))

            completions = model.generate_with_hooks_n(
                messages, layer_hooks=hooks, n=N_TRIALS,
                temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
            )
            choices = parse_choices(completions, pos_a_prompt, pos_b_prompt)
            layer_results[str(target_layer)] = choices

        record = {
            "task_a_id": ta_id,
            "task_b_id": tb_id,
            "direction": direction,
            "baseline_chose_a": baseline_chose_a,
            "layer_choices": layer_results,
        }

        with open(CHECKPOINT_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (len(remaining) - idx - 1) / rate if rate > 0 else 0

        if (idx + 1) % 20 == 0 or idx == 0:
            # Count flips across layers for this ordering
            n_flipping_layers = sum(
                1 for choices in layer_results.values()
                if (baseline_chose_a and choices.count("b") > choices.count("a")) or
                   (not baseline_chose_a and choices.count("a") > choices.count("b"))
            )
            print(
                f"[{idx+1}/{len(remaining)}] "
                f"{ta_id[:15]}v{tb_id[:15]} ({direction}) "
                f"flipping layers: {n_flipping_layers}/{len(SWEEP_LAYERS)}  "
                f"({elapsed:.0f}s, {rate:.2f}/s, ETA {eta/3600:.1f}h)"
            )

    print(f"\nPhase 2 complete. Total time: {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
