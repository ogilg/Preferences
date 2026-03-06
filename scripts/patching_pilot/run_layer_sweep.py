"""Per-layer EOT patching sweep: which layers causally encode the choice at <end_of_turn>?

Phase 1: For each layer individually, patch the EOT residuals from the donor prompt.
Only tests orderings that flipped under all-layer EOT patching (from eot_patch_results.json).
Uses 1 trial per ordering for speed.
"""

import json
from itertools import combinations
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import LayerHook
from src.measurement.elicitation.response_format import CompletionChoiceFormat

EXPERIMENT_DIR = Path("experiments/patching/pilot")
TASKS_PATH = EXPERIMENT_DIR / "selected_tasks.json"
EOT_RESULTS_PATH = EXPERIMENT_DIR / "eot_patch_results.json"
RESULTS_PATH = EXPERIMENT_DIR / "layer_sweep_results.json"

TEMPLATE = (
    "You will be given two tasks. Choose one and complete it.\n"
    "{format_instruction}\n\n"
    "Task A:\n{task_a}\n\n"
    "Task B:\n{task_b}"
)

MAX_NEW_TOKENS = 16
MODEL_NAME = "gemma-3-27b"


def format_prompt(task_a_prompt: str, task_b_prompt: str) -> str:
    fmt = CompletionChoiceFormat()
    return TEMPLATE.format(
        format_instruction=fmt.format_instruction(),
        task_a=task_a_prompt,
        task_b=task_b_prompt,
    )


def parse_choice(completion: str, task_a_prompt: str, task_b_prompt: str) -> str:
    fmt = CompletionChoiceFormat(task_a_prompt=task_a_prompt, task_b_prompt=task_b_prompt)
    return fmt.parse_sync(completion)


def get_prompt_length(model, messages):
    formatted = model.format_messages(messages, add_generation_prompt=True)
    return len(model.tokenizer(formatted, return_tensors="pt").input_ids[0])


def find_flipping_orderings():
    """Find orderings that flipped under all-layer EOT patch (window 5)."""
    with open(EOT_RESULTS_PATH) as f:
        eot_results = json.load(f)

    flipping = []
    for entry in eot_results:
        ta, tb = entry["task_a_id"], entry["task_b_id"]
        cond = entry["conditions"]["eot_patch_5"]

        # AB direction
        ab_base = entry["ab_baseline"]
        ab_patch = cond["ab_patched_with_ba"]
        p_a_base = ab_base.count("a") / len(ab_base)
        p_a_patch = ab_patch.count("a") / len(ab_patch)
        if abs(p_a_patch - p_a_base) >= 0.5:
            flipping.append({"task_a_id": ta, "task_b_id": tb, "direction": "ab",
                             "baseline_chose_a": p_a_base > 0.5})

        # BA direction
        ba_base = entry["ba_baseline"]
        ba_patch = cond["ba_patched_with_ab"]
        p_a_base = ba_base.count("a") / len(ba_base)
        p_a_patch = ba_patch.count("a") / len(ba_patch)
        if abs(p_a_patch - p_a_base) >= 0.5:
            flipping.append({"task_a_id": ta, "task_b_id": tb, "direction": "ba",
                             "baseline_chose_a": p_a_base > 0.5})

    return flipping


def cache_all_layers_hook(layer_cache: dict, positions: list[int], layer_idx: int):
    """Hook that caches residuals at given positions for a specific layer."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:
            layer_cache[layer_idx] = {pos: resid[:, pos, :].clone() for pos in positions}
        return resid
    return hook


def inject_single_layer_hook(cached_values: dict, positions: list[int], target_layer: int, current_layer: int):
    """Hook that injects cached residuals only at the target layer."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1 and current_layer == target_layer:
            for pos in positions:
                resid[:, pos, :] = cached_values[pos]
        return resid
    return hook


def main():
    with open(TASKS_PATH) as f:
        tasks = json.load(f)
    task_by_id = {t["task_id"]: t for t in tasks}

    flipping = find_flipping_orderings()
    print(f"Flipping orderings to test: {len(flipping)}")

    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    n_layers = model.n_layers
    print(f"Layers: {n_layers}")

    # Per-layer flip counts
    layer_flips = {layer: 0 for layer in range(n_layers)}
    layer_correct = {layer: 0 for layer in range(n_layers)}
    total_tested = 0

    for idx, ordering_info in enumerate(flipping):
        ta_id = ordering_info["task_a_id"]
        tb_id = ordering_info["task_b_id"]
        direction = ordering_info["direction"]
        baseline_chose_a = ordering_info["baseline_chose_a"]

        task_a = task_by_id[ta_id]
        task_b = task_by_id[tb_id]

        if direction == "ab":
            recipient_prompt_a, recipient_prompt_b = task_a["prompt"], task_b["prompt"]
            donor_prompt_a, donor_prompt_b = task_b["prompt"], task_a["prompt"]
        else:
            recipient_prompt_a, recipient_prompt_b = task_b["prompt"], task_a["prompt"]
            donor_prompt_a, donor_prompt_b = task_a["prompt"], task_b["prompt"]

        recipient_content = format_prompt(recipient_prompt_a, recipient_prompt_b)
        recipient_messages = [{"role": "user", "content": recipient_content}]
        recipient_len = get_prompt_length(model, recipient_messages)

        donor_content = format_prompt(donor_prompt_a, donor_prompt_b)
        donor_messages = [{"role": "user", "content": donor_content}]
        donor_len = get_prompt_length(model, donor_messages)

        # EOT positions (2 tokens: <end_of_turn> \n)
        recipient_eot = [recipient_len - 5, recipient_len - 4]
        donor_eot = [donor_len - 5, donor_len - 4]

        print(f"[{idx+1}/{len(flipping)}] {ta_id[:20]} vs {tb_id[:20]} ({direction})", end="", flush=True)

        # Single donor pass: cache all layers
        donor_cache = {}
        hooks = []
        for layer in range(n_layers):
            hooks.append((layer, cache_all_layers_hook(donor_cache, donor_eot, layer)))
        model.generate_with_hooks_n(donor_messages, layer_hooks=hooks, n=1, temperature=0.0, max_new_tokens=1)

        # Per-layer patching: one generation per layer
        layer_choices = {}
        for layer in range(n_layers):
            # Map donor positions to recipient positions
            mapped = {rp: donor_cache[layer][dp] for rp, dp in zip(recipient_eot, donor_eot)}

            hooks = []
            for l in range(n_layers):
                if l == layer:
                    hooks.append((l, inject_single_layer_hook(mapped, recipient_eot, layer, l)))
                else:
                    def noop(resid, prompt_len):
                        return resid
                    hooks.append((l, noop))

            completions = model.generate_with_hooks_n(
                recipient_messages, layer_hooks=hooks, n=1, temperature=0.0, max_new_tokens=MAX_NEW_TOKENS
            )
            choice = parse_choice(completions[0], recipient_prompt_a, recipient_prompt_b)
            layer_choices[layer] = choice

            # Check if flipped
            flipped = (baseline_chose_a and choice == "b") or (not baseline_chose_a and choice == "a")
            if flipped:
                layer_flips[layer] += 1
                layer_correct[layer] += 1
            elif choice != ("a" if baseline_chose_a else "b"):
                layer_correct[layer] += 1

        total_tested += 1

        # Print summary for this ordering
        flipped_layers = [l for l in range(n_layers) if layer_choices[l] != ("a" if baseline_chose_a else "b")]
        if flipped_layers:
            print(f"  flipped at layers: {flipped_layers}")
        else:
            print(f"  no single layer flips")

    # Save results
    results = {
        "n_orderings_tested": total_tested,
        "n_layers": n_layers,
        "per_layer": {
            str(layer): {
                "flips": layer_flips[layer],
                "flip_rate": layer_flips[layer] / total_tested if total_tested > 0 else 0,
            }
            for layer in range(n_layers)
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")

    # Print top layers
    print("\nTop 15 layers by flip rate:")
    sorted_layers = sorted(range(n_layers), key=lambda l: layer_flips[l], reverse=True)
    for layer in sorted_layers[:15]:
        print(f"  L{layer}: {layer_flips[layer]}/{total_tested} ({layer_flips[layer]/total_tested:.0%})")


if __name__ == "__main__":
    main()
