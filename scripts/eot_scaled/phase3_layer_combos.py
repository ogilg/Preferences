"""Phase 3: Layer combination patching on flipping orderings.

From Phase 2 results, identify top-5 layers by flip rate. Test:
- All pairs of top-5 (10 combinations)
- All triples of top-5 (10 combinations)
- Top-4 together
- Top-5 together
- Full causal window (all layers with flip rate > 10%)

5 trials per ordering per combination at temperature 1.0.
"""

import argparse
import json
import time
from itertools import combinations as combos
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
PHASE2_CHECKPOINT = EXPERIMENT_DIR / "phase2_checkpoint.jsonl"
CHECKPOINT_PATH = EXPERIMENT_DIR / "phase3_checkpoint.jsonl"

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


def inject_multi_layer_hook(cached_values, positions, target_layers, current_layer):
    def hook(resid, prompt_len):
        if resid.shape[1] > 1 and current_layer in target_layers:
            for pos in positions:
                resid[:, pos, :] = cached_values[current_layer][pos]
        return resid
    return hook


def compute_layer_flip_rates() -> dict[int, float]:
    """Compute per-layer flip rates from Phase 2 checkpoint."""
    layer_flips: dict[int, int] = {}
    total = 0

    with open(PHASE2_CHECKPOINT) as f:
        for line in f:
            rec = json.loads(line)
            baseline_chose_a = rec["baseline_chose_a"]
            total += 1

            for layer_str, choices in rec["layer_choices"].items():
                layer = int(layer_str)
                if layer not in layer_flips:
                    layer_flips[layer] = 0

                n_a = choices.count("a")
                n_b = choices.count("b")
                patched_chose_a = n_a > n_b

                if baseline_chose_a != patched_chose_a:
                    layer_flips[layer] += 1

    return {layer: flips / total for layer, flips in layer_flips.items()} if total > 0 else {}


def design_combinations(flip_rates: dict[int, float]) -> list[tuple[str, list[int]]]:
    """Design layer combinations from Phase 2 results."""
    sorted_layers = sorted(flip_rates.keys(), key=lambda l: flip_rates[l], reverse=True)
    top5 = sorted_layers[:5]
    top4 = sorted_layers[:4]

    print(f"Top-5 layers by flip rate:")
    for l in top5:
        print(f"  L{l}: {flip_rates[l]:.1%}")

    combinations_list = []

    # All pairs of top-5
    for a, b in combos(top5, 2):
        combinations_list.append((f"pair_L{a}_L{b}", sorted([a, b])))

    # All triples of top-5
    for triple in combos(top5, 3):
        name = "triple_" + "_".join(f"L{l}" for l in sorted(triple))
        combinations_list.append((name, sorted(triple)))

    # Top-4
    combinations_list.append(("top4_" + "_".join(f"L{l}" for l in sorted(top4)), sorted(top4)))

    # Top-5
    combinations_list.append(("top5_" + "_".join(f"L{l}" for l in sorted(top5)), sorted(top5)))

    # Full causal window: all layers with flip rate > 10%
    causal_window = sorted([l for l, r in flip_rates.items() if r > 0.10])
    if causal_window:
        combinations_list.append(("causal_window", causal_window))

    print(f"\nDesigned {len(combinations_list)} combinations")
    for name, layers in combinations_list:
        print(f"  {name}: {layers}")

    return combinations_list


def find_flipping_orderings() -> list[dict]:
    """Find orderings that flipped under all-layer EOT patching in Phase 1."""
    flipping = []
    with open(PHASE1_CHECKPOINT) as f:
        for line in f:
            rec = json.loads(line)
            baseline = rec["baseline_choices"]
            patched = rec["patched_choices"]

            base_a = baseline.count("a")
            base_b = baseline.count("b")

            if base_a == base_b:
                continue

            baseline_chose_a = base_a > base_b
            patch_a = patched.count("a")
            patched_chose_a = patch_a > patched.count("b")

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

    # Design combinations from Phase 2
    flip_rates = compute_layer_flip_rates()
    combination_designs = design_combinations(flip_rates)

    flipping = find_flipping_orderings()
    print(f"\nFlipping orderings from Phase 1: {len(flipping)}")

    completed = set()
    if args.resume:
        completed = load_completed(CHECKPOINT_PATH)
        print(f"Resuming: {len(completed)} already done")

    remaining = [
        o for o in flipping
        if f"{o['task_a_id']}_{o['task_b_id']}_{o['direction']}" not in completed
    ]
    print(f"Remaining: {len(remaining)}")
    print(f"Total generations: ~{len(remaining) * len(combination_designs) * N_TRIALS}")

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

        # Test each combination
        combo_results = {}
        for combo_name, combo_layers in combination_designs:
            target_set = set(combo_layers)

            hooks = []
            for l in range(n_layers):
                hooks.append((l, inject_multi_layer_hook(donor_cache, recipient_eot, target_set, l)))

            completions = model.generate_with_hooks_n(
                messages, layer_hooks=hooks, n=N_TRIALS,
                temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
            )
            choices = parse_choices(completions, pos_a_prompt, pos_b_prompt)
            combo_results[combo_name] = choices

        record = {
            "task_a_id": ta_id,
            "task_b_id": tb_id,
            "direction": direction,
            "baseline_chose_a": baseline_chose_a,
            "combo_choices": combo_results,
        }

        with open(CHECKPOINT_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (len(remaining) - idx - 1) / rate if rate > 0 else 0

        if (idx + 1) % 20 == 0 or idx == 0:
            print(
                f"[{idx+1}/{len(remaining)}] "
                f"{ta_id[:15]}v{tb_id[:15]} ({direction})  "
                f"({elapsed:.0f}s, {rate:.2f}/s, ETA {eta/3600:.1f}h)"
            )

    # Save combination designs for reference
    design_path = EXPERIMENT_DIR / "phase3_designs.json"
    with open(design_path, "w") as f:
        json.dump(
            [{"name": name, "layers": layers} for name, layers in combination_designs],
            f, indent=2,
        )

    print(f"\nPhase 3 complete. Total time: {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
