"""Patch end-of-turn tokens from a donor prompt into a recipient prompt.

Tests whether the model's choice is already encoded at structural tokens
(end_of_turn, start_of_turn, model) before generation starts.
"""

import json
import time
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
RESULTS_PATH = EXPERIMENT_DIR / "eot_patch_results.json"

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

# Tokens to patch: end_of_turn, \n, start_of_turn, model, \n
# These are the last 5 tokens before generation
EOT_WINDOW_SIZES = [1, 3, 5]  # patch last 1, 3, or 5 tokens before generation


def format_prompt(task_a_prompt: str, task_b_prompt: str) -> str:
    fmt = CompletionChoiceFormat()
    return TEMPLATE.format(
        format_instruction=fmt.format_instruction(),
        task_a=task_a_prompt,
        task_b=task_b_prompt,
    )


def parse_choices(completions: list[str], task_a_prompt: str, task_b_prompt: str) -> list[str]:
    fmt = CompletionChoiceFormat(
        task_a_prompt=task_a_prompt, task_b_prompt=task_b_prompt
    )
    return [fmt.parse_sync(c) for c in completions]


def cache_residuals_hook(cache: dict, positions: list[int]) -> LayerHook:
    """Hook that caches residual stream values at given positions."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:  # prefill only
            for pos in positions:
                cache[pos] = resid[:, pos, :].clone()
        return resid
    return hook


def inject_residuals_hook(cache: dict, positions: list[int]) -> LayerHook:
    """Hook that replaces residual stream values at given positions from cache."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:  # prefill only
            for pos in positions:
                resid[:, pos, :] = cache[pos]
        return resid
    return hook


def get_prompt_length(model: HuggingFaceModel, messages: list[dict]) -> int:
    formatted = model.format_messages(messages, add_generation_prompt=True)
    return len(model.tokenizer(formatted, return_tensors="pt").input_ids[0])


def run_donor_pass(model: HuggingFaceModel, messages: list[dict], positions: list[int]) -> dict[int, dict[int, torch.Tensor]]:
    """Run forward pass and cache residuals at given positions for each layer.
    Returns {layer: {position: tensor}}."""
    layer_caches = {}
    hooks = []
    for layer in range(model.n_layers):
        cache = {}
        layer_caches[layer] = cache
        hooks.append((layer, cache_residuals_hook(cache, positions)))

    # Single generation to trigger forward pass
    model.generate_with_hooks_n(
        messages, layer_hooks=hooks, n=1, temperature=0.0, max_new_tokens=1
    )
    return layer_caches


def main():
    with open(TASKS_PATH) as f:
        tasks = json.load(f)

    pairs = list(combinations(tasks, 2))
    print(f"Total pairs: {len(pairs)}")

    print(f"Loading model: {MODEL_NAME}")
    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Layers: {model.n_layers}")

    all_results = []

    for pair_idx, (task_a, task_b) in enumerate(pairs):
        prompt_a_text, prompt_b_text = task_a["prompt"], task_b["prompt"]
        id_a, id_b = task_a["task_id"], task_b["task_id"]

        # AB ordering
        ab_content = format_prompt(prompt_a_text, prompt_b_text)
        ab_messages = [{"role": "user", "content": ab_content}]
        ab_len = get_prompt_length(model, ab_messages)

        # BA ordering
        ba_content = format_prompt(prompt_b_text, prompt_a_text)
        ba_messages = [{"role": "user", "content": ba_content}]
        ba_len = get_prompt_length(model, ba_messages)

        print(f"\n[{pair_idx+1}/{len(pairs)}] {id_a} vs {id_b} (AB={ab_len}tok, BA={ba_len}tok)")

        # Baseline for both orderings
        ab_baseline = model.generate_n(ab_messages, n=N_TRIALS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
        ab_choices = parse_choices(ab_baseline, prompt_a_text, prompt_b_text)

        ba_baseline = model.generate_n(ba_messages, n=N_TRIALS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
        ba_choices = parse_choices(ba_baseline, prompt_b_text, prompt_a_text)

        print(f"  AB baseline: {ab_choices}")
        print(f"  BA baseline: {ba_choices}")

        result = {
            "task_a_id": id_a,
            "task_b_id": id_b,
            "ab_prompt_len": ab_len,
            "ba_prompt_len": ba_len,
            "ab_baseline": ab_choices,
            "ba_baseline": ba_choices,
            "conditions": {},
        }

        for window_size in EOT_WINDOW_SIZES:
            # Positions to patch: last `window_size` tokens before generation
            ab_positions = list(range(ab_len - window_size, ab_len))
            ba_positions = list(range(ba_len - window_size, ba_len))

            # Cache BA residuals at EOT positions
            ba_cache = run_donor_pass(model, ba_messages, ba_positions)

            # Inject BA residuals into AB forward pass at corresponding positions
            # (map BA positions to AB positions by offset from end)
            inject_hooks = []
            for layer in range(model.n_layers):
                mapped_cache = {}
                for ba_pos, ab_pos in zip(ba_positions, ab_positions):
                    mapped_cache[ab_pos] = ba_cache[layer][ba_pos]
                inject_hooks.append((layer, inject_residuals_hook(mapped_cache, ab_positions)))

            patched_completions = model.generate_with_hooks_n(
                ab_messages, layer_hooks=inject_hooks, n=N_TRIALS,
                temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
            )
            patched_choices = parse_choices(patched_completions, prompt_a_text, prompt_b_text)

            # Also do the reverse: inject AB into BA
            ab_cache = run_donor_pass(model, ab_messages, ab_positions)

            reverse_hooks = []
            for layer in range(model.n_layers):
                mapped_cache = {}
                for ab_pos, ba_pos in zip(ab_positions, ba_positions):
                    mapped_cache[ba_pos] = ab_cache[layer][ab_pos]
                reverse_hooks.append((layer, inject_residuals_hook(mapped_cache, ba_positions)))

            reverse_completions = model.generate_with_hooks_n(
                ba_messages, layer_hooks=reverse_hooks, n=N_TRIALS,
                temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
            )
            reverse_choices = parse_choices(reverse_completions, prompt_b_text, prompt_a_text)

            cond_name = f"eot_patch_{window_size}"
            result["conditions"][cond_name] = {
                "window_size": window_size,
                "ab_patched_with_ba": patched_choices,
                "ba_patched_with_ab": reverse_choices,
            }
            print(f"  {cond_name}: AB←BA={patched_choices}  BA←AB={reverse_choices}")

        all_results.append(result)

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
