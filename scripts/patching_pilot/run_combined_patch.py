"""Combined patching: full block swap + EOT patch, and EOT-only (2 tokens)."""

import json
import time
from itertools import combinations
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import LayerHook, swap_spans
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.steering.tokenization import find_text_span

EXPERIMENT_DIR = Path("experiments/patching/pilot")
TASKS_PATH = EXPERIMENT_DIR / "selected_tasks.json"
RESULTS_PATH = EXPERIMENT_DIR / "combined_patch_results.json"

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


def compute_bos_offset(model: HuggingFaceModel) -> int:
    test_text = "Hello world"
    ids_with = model.tokenizer(test_text, return_tensors="pt").input_ids
    ids_without = model.tokenizer(test_text, return_offsets_mapping=True, add_special_tokens=False)["input_ids"]
    return len(ids_with[0]) - len(ids_without)


def find_full_block_spans(model, formatted_chat, task_a_text, task_b_text, bos_offset):
    block_a = f"Task A:\n{task_a_text}"
    a_span = find_text_span(model.tokenizer, formatted_chat, block_a)
    a_span = (a_span[0] + bos_offset, a_span[1] + bos_offset)
    block_b = f"Task B:\n{task_b_text}"
    b_span = find_text_span(model.tokenizer, formatted_chat, block_b)
    b_span = (b_span[0] + bos_offset, b_span[1] + bos_offset)
    return a_span, b_span


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


def combined_hook(swap_a_start, swap_a_end, swap_b_start, swap_b_end, cache, inject_positions):
    """Swap task blocks AND inject EOT residuals from donor."""
    swap_len = min(swap_a_end - swap_a_start, swap_b_end - swap_b_start)
    a_swap_start = swap_a_end - swap_len
    b_swap_start = swap_b_end - swap_len

    def hook(resid, prompt_len):
        if resid.shape[1] > 1:
            # Block swap
            a_act = resid[:, a_swap_start:swap_a_end, :].clone()
            resid[:, a_swap_start:swap_a_end, :] = resid[:, b_swap_start:swap_b_end, :]
            resid[:, b_swap_start:swap_b_end, :] = a_act
            # EOT inject
            for pos in inject_positions:
                resid[:, pos, :] = cache[pos]
        return resid
    return hook


def get_prompt_length(model, messages):
    formatted = model.format_messages(messages, add_generation_prompt=True)
    return len(model.tokenizer(formatted, return_tensors="pt").input_ids[0])


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

    pairs = list(combinations(tasks, 2))
    print(f"Total pairs: {len(pairs)}")

    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Layers: {model.n_layers}")
    bos_offset = compute_bos_offset(model)

    all_results = []

    for pair_idx, (task_a, task_b) in enumerate(pairs):
        prompt_a, prompt_b = task_a["prompt"], task_b["prompt"]
        id_a, id_b = task_a["task_id"], task_b["task_id"]

        ab_content = format_prompt(prompt_a, prompt_b)
        ab_messages = [{"role": "user", "content": ab_content}]
        ab_len = get_prompt_length(model, ab_messages)
        ab_chat = model.format_messages(ab_messages, add_generation_prompt=True)

        ba_content = format_prompt(prompt_b, prompt_a)
        ba_messages = [{"role": "user", "content": ba_content}]
        ba_len = get_prompt_length(model, ba_messages)
        ba_chat = model.format_messages(ba_messages, add_generation_prompt=True)

        print(f"\n[{pair_idx+1}/{len(pairs)}] {id_a} vs {id_b}")

        # Baselines
        ab_base = model.generate_n(ab_messages, n=N_TRIALS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
        ab_choices = parse_choices(ab_base, prompt_a, prompt_b)
        ba_base = model.generate_n(ba_messages, n=N_TRIALS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
        ba_choices = parse_choices(ba_base, prompt_b, prompt_a)
        print(f"  AB base: {ab_choices}  BA base: {ba_choices}")

        result = {
            "task_a_id": id_a, "task_b_id": id_b,
            "ab_prompt_len": ab_len, "ba_prompt_len": ba_len,
            "ab_baseline": ab_choices, "ba_baseline": ba_choices,
            "conditions": {},
        }

        # --- Condition 1: EOT-only (2 tokens: <end_of_turn> \n) ---
        eot_positions_ab = [ab_len - 5, ab_len - 4]  # <end_of_turn> and \n
        eot_positions_ba = [ba_len - 5, ba_len - 4]

        ba_cache_2 = run_donor_pass(model, ba_messages, eot_positions_ba)
        inject_hooks = []
        for layer in range(model.n_layers):
            mapped = {ab_pos: ba_cache_2[layer][ba_pos] for ab_pos, ba_pos in zip(eot_positions_ab, eot_positions_ba)}
            inject_hooks.append((layer, inject_residuals_hook(mapped, eot_positions_ab)))
        ab_patched_2 = model.generate_with_hooks_n(ab_messages, layer_hooks=inject_hooks, n=N_TRIALS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
        ab_choices_2 = parse_choices(ab_patched_2, prompt_a, prompt_b)

        ab_cache_2 = run_donor_pass(model, ab_messages, eot_positions_ab)
        inject_hooks = []
        for layer in range(model.n_layers):
            mapped = {ba_pos: ab_cache_2[layer][ab_pos] for ab_pos, ba_pos in zip(eot_positions_ab, eot_positions_ba)}
            inject_hooks.append((layer, inject_residuals_hook(mapped, eot_positions_ba)))
        ba_patched_2 = model.generate_with_hooks_n(ba_messages, layer_hooks=inject_hooks, n=N_TRIALS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
        ba_choices_2 = parse_choices(ba_patched_2, prompt_b, prompt_a)

        result["conditions"]["eot_only_2"] = {
            "ab_patched_with_ba": ab_choices_2,
            "ba_patched_with_ab": ba_choices_2,
        }
        print(f"  eot_only_2: AB←BA={ab_choices_2}  BA←AB={ba_choices_2}")

        # --- Condition 2: Full block swap + EOT patch 5 ---
        ab_a_span, ab_b_span = find_full_block_spans(model, ab_chat, prompt_a, prompt_b, bos_offset)
        eot5_ab = list(range(ab_len - 5, ab_len))
        eot5_ba = list(range(ba_len - 5, ba_len))

        ba_cache_5 = run_donor_pass(model, ba_messages, eot5_ba)
        combo_hooks = []
        for layer in range(model.n_layers):
            mapped = {ab_pos: ba_cache_5[layer][ba_pos] for ab_pos, ba_pos in zip(eot5_ab, eot5_ba)}
            combo_hooks.append((layer, combined_hook(
                ab_a_span[0], ab_a_span[1], ab_b_span[0], ab_b_span[1],
                mapped, eot5_ab
            )))
        ab_combo = model.generate_with_hooks_n(ab_messages, layer_hooks=combo_hooks, n=N_TRIALS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
        ab_combo_choices = parse_choices(ab_combo, prompt_a, prompt_b)

        ba_a_span, ba_b_span = find_full_block_spans(model, ba_chat, prompt_b, prompt_a, bos_offset)
        ab_cache_5 = run_donor_pass(model, ab_messages, eot5_ab)
        combo_hooks = []
        for layer in range(model.n_layers):
            mapped = {ba_pos: ab_cache_5[layer][ab_pos] for ab_pos, ba_pos in zip(eot5_ab, eot5_ba)}
            combo_hooks.append((layer, combined_hook(
                ba_a_span[0], ba_a_span[1], ba_b_span[0], ba_b_span[1],
                mapped, eot5_ba
            )))
        ba_combo = model.generate_with_hooks_n(ba_messages, layer_hooks=combo_hooks, n=N_TRIALS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
        ba_combo_choices = parse_choices(ba_combo, prompt_b, prompt_a)

        result["conditions"]["block_plus_eot5"] = {
            "ab_patched_with_ba": ab_combo_choices,
            "ba_patched_with_ab": ba_combo_choices,
        }
        print(f"  block+eot5: AB←BA={ab_combo_choices}  BA←AB={ba_combo_choices}")

        all_results.append(result)

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
