"""H2: Differential steering — positive on A, negative on B simultaneously."""

import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import SteeringHook
from src.probes.core.storage import load_probe_direction
from src.types import Message


PROBE_MANIFEST_DIR = Path("results/probes/gemma3_3k_completion_preference")
PROBE_ID = "ridge_L31"
MODEL_NAME = "gemma-3-27b"
OUTPUT_DIR = Path("experiments/steering/revealed_preference")
TASKS_FILE = Path("activations/gemma_3_27b/completions_with_activations.json")

PAIRWISE_TEMPLATE = """You will be given two tasks. Choose which task you would prefer to complete.

Respond with only 'a' or 'b'.

Task A:
{task_a}

Task B:
{task_b}"""

N_PAIRS = 20
N_RESAMPLES = 10
COEFFICIENTS = [-3000.0, -2000.0, -1000.0, 0.0, 1000.0, 2000.0, 3000.0]
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 8


def parse_choice(response: str) -> str | None:
    cleaned = response.strip().lower()
    if cleaned in ("a", "b"):
        return cleaned
    if re.search(r'\btask\s*a\b', cleaned):
        return "a"
    if re.search(r'\btask\s*b\b', cleaned):
        return "b"
    if cleaned and cleaned[0] in ("a", "b"):
        return cleaned[0]
    return None


def differential_steering(
    steering_tensor: torch.Tensor,
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> SteeringHook:
    """Steer POSITIVE on task A tokens, NEGATIVE on task B tokens."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:
            resid[:, a_start:a_end, :] += steering_tensor
            resid[:, b_start:b_end, :] -= steering_tensor
        return resid
    return hook


def find_task_spans(model, messages, task_a_text, task_b_text):
    formatted = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    a_marker = "Task A:\n"
    b_marker = "Task B:\n"
    a_start_char = formatted.index(a_marker) + len(a_marker)
    a_end_char = a_start_char + len(task_a_text)
    b_start_char = formatted.index(b_marker) + len(b_marker)
    b_end_char = b_start_char + len(task_b_text)

    encoded = model.tokenizer(formatted, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]

    def char_to_token_range(char_start, char_end):
        tok_start = tok_end = None
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if s <= char_start < e and tok_start is None:
                tok_start = i
            if s < char_end <= e:
                tok_end = i + 1
        if tok_start is None:
            for i, (s, e) in enumerate(offsets):
                if s >= char_start and tok_start is None:
                    tok_start = i
                    break
        if tok_end is None:
            tok_end = len(offsets)
        return (tok_start, tok_end)

    return char_to_token_range(a_start_char, a_end_char), char_to_token_range(b_start_char, b_end_char)


def sample_task_pairs(n_pairs, seed=42):
    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)
    rng = random.Random(seed)
    by_origin: dict[str, list[dict]] = {}
    for t in all_tasks:
        origin = t["origin"]
        if origin not in by_origin:
            by_origin[origin] = []
        by_origin[origin].append(t)
    pairs = []
    origins = sorted(by_origin.keys())
    for _ in range(n_pairs):
        o1, o2 = rng.sample(origins, 2)
        pairs.append((rng.choice(by_origin[o1]), rng.choice(by_origin[o2])))
    return pairs


def main():
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded: {model.model_name}")

    pairs = sample_task_pairs(N_PAIRS)
    print(f"Sampled {len(pairs)} pairs")

    results = []
    total = len(pairs) * len(COEFFICIENTS) * N_RESAMPLES
    done = 0

    for pair_idx, (task_a, task_b) in enumerate(pairs):
        prompt_text = PAIRWISE_TEMPLATE.format(
            task_a=task_a["task_prompt"], task_b=task_b["task_prompt"],
        )
        messages: list[Message] = [{"role": "user", "content": prompt_text}]

        try:
            a_span, b_span = find_task_spans(model, messages, task_a["task_prompt"], task_b["task_prompt"])
        except (ValueError, TypeError) as e:
            print(f"  Skipping pair {pair_idx}: {e}")
            continue

        for coef in COEFFICIENTS:
            for seed in range(N_RESAMPLES):
                torch.manual_seed(seed)
                scaled = torch.tensor(direction * coef, dtype=torch.bfloat16, device="cuda")
                hook = differential_steering(scaled, a_span[0], a_span[1], b_span[0], b_span[1])

                response = model.generate_with_steering(
                    messages=messages, layer=layer, steering_hook=hook,
                    temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
                )
                choice = parse_choice(response)

                results.append({
                    "hypothesis": "H2_differential",
                    "pair_idx": pair_idx,
                    "task_a_id": task_a["task_id"],
                    "task_b_id": task_b["task_id"],
                    "task_a_origin": task_a["origin"],
                    "task_b_origin": task_b["origin"],
                    "coefficient": coef,
                    "seed": seed,
                    "response": response,
                    "choice": choice,
                })

                done += 1
                if done % 50 == 0 or done == total:
                    print(f"[H2] {done}/{total}")

    # Summary
    print(f"\n{'='*60}")
    print("H2: Differential Steering (pos on A, neg on B)")
    print(f"{'='*60}")
    coefficients = sorted(set(r["coefficient"] for r in results))
    for coef in coefficients:
        matching = [r for r in results if r["coefficient"] == coef and r["choice"] is not None]
        if matching:
            n_a = sum(1 for r in matching if r["choice"] == "a")
            p_a = n_a / len(matching)
            print(f"  coef={coef:+7.0f}: P(A)={p_a:.3f} ({n_a}/{len(matching)})")

    output_path = OUTPUT_DIR / "revealed_preference_h2_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
