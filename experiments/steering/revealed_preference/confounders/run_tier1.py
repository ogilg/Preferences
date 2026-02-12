"""Tier 1 experiments: E1 (order counterbalancing), E3 (same-task), E8 (random control).

Loads model once. Runs all experiments on borderline pairs from E2 screening.
Also runs probe direction on borderline pairs as the main effect baseline.
"""

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
OUTPUT_DIR = Path("experiments/steering/revealed_preference/confounders")
TASKS_FILE = Path("activations/gemma_3_27b/completions_with_activations.json")
SCREENING_FILE = OUTPUT_DIR / "e2_screening_results.json"
BORDERLINE_FILE = OUTPUT_DIR / "borderline_pairs.json"

PAIRWISE_TEMPLATE = """You will be given two tasks. Choose which task you would prefer to complete.

Respond with only 'a' or 'b'.

Task A:
{task_a}

Task B:
{task_b}"""

COEFFICIENTS = [-3000.0, -2000.0, -1000.0, 0.0, 1000.0, 2000.0, 3000.0]
N_RESAMPLES = 15
N_RANDOM_DIRS = 5
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
    a_start: int, a_end: int,
    b_start: int, b_end: int,
) -> SteeringHook:
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


def generate_random_orthogonal_directions(direction: np.ndarray, n_dirs: int, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    directions = []
    for _ in range(n_dirs):
        v = rng.randn(len(direction))
        v = v - np.dot(v, direction) * direction
        v = v / np.linalg.norm(v)
        directions.append(v)
    return directions


def load_borderline_pairs() -> list[dict]:
    with open(BORDERLINE_FILE) as f:
        borderline_info = json.load(f)
    with open(SCREENING_FILE) as f:
        screening = json.load(f)

    borderline_indices = {b["pair_idx"] for b in borderline_info}
    pairs_by_idx: dict[int, dict] = {}
    for r in screening:
        idx = r["pair_idx"]
        if idx in borderline_indices and idx not in pairs_by_idx:
            pairs_by_idx[idx] = {
                "pair_idx": idx,
                "task_a_id": r["task_a_id"],
                "task_b_id": r["task_b_id"],
                "task_a_prompt": r["task_a_prompt"],
                "task_b_prompt": r["task_b_prompt"],
                "task_a_origin": r["task_a_origin"],
                "task_b_origin": r["task_b_origin"],
            }

    return [pairs_by_idx[b["pair_idx"]] for b in borderline_info]


def run_steering_trial(model, messages, layer, direction, coef, a_span, b_span, seed):
    torch.manual_seed(seed)
    scaled = torch.tensor(direction * coef, dtype=torch.bfloat16, device="cuda")
    hook = differential_steering(scaled, a_span[0], a_span[1], b_span[0], b_span[1])
    response = model.generate_with_steering(
        messages=messages, layer=layer, steering_hook=hook,
        temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
    )
    return response, parse_choice(response)


def run_e1_probe_on_borderline(model, layer, direction, pairs):
    """E1 baseline + order counterbalancing: probe direction on borderline pairs."""
    print(f"\n{'='*60}")
    print(f"E1: PROBE ON BORDERLINE + ORDER COUNTERBALANCING")
    print(f"  {len(pairs)} pairs x {len(COEFFICIENTS)} coefs x {N_RESAMPLES} resamples x 2 orderings")
    print(f"{'='*60}")

    results = []
    total = len(pairs) * len(COEFFICIENTS) * N_RESAMPLES * 2
    done = 0

    for pair in pairs:
        # Original order: (A, B)
        for ordering in ["original", "swapped"]:
            if ordering == "original":
                ta_prompt = pair["task_a_prompt"]
                tb_prompt = pair["task_b_prompt"]
            else:
                ta_prompt = pair["task_b_prompt"]
                tb_prompt = pair["task_a_prompt"]

            prompt_text = PAIRWISE_TEMPLATE.format(task_a=ta_prompt, task_b=tb_prompt)
            messages: list[Message] = [{"role": "user", "content": prompt_text}]

            try:
                a_span, b_span = find_task_spans(model, messages, ta_prompt, tb_prompt)
            except (ValueError, TypeError) as e:
                print(f"  Skipping pair {pair['pair_idx']} ({ordering}): {e}")
                continue

            for coef in COEFFICIENTS:
                for seed in range(N_RESAMPLES):
                    response, choice = run_steering_trial(
                        model, messages, layer, direction, coef, a_span, b_span, seed,
                    )
                    results.append({
                        "experiment": "E1_order_counterbalance",
                        "pair_idx": pair["pair_idx"],
                        "ordering": ordering,
                        "task_a_id": pair["task_a_id"],
                        "task_b_id": pair["task_b_id"],
                        "coefficient": coef,
                        "seed": seed,
                        "response": response,
                        "choice": choice,
                    })

                    done += 1
                    if done % 200 == 0:
                        print(f"  [{done}/{total}]")

    output_path = OUTPUT_DIR / "e1_order_counterbalance_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_path}")

    # Quick summary
    for ordering in ["original", "swapped"]:
        print(f"\n  {ordering.upper()} ordering:")
        for coef in COEFFICIENTS:
            matching = [r for r in results if r["ordering"] == ordering and r["coefficient"] == coef and r["choice"] is not None]
            if matching:
                n_a = sum(1 for r in matching if r["choice"] == "a")
                p_a = n_a / len(matching)
                print(f"    coef={coef:+7.0f}: P(A)={p_a:.3f} ({n_a}/{len(matching)})")

    return results


def run_e3_same_task(model, layer, direction):
    """E3: Same task as both A and B. If P(A) shifts, it's a position artifact."""
    print(f"\n{'='*60}")
    print("E3: SAME-TASK PAIRS")
    print(f"{'='*60}")

    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)

    # Sample 30 diverse tasks
    rng = random.Random(777)
    by_origin: dict[str, list[dict]] = {}
    for t in all_tasks:
        origin = t["origin"]
        if origin not in by_origin:
            by_origin[origin] = []
        by_origin[origin].append(t)

    tasks = []
    for origin in sorted(by_origin.keys()):
        sampled = rng.sample(by_origin[origin], min(4, len(by_origin[origin])))
        tasks.extend(sampled)
    tasks = tasks[:20]
    print(f"  {len(tasks)} tasks sampled")

    results = []
    total = len(tasks) * len(COEFFICIENTS) * N_RESAMPLES
    done = 0

    for task_idx, task in enumerate(tasks):
        prompt_text = PAIRWISE_TEMPLATE.format(
            task_a=task["task_prompt"], task_b=task["task_prompt"],
        )
        messages: list[Message] = [{"role": "user", "content": prompt_text}]

        try:
            a_span, b_span = find_task_spans(model, messages, task["task_prompt"], task["task_prompt"])
        except (ValueError, TypeError) as e:
            print(f"  Skipping task {task_idx}: {e}")
            continue

        for coef in COEFFICIENTS:
            for seed in range(N_RESAMPLES):
                response, choice = run_steering_trial(
                    model, messages, layer, direction, coef, a_span, b_span, seed,
                )
                results.append({
                    "experiment": "E3_same_task",
                    "task_idx": task_idx,
                    "task_id": task["task_id"],
                    "task_origin": task["origin"],
                    "coefficient": coef,
                    "seed": seed,
                    "response": response,
                    "choice": choice,
                })

                done += 1
                if done % 200 == 0:
                    print(f"  [{done}/{total}]")

    output_path = OUTPUT_DIR / "e3_same_task_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_path}")

    # Quick summary
    for coef in COEFFICIENTS:
        matching = [r for r in results if r["coefficient"] == coef and r["choice"] is not None]
        if matching:
            n_a = sum(1 for r in matching if r["choice"] == "a")
            p_a = n_a / len(matching)
            print(f"  coef={coef:+7.0f}: P(A)={p_a:.3f} ({n_a}/{len(matching)})")

    return results


def run_e8_random_control(model, layer, direction, pairs):
    """E8: Random orthogonal directions on borderline pairs."""
    print(f"\n{'='*60}")
    print(f"E8: RANDOM DIRECTION CONTROL ON BORDERLINE PAIRS")
    print(f"  {len(pairs)} pairs x {N_RANDOM_DIRS} random dirs x 3 coefs x {N_RESAMPLES} resamples")
    print(f"{'='*60}")

    random_dirs = generate_random_orthogonal_directions(direction, N_RANDOM_DIRS)
    test_coefs = [-3000.0, 0.0, 3000.0]  # just extremes + zero

    results = []
    total = len(pairs) * N_RANDOM_DIRS * len(test_coefs) * N_RESAMPLES
    done = 0

    for pair in pairs:
        prompt_text = PAIRWISE_TEMPLATE.format(
            task_a=pair["task_a_prompt"], task_b=pair["task_b_prompt"],
        )
        messages: list[Message] = [{"role": "user", "content": prompt_text}]

        try:
            a_span, b_span = find_task_spans(model, messages, pair["task_a_prompt"], pair["task_b_prompt"])
        except (ValueError, TypeError) as e:
            print(f"  Skipping pair {pair['pair_idx']}: {e}")
            continue

        for dir_idx, rand_dir in enumerate(random_dirs):
            for coef in test_coefs:
                for seed in range(N_RESAMPLES):
                    torch.manual_seed(seed)
                    scaled = torch.tensor(rand_dir * coef, dtype=torch.bfloat16, device="cuda")
                    hook = differential_steering(scaled, a_span[0], a_span[1], b_span[0], b_span[1])
                    response = model.generate_with_steering(
                        messages=messages, layer=layer, steering_hook=hook,
                        temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
                    )
                    choice = parse_choice(response)

                    results.append({
                        "experiment": "E8_random_control",
                        "pair_idx": pair["pair_idx"],
                        "dir_idx": dir_idx,
                        "coefficient": coef,
                        "seed": seed,
                        "response": response,
                        "choice": choice,
                    })

                    done += 1
                    if done % 200 == 0:
                        print(f"  [{done}/{total}]")

    output_path = OUTPUT_DIR / "e8_random_control_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_path}")

    # Quick summary per random direction
    for dir_idx in range(N_RANDOM_DIRS):
        neg = [r for r in results if r["dir_idx"] == dir_idx and r["coefficient"] == -3000.0 and r["choice"] is not None]
        pos = [r for r in results if r["dir_idx"] == dir_idx and r["coefficient"] == 3000.0 and r["choice"] is not None]
        if neg and pos:
            p_a_neg = sum(1 for r in neg if r["choice"] == "a") / len(neg)
            p_a_pos = sum(1 for r in pos if r["choice"] == "a") / len(pos)
            print(f"  Dir {dir_idx}: P(A) {p_a_neg:.3f} → {p_a_pos:.3f} (Δ={p_a_pos-p_a_neg:+.3f})")

    return results


def main():
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded: {model.model_name}")

    pairs = load_borderline_pairs()
    print(f"Loaded {len(pairs)} borderline pairs")

    # Run experiments
    e1_results = run_e1_probe_on_borderline(model, layer, direction, pairs)
    e3_results = run_e3_same_task(model, layer, direction)
    e8_results = run_e8_random_control(model, layer, direction, pairs)

    print(f"\n{'='*60}")
    print("ALL TIER 1 EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"  E1: {len(e1_results)} observations")
    print(f"  E3: {len(e3_results)} observations")
    print(f"  E8: {len(e8_results)} observations")


if __name__ == "__main__":
    main()
