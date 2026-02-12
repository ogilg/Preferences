"""Followup v2: All conditions on utility-matched pairs.

Conditions:
1. Probe differential (both orderings) — main effect + position control
2. Same-task — pure position artifact
3. Header-only — position vs content diagnostic
4. Random directions — specificity control

Usage:
    python experiments/steering/revealed_preference/confounders/followup_v2/run_experiment.py --pilot
    python experiments/steering/revealed_preference/confounders/followup_v2/run_experiment.py --condition probe
    python experiments/steering/revealed_preference/confounders/followup_v2/run_experiment.py --condition all
"""

import argparse
import json
import random
import re
import time
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
OUTPUT_DIR = Path("experiments/steering/revealed_preference/confounders/followup_v2")
PAIRS_FILE = OUTPUT_DIR / "utility_matched_pairs.json"
TASKS_FILE = Path("activations/gemma_3_27b/completions_with_activations.json")

PAIRWISE_TEMPLATE = """You will be given two tasks. Choose which task you would prefer to complete.

Respond with only 'a' or 'b'.

Task A:
{task_a}

Task B:
{task_b}"""

COEFFICIENTS = [-3000.0, -1500.0, 0.0, 1500.0, 3000.0]
N_RESAMPLES = 15
N_RANDOM_DIRS = 20
N_SAME_TASK = 30
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


def find_header_spans(model, messages):
    """Find token spans for just the "Task A:\n" and "Task B:\n" headers."""
    formatted = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    a_header = "Task A:\n"
    b_header = "Task B:\n"
    a_start_char = formatted.index(a_header)
    a_end_char = a_start_char + len(a_header)
    b_start_char = formatted.index(b_header)
    b_end_char = b_start_char + len(b_header)

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


def differential_steering(steering_tensor, a_start, a_end, b_start, b_end) -> SteeringHook:
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:
            resid[:, a_start:a_end, :] += steering_tensor
            resid[:, b_start:b_end, :] -= steering_tensor
        return resid
    return hook


def generate_random_orthogonal_directions(direction: np.ndarray, n_dirs: int, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    directions = []
    for _ in range(n_dirs):
        v = rng.randn(len(direction))
        v = v - np.dot(v, direction) * direction
        v = v / np.linalg.norm(v)
        directions.append(v)
    return directions


def run_trial(model, messages, layer, direction, coef, a_span, b_span, seed):
    torch.manual_seed(seed)
    scaled = torch.tensor(direction * coef, dtype=torch.bfloat16, device="cuda")
    hook = differential_steering(scaled, a_span[0], a_span[1], b_span[0], b_span[1])
    response = model.generate_with_steering(
        messages=messages, layer=layer, steering_hook=hook,
        temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
    )
    return response, parse_choice(response)


def load_pairs(subset: str | None = None, max_pairs: int | None = None) -> list[dict]:
    with open(PAIRS_FILE) as f:
        pairs = json.load(f)
    if subset == "borderline":
        pairs = [p for p in pairs if p["delta_mu_bin"] == "0-1"]
    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def save_results(results: list[dict], name: str):
    path = OUTPUT_DIR / f"{name}_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {path}")


def run_probe_differential(model, layer, direction, pairs, coefficients, n_resamples):
    """Probe differential in both orderings on all pairs."""
    print(f"\n{'='*60}")
    print(f"PROBE DIFFERENTIAL (both orderings)")
    print(f"  {len(pairs)} pairs x {len(coefficients)} coefs x {n_resamples} resamples x 2 orderings")
    total = len(pairs) * len(coefficients) * n_resamples * 2
    print(f"  Total trials: {total}")
    print(f"{'='*60}")

    results = []
    done = 0
    t0 = time.time()

    for pair in pairs:
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

            for coef in coefficients:
                for seed in range(n_resamples):
                    response, choice = run_trial(
                        model, messages, layer, direction, coef, a_span, b_span, seed,
                    )
                    results.append({
                        "condition": "probe_differential",
                        "pair_idx": pair["pair_idx"],
                        "ordering": ordering,
                        "task_a_id": pair["task_a_id"],
                        "task_b_id": pair["task_b_id"],
                        "delta_mu": pair["delta_mu"],
                        "delta_mu_bin": pair["delta_mu_bin"],
                        "coefficient": coef,
                        "seed": seed,
                        "response": response,
                        "choice": choice,
                    })

                    done += 1
                    if done % 500 == 0:
                        elapsed = time.time() - t0
                        rate = done / elapsed
                        eta = (total - done) / rate / 60
                        print(f"  [{done}/{total}] {rate:.1f} trials/s, ETA {eta:.0f}m")

    save_results(results, "probe_differential")
    _summarize_probe(results)
    return results


def run_same_task(model, layer, direction, coefficients, n_resamples, n_tasks=N_SAME_TASK):
    """Same task as both A and B — pure position artifact test."""
    print(f"\n{'='*60}")
    print(f"SAME-TASK CONTROL")
    print(f"  {n_tasks} tasks x {len(coefficients)} coefs x {n_resamples} resamples")
    print(f"{'='*60}")

    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)

    # Sample diverse tasks
    rng = random.Random(555)
    by_origin: dict[str, list[dict]] = {}
    for t in all_tasks:
        origin = t["origin"]
        if origin not in by_origin:
            by_origin[origin] = []
        by_origin[origin].append(t)

    tasks = []
    for origin in sorted(by_origin.keys()):
        sampled = rng.sample(by_origin[origin], min(6, len(by_origin[origin])))
        tasks.extend(sampled)
    rng.shuffle(tasks)
    tasks = tasks[:n_tasks]
    print(f"  Sampled {len(tasks)} tasks")

    results = []
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

        for coef in coefficients:
            for seed in range(n_resamples):
                response, choice = run_trial(
                    model, messages, layer, direction, coef, a_span, b_span, seed,
                )
                results.append({
                    "condition": "same_task",
                    "task_idx": task_idx,
                    "task_id": task["task_id"],
                    "task_origin": task["origin"],
                    "coefficient": coef,
                    "seed": seed,
                    "response": response,
                    "choice": choice,
                })
                done += 1

    save_results(results, "same_task")
    _summarize_condition(results, "same_task")
    return results


def run_header_only(model, layer, direction, pairs, coefficients, n_resamples):
    """Header-only steering — steer only "Task A:\n" / "Task B:\n" tokens."""
    print(f"\n{'='*60}")
    print(f"HEADER-ONLY STEERING")
    print(f"  {len(pairs)} pairs x {len(coefficients)} coefs x {n_resamples} resamples x 2 orderings")
    print(f"{'='*60}")

    results = []
    done = 0

    for pair in pairs:
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
                a_header_span, b_header_span = find_header_spans(model, messages)
            except (ValueError, TypeError) as e:
                print(f"  Skipping pair {pair['pair_idx']} ({ordering}): {e}")
                continue

            for coef in coefficients:
                for seed in range(n_resamples):
                    response, choice = run_trial(
                        model, messages, layer, direction, coef,
                        a_header_span, b_header_span, seed,
                    )
                    results.append({
                        "condition": "header_only",
                        "pair_idx": pair["pair_idx"],
                        "ordering": ordering,
                        "task_a_id": pair["task_a_id"],
                        "task_b_id": pair["task_b_id"],
                        "delta_mu": pair["delta_mu"],
                        "delta_mu_bin": pair["delta_mu_bin"],
                        "coefficient": coef,
                        "seed": seed,
                        "response": response,
                        "choice": choice,
                    })
                    done += 1

    save_results(results, "header_only")
    _summarize_probe(results)
    return results


def run_random_directions(model, layer, direction, pairs, coefficients, n_resamples):
    """Random orthogonal directions on borderline pairs."""
    print(f"\n{'='*60}")
    print(f"RANDOM DIRECTION CONTROL")
    print(f"  {len(pairs)} pairs x {N_RANDOM_DIRS} dirs x {len(coefficients)} coefs x {n_resamples} resamples")
    print(f"{'='*60}")

    random_dirs = generate_random_orthogonal_directions(direction, N_RANDOM_DIRS)

    results = []
    done = 0
    total = len(pairs) * N_RANDOM_DIRS * len(coefficients) * n_resamples
    t0 = time.time()

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
            for coef in coefficients:
                for seed in range(n_resamples):
                    torch.manual_seed(seed)
                    scaled = torch.tensor(rand_dir * coef, dtype=torch.bfloat16, device="cuda")
                    hook = differential_steering(scaled, a_span[0], a_span[1], b_span[0], b_span[1])
                    response = model.generate_with_steering(
                        messages=messages, layer=layer, steering_hook=hook,
                        temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
                    )
                    choice = parse_choice(response)

                    results.append({
                        "condition": "random_direction",
                        "pair_idx": pair["pair_idx"],
                        "dir_idx": dir_idx,
                        "delta_mu": pair["delta_mu"],
                        "delta_mu_bin": pair["delta_mu_bin"],
                        "coefficient": coef,
                        "seed": seed,
                        "response": response,
                        "choice": choice,
                    })

                    done += 1
                    if done % 1000 == 0:
                        elapsed = time.time() - t0
                        rate = done / elapsed
                        eta = (total - done) / rate / 60
                        print(f"  [{done}/{total}] {rate:.1f} trials/s, ETA {eta:.0f}m")

    save_results(results, "random_directions")
    return results


def _summarize_probe(results):
    for ordering in ["original", "swapped"]:
        print(f"\n  {ordering.upper()} ordering:")
        for coef in sorted(set(r["coefficient"] for r in results)):
            matching = [r for r in results
                       if r["ordering"] == ordering and r["coefficient"] == coef and r["choice"] is not None]
            if matching:
                n_a = sum(1 for r in matching if r["choice"] == "a")
                p_a = n_a / len(matching)
                print(f"    coef={coef:+7.0f}: P(A)={p_a:.3f} ({n_a}/{len(matching)})")


def _summarize_condition(results, condition):
    print(f"\n  {condition.upper()}:")
    for coef in sorted(set(r["coefficient"] for r in results)):
        matching = [r for r in results
                   if r["coefficient"] == coef and r["choice"] is not None]
        if matching:
            n_a = sum(1 for r in matching if r["choice"] == "a")
            p_a = n_a / len(matching)
            print(f"    coef={coef:+7.0f}: P(A)={p_a:.3f} ({n_a}/{len(matching)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run small pilot to validate pipeline")
    parser.add_argument("--condition", choices=["probe", "same_task", "header", "random", "all"],
                       default="all")
    parser.add_argument("--subset", choices=["borderline", "all"], default="all",
                       help="Which pairs to run on")
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    print(f"Loaded probe direction from layer {layer}, dim={len(direction)}")

    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded: {model.model_name}")

    if args.pilot:
        coefficients = [-3000.0, 0.0, 3000.0]
        n_resamples = 5
        pairs = load_pairs(max_pairs=5)
        borderline_pairs = [p for p in pairs if p["delta_mu_bin"] == "0-1"]
        if not borderline_pairs:
            borderline_pairs = pairs[:3]
        print(f"\nPILOT MODE: {len(pairs)} pairs, {len(coefficients)} coefs, {n_resamples} resamples")
    else:
        coefficients = COEFFICIENTS
        n_resamples = N_RESAMPLES
        pairs = load_pairs(subset=args.subset if args.subset != "all" else None,
                          max_pairs=args.max_pairs)
        borderline_pairs = [p for p in pairs if p["delta_mu_bin"] == "0-1"]
        print(f"\nFULL MODE: {len(pairs)} pairs, {len(borderline_pairs)} borderline")

    if args.condition in ("probe", "all"):
        run_probe_differential(model, layer, direction, pairs, coefficients, n_resamples)

    if args.condition in ("same_task", "all"):
        n_tasks = 10 if args.pilot else N_SAME_TASK
        run_same_task(model, layer, direction, coefficients, n_resamples, n_tasks=n_tasks)

    if args.condition in ("header", "all"):
        header_pairs = borderline_pairs if not args.pilot else borderline_pairs[:3]
        run_header_only(model, layer, direction, header_pairs, coefficients, n_resamples)

    if args.condition in ("random", "all"):
        random_pairs = borderline_pairs if not args.pilot else borderline_pairs[:3]
        run_random_directions(model, layer, direction, random_pairs, coefficients, n_resamples)

    print("\n" + "="*60)
    print("ALL CONDITIONS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
