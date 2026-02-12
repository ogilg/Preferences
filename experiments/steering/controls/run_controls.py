"""Negative results and controls for steering experiments.

Tests designed to find boundaries and failures:
1. Wrong layer: Steer at L15 and L49 instead of L31
2. Equal steering: +A, +B simultaneously (should NOT shift preference)
3. Random directions: 10 random orthogonal directions on revealed preferences
4. Reversed polarity: Does negative steering on A + positive on B reverse H2?
"""

import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import SteeringHook, all_tokens_steering
from src.probes.core.storage import load_probe_direction
from src.types import Message


PROBE_MANIFEST_DIR = Path("results/probes/gemma3_3k_completion_preference")
PROBE_ID = "ridge_L31"
MODEL_NAME = "gemma-3-27b"
OUTPUT_DIR = Path("experiments/steering/controls")
TASKS_FILE = Path("activations/gemma_3_27b/completions_with_activations.json")

PAIRWISE_TEMPLATE = """You will be given two tasks. Choose which task you would prefer to complete.

Respond with only 'a' or 'b'.

Task A:
{task_a}

Task B:
{task_b}"""

N_PAIRS = 30
N_RESAMPLES = 10
COEFFICIENTS = [-3000.0, 0.0, 3000.0]
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


def differential_steering(steering_tensor, a_start, a_end, b_start, b_end) -> SteeringHook:
    """Standard: +A, -B."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:
            resid[:, a_start:a_end, :] += steering_tensor
            resid[:, b_start:b_end, :] -= steering_tensor
        return resid
    return hook


def equal_steering(steering_tensor, a_start, a_end, b_start, b_end) -> SteeringHook:
    """Control: +A, +B (both positive — should NOT shift preference)."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:
            resid[:, a_start:a_end, :] += steering_tensor
            resid[:, b_start:b_end, :] += steering_tensor
        return resid
    return hook


def sample_task_pairs(n_pairs, seed=999):
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


def generate_random_orthogonal_directions(direction, n_dirs=10, seed=42):
    """Generate random directions orthogonal to probe direction."""
    rng = np.random.RandomState(seed)
    directions = []
    for _ in range(n_dirs):
        v = rng.randn(len(direction))
        # Orthogonalize against probe direction
        v = v - np.dot(v, direction) * direction
        # Normalize to same norm as probe direction
        v = v / np.linalg.norm(v)
        directions.append(v)
    return directions


def run_experiment(model, pairs, layer, direction_vec, coefs, hook_factory, label):
    """Run an experiment with given hook factory and return results."""
    results = []
    total = len(pairs) * len(coefs) * N_RESAMPLES
    done = 0

    for pair_idx, (task_a, task_b) in enumerate(pairs):
        prompt_text = PAIRWISE_TEMPLATE.format(
            task_a=task_a["task_prompt"], task_b=task_b["task_prompt"],
        )
        messages: list[Message] = [{"role": "user", "content": prompt_text}]

        try:
            a_span, b_span = find_task_spans(model, messages, task_a["task_prompt"], task_b["task_prompt"])
        except (ValueError, TypeError):
            continue

        for coef in coefs:
            for seed in range(N_RESAMPLES):
                torch.manual_seed(seed)
                scaled = torch.tensor(direction_vec * coef, dtype=torch.bfloat16, device="cuda")
                hook = hook_factory(scaled, a_span[0], a_span[1], b_span[0], b_span[1])

                response = model.generate_with_steering(
                    messages=messages, layer=layer, steering_hook=hook,
                    temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
                )
                choice = parse_choice(response)

                results.append({
                    "experiment": label,
                    "pair_idx": pair_idx,
                    "task_a_id": task_a["task_id"],
                    "task_b_id": task_b["task_id"],
                    "coefficient": coef,
                    "seed": seed,
                    "response": response,
                    "choice": choice,
                })

                done += 1
                if done % 50 == 0:
                    print(f"  [{label}] {done}/{total}")

    return results


def run_wrong_layer(model, pairs, direction):
    """Test steering at wrong layers (L15, L49)."""
    results = []
    # We need probes at different layers - load from the manifest
    # The probe direction is from L31, but we can apply it at other layers
    # This is intentionally "wrong" - the direction is L31-specific

    for wrong_layer in [15, 49]:
        print(f"\n  Testing wrong layer: L{wrong_layer}")
        layer_results = run_experiment(
            model, pairs, wrong_layer, direction,
            COEFFICIENTS, differential_steering, f"wrong_layer_L{wrong_layer}"
        )
        results.extend(layer_results)

    return results


def summarize(results, label):
    """Print summary for an experiment."""
    valid = [r for r in results if r["choice"] is not None and r["experiment"] == label]
    if not valid:
        print(f"  {label}: no valid results")
        return

    coefficients = sorted(set(r["coefficient"] for r in valid))
    print(f"\n  {label}:")
    for coef in coefficients:
        matching = [r for r in valid if r["coefficient"] == coef]
        n_a = sum(1 for r in matching if r["choice"] == "a")
        p_a = n_a / len(matching)
        print(f"    coef={coef:+7.0f}: P(A)={p_a:.3f} ({n_a}/{len(matching)})")

    # Effect size: P(A) at max - P(A) at min
    if len(coefficients) >= 2:
        min_c = [r for r in valid if r["coefficient"] == min(coefficients)]
        max_c = [r for r in valid if r["coefficient"] == max(coefficients)]
        p_min = sum(1 for r in min_c if r["choice"] == "a") / len(min_c)
        p_max = sum(1 for r in max_c if r["choice"] == "a") / len(max_c)
        print(f"    ΔP(A) = {p_max - p_min:+.3f}")


def main():
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded: {model.model_name}")

    pairs = sample_task_pairs(N_PAIRS)
    print(f"Sampled {len(pairs)} pairs")

    all_results = []

    # 1. Differential steering (probe direction) — baseline for comparison
    print("\n=== Control 0: Differential steering (probe direction, baseline) ===")
    baseline = run_experiment(model, pairs, layer, direction, COEFFICIENTS, differential_steering, "baseline_differential")
    all_results.extend(baseline)
    summarize(all_results, "baseline_differential")

    # 2. Equal steering (+A, +B) — should NOT shift preference
    print("\n=== Control 1: Equal steering (+A, +B) ===")
    equal = run_experiment(model, pairs, layer, direction, COEFFICIENTS, equal_steering, "equal_both_positive")
    all_results.extend(equal)
    summarize(all_results, "equal_both_positive")

    # 3. Wrong layer steering
    print("\n=== Control 2: Wrong layer steering ===")
    wrong_layer = run_wrong_layer(model, pairs, direction)
    all_results.extend(wrong_layer)
    summarize(all_results, "wrong_layer_L15")
    summarize(all_results, "wrong_layer_L49")

    # 4. Random orthogonal directions
    print("\n=== Control 3: Random orthogonal directions ===")
    random_dirs = generate_random_orthogonal_directions(direction, n_dirs=10)
    for dir_idx, rand_dir in enumerate(random_dirs):
        rand_results = run_experiment(
            model, pairs, layer, rand_dir,
            COEFFICIENTS, differential_steering, f"random_dir_{dir_idx}"
        )
        all_results.extend(rand_results)
        summarize(all_results, f"random_dir_{dir_idx}")

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY: ΔP(A) across all conditions")
    print(f"{'='*60}")

    experiments = sorted(set(r["experiment"] for r in all_results))
    for exp in experiments:
        valid = [r for r in all_results if r["experiment"] == exp and r["choice"] is not None]
        if not valid:
            continue
        min_coef = min(r["coefficient"] for r in valid)
        max_coef = max(r["coefficient"] for r in valid)
        min_c = [r for r in valid if r["coefficient"] == min_coef]
        max_c = [r for r in valid if r["coefficient"] == max_coef]
        p_min = sum(1 for r in min_c if r["choice"] == "a") / len(min_c) if min_c else 0
        p_max = sum(1 for r in max_c if r["choice"] == "a") / len(max_c) if max_c else 0
        delta = p_max - p_min
        print(f"  {exp:>30}: ΔP(A) = {delta:+.3f}  (n_min={len(min_c)}, n_max={len(max_c)})")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "control_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
