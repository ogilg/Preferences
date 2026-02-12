"""Phase 3: Revealed preference steering.

Test whether steering causally shifts which task the model chooses in pairwise settings.

Hypotheses tested:
- H3: Last-token steering (autoregressive) during choice generation
- H1: Task-selective steering (only steer on one task's token positions)
"""

import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import all_tokens_steering, autoregressive_steering, SteeringHook
from src.probes.core.storage import load_probe_direction
from src.types import Message


PROBE_MANIFEST_DIR = Path("results/probes/gemma3_3k_completion_preference")
PROBE_ID = "ridge_L31"
MODEL_NAME = "gemma-3-27b"
OUTPUT_DIR = Path("experiments/steering/revealed_preference")

TASKS_FILE = Path("activations/gemma_3_27b/completions_with_activations.json")

# Pairwise prompt template
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
    """Parse choice response. Returns 'a', 'b', or None."""
    cleaned = response.strip().lower()
    if cleaned in ("a", "b"):
        return cleaned
    # Check for "task a" / "task b"
    if re.search(r'\btask\s*a\b', cleaned):
        return "a"
    if re.search(r'\btask\s*b\b', cleaned):
        return "b"
    # First character
    if cleaned and cleaned[0] in ("a", "b"):
        return cleaned[0]
    return None


def sample_task_pairs(n_pairs: int, seed: int = 42) -> list[tuple[dict, dict]]:
    """Sample diverse task pairs from different origins."""
    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)

    rng = random.Random(seed)
    # Group by origin
    by_origin: dict[str, list[dict]] = {}
    for t in all_tasks:
        origin = t["origin"]
        if origin not in by_origin:
            by_origin[origin] = []
        by_origin[origin].append(t)

    origins = sorted(by_origin.keys())
    pairs = []

    # Cross-origin pairs for diversity
    for _ in range(n_pairs):
        o1, o2 = rng.sample(origins, 2)
        t1 = rng.choice(by_origin[o1])
        t2 = rng.choice(by_origin[o2])
        pairs.append((t1, t2))

    return pairs


def position_selective_steering(
    steering_tensor: torch.Tensor,
    target_start: int,
    target_end: int,
) -> SteeringHook:
    """Steer only at specific token positions (for task-selective steering)."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        # During prompt processing: resid is [batch, seq, hidden]
        # During generation: resid is [batch, 1, hidden] (KV cache)
        if resid.shape[1] > 1:
            # Prompt processing — steer only target positions
            resid[:, target_start:target_end, :] += steering_tensor
        # During generation: don't steer (we only want to affect task encoding)
        return resid
    return hook


def find_task_spans(
    model: HuggingFaceModel,
    messages: list[Message],
    task_a_text: str,
    task_b_text: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find token spans for task A and task B in the formatted prompt."""
    formatted = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    # Find character positions
    a_marker = "Task A:\n"
    b_marker = "Task B:\n"
    a_start_char = formatted.index(a_marker) + len(a_marker)
    a_end_char = a_start_char + len(task_a_text)
    b_start_char = formatted.index(b_marker) + len(b_marker)
    b_end_char = b_start_char + len(task_b_text)

    # Tokenize full prompt
    full_tokens = model.tokenizer(formatted, return_tensors="pt")
    # Use offset mapping to find token positions
    encoded = model.tokenizer(formatted, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]

    def char_to_token_range(char_start: int, char_end: int) -> tuple[int, int]:
        tok_start = None
        tok_end = None
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if s <= char_start < e and tok_start is None:
                tok_start = i
            if s < char_end <= e:
                tok_end = i + 1
        if tok_start is None:
            # Fallback: find first token that starts at or after char_start
            for i, (s, e) in enumerate(offsets):
                if s >= char_start and tok_start is None:
                    tok_start = i
                    break
        if tok_end is None:
            tok_end = len(offsets)
        return (tok_start, tok_end)

    a_span = char_to_token_range(a_start_char, a_end_char)
    b_span = char_to_token_range(b_start_char, b_end_char)
    return a_span, b_span


def run_h3_autoregressive(
    model: HuggingFaceModel,
    layer: int,
    direction: np.ndarray,
    pairs: list[tuple[dict, dict]],
    coefficients: list[float],
    n_resamples: int,
    pilot: bool = False,
) -> list[dict]:
    """H3: Autoregressive steering (last-token only) during choice generation."""
    results = []
    total = len(pairs) * len(coefficients) * n_resamples
    done = 0

    for pair_idx, (task_a, task_b) in enumerate(pairs):
        prompt_text = PAIRWISE_TEMPLATE.format(
            task_a=task_a["task_prompt"],
            task_b=task_b["task_prompt"],
        )
        messages: list[Message] = [{"role": "user", "content": prompt_text}]

        for coef in coefficients:
            for seed in range(n_resamples):
                torch.manual_seed(seed)
                scaled = torch.tensor(direction * coef, dtype=torch.bfloat16, device="cuda")
                hook = autoregressive_steering(scaled)

                response = model.generate_with_steering(
                    messages=messages,
                    layer=layer,
                    steering_hook=hook,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                choice = parse_choice(response)

                results.append({
                    "hypothesis": "H3_autoregressive",
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
                    print(f"[H3] {done}/{total}")

    return results


def run_h1_task_selective(
    model: HuggingFaceModel,
    layer: int,
    direction: np.ndarray,
    pairs: list[tuple[dict, dict]],
    coefficients: list[float],
    n_resamples: int,
) -> list[dict]:
    """H1: Task-selective steering — only steer on one task's token positions."""
    results = []
    total = len(pairs) * len(coefficients) * n_resamples * 2  # x2 for steer-on-A vs steer-on-B
    done = 0

    for pair_idx, (task_a, task_b) in enumerate(pairs):
        prompt_text = PAIRWISE_TEMPLATE.format(
            task_a=task_a["task_prompt"],
            task_b=task_b["task_prompt"],
        )
        messages: list[Message] = [{"role": "user", "content": prompt_text}]

        # Find task spans
        try:
            a_span, b_span = find_task_spans(model, messages, task_a["task_prompt"], task_b["task_prompt"])
        except (ValueError, TypeError) as e:
            print(f"  Skipping pair {pair_idx}: span detection failed ({e})")
            continue

        for steer_target in ["task_a", "task_b"]:
            span = a_span if steer_target == "task_a" else b_span
            for coef in coefficients:
                for seed in range(n_resamples):
                    torch.manual_seed(seed)
                    scaled = torch.tensor(direction * coef, dtype=torch.bfloat16, device="cuda")
                    hook = position_selective_steering(scaled, span[0], span[1])

                    response = model.generate_with_steering(
                        messages=messages,
                        layer=layer,
                        steering_hook=hook,
                        temperature=TEMPERATURE,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    choice = parse_choice(response)

                    results.append({
                        "hypothesis": "H1_task_selective",
                        "pair_idx": pair_idx,
                        "steer_target": steer_target,
                        "task_a_id": task_a["task_id"],
                        "task_b_id": task_b["task_id"],
                        "task_a_origin": task_a["origin"],
                        "task_b_origin": task_b["origin"],
                        "coefficient": coef,
                        "seed": seed,
                        "response": response,
                        "choice": choice,
                        "steer_span": list(span),
                    })

                    done += 1
                    if done % 50 == 0 or done == total:
                        print(f"[H1] {done}/{total}")

    return results


def summarize_results(results: list[dict], hypothesis: str) -> None:
    """Print summary of choice rates by coefficient."""
    hyp_results = [r for r in results if r["hypothesis"] == hypothesis]
    if not hyp_results:
        return

    coefficients = sorted(set(r["coefficient"] for r in hyp_results))

    print(f"\n{'='*60}")
    print(f"{hypothesis} Summary")
    print(f"{'='*60}")

    if hypothesis == "H1_task_selective":
        for target in ["task_a", "task_b"]:
            print(f"\n  Steered on: {target}")
            for coef in coefficients:
                matching = [r for r in hyp_results
                           if r["coefficient"] == coef and r.get("steer_target") == target
                           and r["choice"] is not None]
                if matching:
                    n_a = sum(1 for r in matching if r["choice"] == "a")
                    p_a = n_a / len(matching)
                    print(f"    coef={coef:+7.0f}: P(A)={p_a:.3f} ({n_a}/{len(matching)})")
    else:
        for coef in coefficients:
            matching = [r for r in hyp_results
                       if r["coefficient"] == coef and r["choice"] is not None]
            if matching:
                n_a = sum(1 for r in matching if r["choice"] == "a")
                p_a = n_a / len(matching)
                print(f"  coef={coef:+7.0f}: P(A)={p_a:.3f} ({n_a}/{len(matching)})")


def main(pilot: bool = False, hypothesis: str = "h3"):
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    print(f"Loaded probe direction from layer {layer}")

    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded: {model.model_name}")

    n_pairs = 5 if pilot else N_PAIRS
    n_resamples = 3 if pilot else N_RESAMPLES
    coefficients = [-2000.0, 0.0, 2000.0] if pilot else COEFFICIENTS

    pairs = sample_task_pairs(n_pairs)
    print(f"Sampled {len(pairs)} task pairs")

    all_results = []

    if hypothesis in ("h3", "all"):
        print("\n--- Running H3: Autoregressive steering ---")
        h3_results = run_h3_autoregressive(
            model, layer, direction, pairs, coefficients, n_resamples, pilot=pilot
        )
        all_results.extend(h3_results)
        summarize_results(h3_results, "H3_autoregressive")

    if hypothesis in ("h1", "all"):
        print("\n--- Running H1: Task-selective steering ---")
        h1_results = run_h1_task_selective(
            model, layer, direction, pairs, coefficients, n_resamples
        )
        all_results.extend(h1_results)
        summarize_results(h1_results, "H1_task_selective")

    # Save
    output_path = OUTPUT_DIR / f"revealed_preference_{hypothesis}_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--hypothesis", choices=["h3", "h1", "all"], default="h3")
    args = parser.parse_args()
    main(pilot=args.pilot, hypothesis=args.hypothesis)
