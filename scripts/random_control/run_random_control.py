"""
Random direction control experiment.

Tests whether steering effects from the replication (Phase 1) are specific to the
probe direction, or whether any unit vector in activation space shifts pairwise
choices equally.

Design (per random direction):
  - 77 borderline pairs from replication Phase 1
  - Conditions: boost_a, diff_ab (no control — reuse replication data)
  - Coefficients: -2641, +2641
  - Resamples: 10
  - Seeds: 100, 101, 102 (extend to 104 if needed)

Probe re-run (within-experiment comparison):
  - Same pairs, same conditions + control at coef=0
  - Coefficients: -2641, 0, +2641
  - Resamples: 10

Usage:
  python scripts/random_control/run_random_control.py --mode pilot
  python scripts/random_control/run_random_control.py --mode probe
  python scripts/random_control/run_random_control.py --mode random --seed 100
  python scripts/random_control/run_random_control.py --mode random --seed 101
  python scripts/random_control/run_random_control.py --mode random --seed 102
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPLICATION_DIR = REPO_ROOT / "experiments" / "steering" / "replication"
REPLICATION_RESULTS = REPLICATION_DIR / "results"
EXP_DIR = REPLICATION_DIR / "random_control"
RESULTS_DIR = EXP_DIR / "results"
ASSETS_DIR = EXP_DIR / "assets"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PROBE_MANIFEST_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_raw" / "gemma3_10k_heldout_std_raw"

PROMPT_TEMPLATE = (
    "You will be given two tasks. Choose which task you would prefer to complete.\n\n"
    "Respond with only 'a' or 'b'.\n\n"
    "Task A:\n{task_a}\n\n"
    "Task B:\n{task_b}"
)

LAYER = 31
ACTIVATION_DIM = 5376
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 8
N_RESAMPLES = 10
PROBE_COEFFICIENTS = [-2641.0, 0.0, 2641.0]
RANDOM_COEFFICIENTS = [-2641.0, 2641.0]  # skip control for random dirs
RANDOM_SEEDS = [100, 101, 102, 103, 104]
CONDITIONS = ["boost_a", "diff_ab"]  # skip redundant conditions


def parse_response(response: str) -> str:
    r = response.strip().lower()
    if r.startswith("a"):
        return "a"
    elif r.startswith("b"):
        return "b"
    return "parse_fail"


def get_token_spans(tokenizer, task_a: str, task_b: str):
    from src.steering.tokenization import find_pairwise_task_spans
    prompt = PROMPT_TEMPLATE.format(task_a=task_a, task_b=task_b)
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True
    )
    a_span, b_span = find_pairwise_task_spans(
        tokenizer, formatted, task_a, task_b,
        a_marker="Task A:", b_marker="Task B:"
    )
    return a_span, b_span


def make_position_hook(direction: np.ndarray, coefficient: float, span: tuple[int, int], device, dtype):
    import torch
    from src.models.base import position_selective_steering, noop_steering
    if coefficient == 0.0:
        return noop_steering()
    tensor = torch.tensor(direction * coefficient, dtype=dtype, device=device)
    return position_selective_steering(tensor, span[0], span[1])


def make_differential_hook(direction: np.ndarray, coefficient: float,
                            pos_span: tuple[int, int], neg_span: tuple[int, int], device, dtype):
    import torch
    from src.models.base import differential_steering, noop_steering
    if coefficient == 0.0:
        return noop_steering()
    tensor = torch.tensor(direction * coefficient, dtype=dtype, device=device)
    return differential_steering(tensor, pos_span[0], pos_span[1], neg_span[0], neg_span[1])


def generate_random_direction(seed: int, dim: int = ACTIVATION_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(dim)
    direction = direction / np.linalg.norm(direction)
    return direction


def run_steering_reduced(
    model,
    tokenizer,
    pairs: list[dict],
    direction: np.ndarray,
    layer: int,
    coefficients: list[float],
    n_resamples: int,
    include_control: bool,
    label: str = "",
    device: str = "cuda",
) -> list[dict]:
    """
    Run reduced steering experiment: boost_a and diff_ab only.
    If include_control, also runs coef=0 control (model.generate without hook).
    """
    import torch
    from src.types import Message

    dtype = torch.bfloat16
    results = []

    n_conditions = len(conditions_for(coefficients, include_control))
    total_estimate = len(pairs) * 2 * n_conditions * n_resamples
    done = 0
    t0 = time.time()

    for pair in pairs:
        for ordering in ["original", "swapped"]:
            if ordering == "original":
                task_a, task_b = pair["task_a_text"], pair["task_b_text"]
            else:
                task_a, task_b = pair["task_b_text"], pair["task_a_text"]

            try:
                a_span, b_span = get_token_spans(tokenizer, task_a, task_b)
            except ValueError as e:
                print(f"  WARNING: Token span error for {pair['pair_id']} {ordering}: {e}")
                continue

            prompt = PROMPT_TEMPLATE.format(task_a=task_a, task_b=task_b)
            messages: list[Message] = [{"role": "user", "content": prompt}]

            # Control (coef=0) if requested
            if include_control:
                responses = []
                for _ in range(n_resamples):
                    resp = model.generate(messages, temperature=TEMPERATURE)
                    responses.append(parse_response(resp))
                    done += 1
                results.append({
                    "pair_id": pair["pair_id"],
                    "ordering": ordering,
                    "condition": "control",
                    "coefficient": 0.0,
                    "responses": responses,
                })

            # Steering conditions
            for coef in coefficients:
                if coef == 0.0:
                    continue

                for condition in CONDITIONS:
                    if condition == "boost_a":
                        hook = make_position_hook(direction, coef, a_span, device, dtype)
                    elif condition == "diff_ab":
                        hook = make_differential_hook(direction, coef, a_span, b_span, device, dtype)
                    else:
                        raise ValueError(f"Unknown condition: {condition}")

                    responses = []
                    for _ in range(n_resamples):
                        resp = model.generate_with_steering(
                            messages=messages,
                            layer=layer,
                            steering_hook=hook,
                            temperature=TEMPERATURE,
                        )
                        responses.append(parse_response(resp))
                        done += 1

                    results.append({
                        "pair_id": pair["pair_id"],
                        "ordering": ordering,
                        "condition": condition,
                        "coefficient": coef,
                        "responses": responses,
                    })

            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_estimate - done) / rate if rate > 0 else 0
            print(f"  {label} {done}/{total_estimate} ({rate:.1f}/s, ETA {eta/60:.0f}min) | {pair['pair_id']} {ordering}")

    return results


def conditions_for(coefficients: list[float], include_control: bool) -> list:
    """Count the total condition×coef combinations."""
    n = 0
    if include_control:
        n += 1  # control at coef=0
    n += len([c for c in coefficients if c != 0.0]) * len(CONDITIONS)
    return list(range(n))


def load_borderline_pairs() -> list[dict]:
    """Load pairs.json and filter to borderline pairs from screening."""
    pairs_path = REPLICATION_RESULTS / "pairs.json"
    screening_path = REPLICATION_RESULTS / "screening.json"

    with open(pairs_path) as f:
        all_pairs = json.load(f)
    with open(screening_path) as f:
        screening = json.load(f)

    borderline_ids = set(screening["borderline_pair_ids"])
    borderline = [p for p in all_pairs if p["pair_id"] in borderline_ids]
    print(f"Loaded {len(borderline)} borderline pairs (of {len(all_pairs)} total)")
    return borderline


def load_model():
    from src.models.huggingface_model import HuggingFaceModel
    print("Loading model: gemma-3-27b...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded. Layers: {model.n_layers}, hidden_dim: {model.hidden_dim}")
    return model


def run_pilot(model, tokenizer, pairs: list[dict]):
    """Quick sanity check: 5 pairs, probe direction, 3 resamples."""
    from src.probes.core.storage import load_probe_direction

    print("=== PILOT: 5 pairs, probe direction, 3 resamples ===")
    layer, probe_dir = load_probe_direction(PROBE_MANIFEST_DIR, "ridge_L31")
    pilot_pairs = pairs[:5]

    results = run_steering_reduced(
        model=model,
        tokenizer=tokenizer,
        pairs=pilot_pairs,
        direction=probe_dir,
        layer=layer,
        coefficients=[-2641.0, 2641.0],
        n_resamples=3,
        include_control=True,
        label="Pilot",
    )

    # Quick summary
    total = sum(len(r["responses"]) for r in results)
    parsed = sum(sum(1 for x in r["responses"] if x != "parse_fail") for r in results)
    print(f"\nPilot done. Parse rate: {parsed}/{total} = {parsed/total:.1%}")

    out_path = RESULTS_DIR / "pilot.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "parse_rate": parsed / total if total > 0 else 0}, f, indent=2)
    print(f"Saved pilot to {out_path}")
    return results


def run_probe(model, tokenizer, pairs: list[dict]):
    """Re-run probe direction with reduced design for within-experiment comparison."""
    from src.probes.core.storage import load_probe_direction

    print(f"=== PROBE RE-RUN: {len(pairs)} pairs, probe direction ===")
    layer, probe_dir = load_probe_direction(PROBE_MANIFEST_DIR, "ridge_L31")

    results = run_steering_reduced(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        direction=probe_dir,
        layer=layer,
        coefficients=PROBE_COEFFICIENTS,
        n_resamples=N_RESAMPLES,
        include_control=True,
        label="Probe",
    )

    out_path = RESULTS_DIR / "probe_rerun.json"
    with open(out_path, "w") as f:
        json.dump({
            "direction_type": "probe",
            "probe_id": "ridge_L31",
            "layer": layer,
            "coefficients": PROBE_COEFFICIENTS,
            "n_resamples": N_RESAMPLES,
            "results": results,
        }, f, indent=2)
    print(f"Saved probe re-run to {out_path}")
    return results


def run_random(model, tokenizer, pairs: list[dict], seed: int):
    """Run one random direction."""
    print(f"=== RANDOM DIRECTION: seed={seed} ===")
    direction = generate_random_direction(seed, dim=ACTIVATION_DIM)
    print(f"  Direction norm check: {np.linalg.norm(direction):.6f} (should be 1.0)")

    results = run_steering_reduced(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        direction=direction,
        layer=LAYER,
        coefficients=RANDOM_COEFFICIENTS,
        n_resamples=N_RESAMPLES,
        include_control=False,
        label=f"Random(seed={seed})",
    )

    out_path = RESULTS_DIR / f"random_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump({
            "direction_type": "random",
            "seed": seed,
            "layer": LAYER,
            "activation_dim": ACTIVATION_DIM,
            "coefficients": RANDOM_COEFFICIENTS,
            "n_resamples": N_RESAMPLES,
            "results": results,
        }, f, indent=2)
    print(f"Saved random seed={seed} to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["pilot", "probe", "random"],
                        help="pilot=sanity check; probe=probe re-run; random=random direction")
    parser.add_argument("--seed", type=int, help="Random seed (required for --mode random)")
    parser.add_argument("--n-pairs", type=int, default=None,
                        help="Number of pairs to run (for testing; None = all borderline pairs)")
    args = parser.parse_args()

    if args.mode == "random" and args.seed is None:
        parser.error("--seed is required for --mode random")

    pairs = load_borderline_pairs()
    if args.n_pairs is not None:
        pairs = pairs[:args.n_pairs]
        print(f"Using first {len(pairs)} pairs (test mode)")

    model = load_model()
    tokenizer = model.tokenizer

    if args.mode == "pilot":
        run_pilot(model, tokenizer, pairs)
    elif args.mode == "probe":
        run_probe(model, tokenizer, pairs)
    elif args.mode == "random":
        run_random(model, tokenizer, pairs, seed=args.seed)


if __name__ == "__main__":
    main()
