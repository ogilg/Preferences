"""
Steering Replication & Extension experiment runner.

Phases:
  0 - Coefficient calibration + pilot
  1 - L31 replication (pair construction → screening → steering)
  2 - Utility-bin analysis + decisive pairs steering
  3 - Multi-layer steering

Usage:
  python scripts/replication/run_experiment.py --phase 0
  python scripts/replication/run_experiment.py --phase 1
  python scripts/replication/run_experiment.py --phase 2
  python scripts/replication/run_experiment.py --phase 3

Results saved under experiments/steering/replication/results/.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# ────────────────────────────────────────────────────────────────────────────────
# JSONL checkpointing
# ────────────────────────────────────────────────────────────────────────────────

def append_jsonl(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def completed_keys(path: Path) -> set[tuple[str, str]]:
    """Get set of (pair_id, ordering) already completed in a JSONL file."""
    records = load_jsonl(path)
    return {(r["pair_id"], r["ordering"]) for r in records}


def completed_condition_keys(path: Path) -> set[tuple[str, str, str, float]]:
    """Get set of (pair_id, ordering, condition, coefficient) already completed."""
    records = load_jsonl(path)
    return {(r["pair_id"], r["ordering"], r["condition"], r["coefficient"]) for r in records}


def block_condition_counts(path: Path) -> dict[tuple[str, str], int]:
    """Count completed conditions per (pair_id, ordering) block."""
    from collections import Counter
    records = load_jsonl(path)
    return Counter((r["pair_id"], r["ordering"]) for r in records)


# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "steering" / "replication"
RESULTS_DIR = EXP_DIR / "results"
ASSETS_DIR = EXP_DIR / "assets"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PROBE_MANIFEST_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_raw" / "gemma3_10k_heldout_std_raw"
THURSTONIAN_CSV = REPO_ROOT / "results" / "experiments" / "gemma3_10k_run1" / "pre_task_active_learning" / "completion_preference_gemma-3-27b_completion_canonical_seed0" / "thurstonian_80fa9dc8.csv"
ACTIVATIONS_PATH = REPO_ROOT / "activations" / "gemma_3_27b" / "activations_prompt_last.npz"

PROMPT_TEMPLATE = (
    "You will be given two tasks. Choose which task you would prefer to complete.\n\n"
    "Respond with only 'a' or 'b'.\n\n"
    "Task A:\n{task_a}\n\n"
    "Task B:\n{task_b}"
)

SEED = 42
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 8
SCREENING_RESAMPLES = 10
STEERING_RESAMPLES = 15
BIN_WIDTH = 2
PAIRS_PER_BIN = 30
PROBE_IDS = ["ridge_L15", "ridge_L31", "ridge_L37", "ridge_L43", "ridge_L49", "ridge_L55"]
PRIMARY_PROBE = "ridge_L31"


# ────────────────────────────────────────────────────────────────────────────────
# Task loading
# ────────────────────────────────────────────────────────────────────────────────

def load_task_texts() -> dict[str, str]:
    """Load all task prompts keyed by task_id."""
    from src.task_data.loader import load_tasks
    from src.task_data.task import OriginDataset

    origins = [
        OriginDataset.WILDCHAT,
        OriginDataset.ALPACA,
        OriginDataset.MATH,
        OriginDataset.BAILBENCH,
    ]
    tasks = load_tasks(n=100_000, origins=origins)
    return {t.id: t.prompt for t in tasks}


def load_thurstonian() -> pd.DataFrame:
    return pd.read_csv(THURSTONIAN_CSV)


# ────────────────────────────────────────────────────────────────────────────────
# Pair construction
# ────────────────────────────────────────────────────────────────────────────────

def construct_pairs(df: pd.DataFrame, task_texts: dict[str, str], seed: int = SEED) -> list[dict]:
    """
    Bin tasks by mu (width=2), sample PAIRS_PER_BIN pairs within each bin.
    Returns list of pair dicts with task texts and mu values.
    """
    rng = random.Random(seed)

    # Filter to tasks we have text for
    df = df[df["task_id"].isin(task_texts)].copy()

    mu_min = df["mu"].min()
    mu_max = df["mu"].max()

    # Create bins aligned to BIN_WIDTH boundaries
    bin_edges = list(range(int(np.floor(mu_min / BIN_WIDTH)) * BIN_WIDTH,
                           int(np.ceil(mu_max / BIN_WIDTH)) * BIN_WIDTH + BIN_WIDTH,
                           BIN_WIDTH))

    pairs = []
    pair_id = 0

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bin_df = df[(df["mu"] >= lo) & (df["mu"] < hi)]
        if len(bin_df) < 2:
            continue

        bin_tasks = list(bin_df.itertuples(index=False))
        rng.shuffle(bin_tasks)

        # Sample PAIRS_PER_BIN pairs (without replacement)
        n_pairs = min(PAIRS_PER_BIN, len(bin_tasks) // 2)
        used = set()
        bin_pairs = []
        for j in range(len(bin_tasks) - 1):
            if len(bin_pairs) >= n_pairs:
                break
            if j in used:
                continue
            # pair j with j+1 (or find next unused)
            for k in range(j + 1, len(bin_tasks)):
                if k not in used:
                    used.add(j)
                    used.add(k)
                    bin_pairs.append((bin_tasks[j], bin_tasks[k]))
                    break

        for task_a, task_b in bin_pairs:
            pairs.append({
                "pair_id": f"pair_{pair_id:04d}",
                "task_a_id": task_a.task_id,
                "task_b_id": task_b.task_id,
                "task_a_text": task_texts[task_a.task_id],
                "task_b_text": task_texts[task_b.task_id],
                "mu_a": float(task_a.mu),
                "mu_b": float(task_b.mu),
                "delta_mu": float(abs(task_a.mu - task_b.mu)),
                "bin": f"[{lo},{hi})",
            })
            pair_id += 1

    print(f"Constructed {len(pairs)} pairs across {len(bin_edges)-1} bins")
    return pairs


# ────────────────────────────────────────────────────────────────────────────────
# Response parsing
# ────────────────────────────────────────────────────────────────────────────────

def parse_response(response: str) -> str:
    """Parse 'a' or 'b' from model response. Returns 'a', 'b', or 'parse_fail'."""
    r = response.strip().lower()
    if r.startswith("a"):
        return "a"
    elif r.startswith("b"):
        return "b"
    return "parse_fail"


# ────────────────────────────────────────────────────────────────────────────────
# Screening
# ────────────────────────────────────────────────────────────────────────────────

def run_screening(model, pairs: list[dict], output_path: Path) -> dict:
    """
    Run screening: all pairs at coef=0, 2 orderings × SCREENING_RESAMPLES each.
    Identifies borderline pairs. Appends results to JSONL as each (pair, ordering) completes.
    """
    from src.types import Message

    already_done = completed_keys(output_path)
    total_blocks = len(pairs) * 2
    done_blocks = 0
    t0 = time.time()

    for pair in pairs:
        for ordering in ["original", "swapped"]:
            if (pair["pair_id"], ordering) in already_done:
                done_blocks += 1
                continue

            if ordering == "original":
                task_a, task_b = pair["task_a_text"], pair["task_b_text"]
                task_a_id, task_b_id = pair["task_a_id"], pair["task_b_id"]
            else:
                task_a, task_b = pair["task_b_text"], pair["task_a_text"]
                task_a_id, task_b_id = pair["task_b_id"], pair["task_a_id"]

            messages: list[Message] = [
                {"role": "user", "content": PROMPT_TEMPLATE.format(task_a=task_a, task_b=task_b)}
            ]
            raw = model.generate_n(messages, n=SCREENING_RESAMPLES, temperature=TEMPERATURE)
            responses = [parse_response(r) for r in raw]

            n_a = responses.count("a")
            n_valid = sum(1 for r in responses if r != "parse_fail")
            p_a = n_a / n_valid if n_valid > 0 else 0.5

            record = {
                "pair_id": pair["pair_id"],
                "ordering": ordering,
                "task_a_id": task_a_id,
                "task_b_id": task_b_id,
                "responses": responses,
                "p_a": p_a,
                "n_valid": n_valid,
                "borderline": (0 < n_a < n_valid) if n_valid > 0 else False,
            }
            append_jsonl(output_path, record)
            done_blocks += 1

            elapsed = time.time() - t0
            rate = done_blocks / elapsed if elapsed > 0 else 0
            print(f"  Screening {done_blocks}/{total_blocks} blocks ({rate:.1f} blocks/s) | pair={pair['pair_id']} ordering={ordering} p_a={p_a:.2f}")

    # Load all results (including previously checkpointed) and compute borderline stats
    results = load_jsonl(output_path)
    borderline_pair_ids = set()
    for r in results:
        if r["borderline"]:
            borderline_pair_ids.add(r["pair_id"])

    print(f"\nScreening complete: {len(borderline_pair_ids)}/{len(pairs)} borderline pairs")

    screening_data = {
        "n_pairs": len(pairs),
        "n_borderline": len(borderline_pair_ids),
        "borderline_pair_ids": list(borderline_pair_ids),
        "results": results,
    }
    return screening_data


# ────────────────────────────────────────────────────────────────────────────────
# Steering (Phase 1 + Phase 2)
# ────────────────────────────────────────────────────────────────────────────────

def get_token_spans(tokenizer, task_a: str, task_b: str):
    """Get token spans for task A and B in the formatted prompt."""
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
    """Build a position-selective hook for the given span and coefficient."""
    import torch
    from src.models.base import position_selective_steering, noop_steering
    if coefficient == 0.0:
        return noop_steering()
    tensor = torch.tensor(direction * coefficient, dtype=dtype, device=device)
    return position_selective_steering(tensor, span[0], span[1])


def make_differential_hook(direction: np.ndarray, coefficient: float,
                            pos_span: tuple[int, int], neg_span: tuple[int, int], device, dtype):
    """Build differential steering hook (+coef on pos_span, -coef on neg_span)."""
    import torch
    from src.models.base import differential_steering, noop_steering
    if coefficient == 0.0:
        return noop_steering()
    tensor = torch.tensor(direction * coefficient, dtype=dtype, device=device)
    return differential_steering(tensor, pos_span[0], pos_span[1], neg_span[0], neg_span[1])


ALL_STEERING_CONDITIONS = ["boost_a", "boost_b", "suppress_a", "suppress_b", "diff_ab", "diff_ba"]


def _make_hook_for_condition(
    condition: str, direction: np.ndarray, coef: float,
    a_span: tuple[int, int], b_span: tuple[int, int], device, dtype,
):
    """Build the appropriate steering hook for a given condition name."""
    if condition == "boost_a":
        return make_position_hook(direction, coef, a_span, device, dtype)
    elif condition == "boost_b":
        return make_position_hook(direction, coef, b_span, device, dtype)
    elif condition == "suppress_a":
        return make_position_hook(direction, -coef, a_span, device, dtype)
    elif condition == "suppress_b":
        return make_position_hook(direction, -coef, b_span, device, dtype)
    elif condition == "diff_ab":
        return make_differential_hook(direction, coef, a_span, b_span, device, dtype)
    elif condition == "diff_ba":
        return make_differential_hook(direction, coef, b_span, a_span, device, dtype)
    raise ValueError(f"Unknown condition: {condition}")


def run_steering_batch(
    model,
    tokenizer,
    pairs: list[dict],
    direction: np.ndarray,
    layer: int,
    coefficients: list[float],
    n_resamples: int,
    output_path: Path,
    conditions: list[str] = ALL_STEERING_CONDITIONS,
    extra_fields_fn: callable | None = None,
    label: str = "",
    device: str = "cuda",
) -> list[dict]:
    """
    Run steering experiment on a set of pairs. Appends results to JSONL per condition.

    Args:
        conditions: Which steering conditions to run (default: all 6).
        extra_fields_fn: If provided, called with each pair dict to produce
            extra fields merged into every JSONL record for that pair.
    """
    import torch
    from src.types import Message

    dtype = torch.bfloat16
    already_done = completed_condition_keys(output_path)
    counts = block_condition_counts(output_path)
    n_conditions_per_block = len(coefficients) * len(conditions) + 1  # steered + 1 control
    total_blocks = len(pairs) * 2
    done_blocks = 0
    t0 = time.time()

    for pair in pairs:
        extra = extra_fields_fn(pair) if extra_fields_fn else {}

        for ordering in ["original", "swapped"]:
            if counts.get((pair["pair_id"], ordering), 0) >= n_conditions_per_block:
                done_blocks += 1
                continue

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

            # Control condition (coef=0)
            if (pair["pair_id"], ordering, "control", 0.0) not in already_done:
                raw = model.generate_n(messages, n=n_resamples, temperature=TEMPERATURE)
                responses = [parse_response(r) for r in raw]
                record = {
                    "pair_id": pair["pair_id"],
                    "ordering": ordering,
                    "condition": "control",
                    "coefficient": 0.0,
                    "responses": responses,
                    **extra,
                }
                append_jsonl(output_path, record)

            # Steering conditions
            for coef in coefficients:
                for condition in conditions:
                    if (pair["pair_id"], ordering, condition, coef) in already_done:
                        continue

                    hook = _make_hook_for_condition(
                        condition, direction, coef, a_span, b_span, device, dtype,
                    )
                    raw = model.generate_with_steering_n(
                        messages=messages,
                        layer=layer,
                        steering_hook=hook,
                        n=n_resamples,
                        temperature=TEMPERATURE,
                    )
                    responses = [parse_response(r) for r in raw]
                    record = {
                        "pair_id": pair["pair_id"],
                        "ordering": ordering,
                        "condition": condition,
                        "coefficient": coef,
                        "responses": responses,
                        **extra,
                    }
                    append_jsonl(output_path, record)

            done_blocks += 1
            elapsed = time.time() - t0
            rate = done_blocks / elapsed if elapsed > 0 else 0
            eta = (total_blocks - done_blocks) / rate if rate > 0 else 0
            print(f"  {label} {done_blocks}/{total_blocks} blocks ({rate:.1f} blocks/s, ETA {eta/60:.0f}min) | {pair['pair_id']} {ordering}")

    return load_jsonl(output_path)


# ────────────────────────────────────────────────────────────────────────────────
# Phase 3: Multi-layer steering
# ────────────────────────────────────────────────────────────────────────────────

def run_multi_layer_steering(
    model,
    tokenizer,
    pairs: list[dict],
    directions: dict[str, np.ndarray],  # probe_id -> direction
    layers: dict[str, int],              # probe_id -> layer
    coefficients: list[float],
    n_resamples: int,
    output_path: Path,
    device: str = "cuda",
) -> list[dict]:
    """
    Phase 3: Multi-layer steering. Appends results to JSONL per condition.

    Conditions use boost_a (position-selective on task A tokens).
    Coefficient strategy: split (total budget / n_layers) + full at each layer.
    """
    import torch
    from src.models.base import position_selective_steering
    from src.types import Message

    dtype = torch.bfloat16
    already_done = completed_condition_keys(output_path)

    L31_dir = directions["ridge_L31"]
    L31_layer = layers["ridge_L31"]
    L37_dir = directions["ridge_L37"]
    L37_layer = layers["ridge_L37"]
    L43_dir = directions["ridge_L43"]
    L43_layer = layers["ridge_L43"]

    total_blocks = len(pairs) * 2
    done_blocks = 0
    t0 = time.time()

    for pair in pairs:
        for ordering in ["original", "swapped"]:
            if ordering == "original":
                task_a, task_b = pair["task_a_text"], pair["task_b_text"]
            else:
                task_a, task_b = pair["task_b_text"], pair["task_a_text"]

            try:
                a_span, _ = get_token_spans(tokenizer, task_a, task_b)
            except ValueError as e:
                print(f"  WARNING: Token span error for {pair['pair_id']} {ordering}: {e}")
                continue

            messages: list[Message] = [
                {"role": "user", "content": PROMPT_TEMPLATE.format(task_a=task_a, task_b=task_b)}
            ]

            for coef in coefficients:
                multi_conditions = [
                    ("L31_only", [(L31_layer, L31_dir, 1.0)]),
                    ("L31_L37_same_split", [(L31_layer, L31_dir, 0.5), (L37_layer, L31_dir, 0.5)]),
                    ("L31_L37_same_full", [(L31_layer, L31_dir, 1.0), (L37_layer, L31_dir, 1.0)]),
                    ("L31_L37_layer_split", [(L31_layer, L31_dir, 0.5), (L37_layer, L37_dir, 0.5)]),
                    ("L31_L43_layer_split", [(L31_layer, L31_dir, 0.5), (L43_layer, L43_dir, 0.5)]),
                    ("L31_L37_L43_layer_split", [(L31_layer, L31_dir, 1/3), (L37_layer, L37_dir, 1/3), (L43_layer, L43_dir, 1/3)]),
                ]

                for cond_name, layer_dir_factors in multi_conditions:
                    if (pair["pair_id"], ordering, cond_name, coef) in already_done:
                        continue

                    hook_list = []
                    for lyr, direction, factor in layer_dir_factors:
                        actual_coef = coef * factor
                        if actual_coef == 0.0:
                            continue
                        tensor = torch.tensor(direction * actual_coef, dtype=dtype, device=device)
                        hook = position_selective_steering(tensor, a_span[0], a_span[1])
                        hook_list.append((lyr, hook))

                    if not hook_list:
                        raw = model.generate_n(messages, n=n_resamples, temperature=TEMPERATURE)
                    elif len(hook_list) == 1:
                        raw = model.generate_with_steering_n(
                            messages=messages,
                            layer=hook_list[0][0],
                            steering_hook=hook_list[0][1],
                            n=n_resamples,
                            temperature=TEMPERATURE,
                        )
                    else:
                        raw = model.generate_with_multi_layer_steering_n(
                            messages=messages,
                            layer_hooks=hook_list,
                            n=n_resamples,
                            temperature=TEMPERATURE,
                        )
                    responses = [parse_response(r) for r in raw]

                    record = {
                        "pair_id": pair["pair_id"],
                        "ordering": ordering,
                        "condition": cond_name,
                        "coefficient": coef,
                        "responses": responses,
                    }
                    append_jsonl(output_path, record)

            done_blocks += 1
            elapsed = time.time() - t0
            rate = done_blocks / elapsed if elapsed > 0 else 0
            eta = (total_blocks - done_blocks) / rate if rate > 0 else 0
            print(f"  Phase3 {done_blocks}/{total_blocks} blocks ({rate:.1f} blocks/s, ETA {eta/60:.0f}min) | {pair['pair_id']} {ordering}")

    return load_jsonl(output_path)


# ────────────────────────────────────────────────────────────────────────────────
# Position-controlled P(steered)
# ────────────────────────────────────────────────────────────────────────────────

def compute_position_controlled_p_steered(results: list[dict]) -> dict[str, dict[float, list[float]]]:
    """
    For each condition (excluding control), compute P(steered task picked) per coefficient.

    For boost_a: steered task is A → pick if response=='a'
    For boost_b: steered task is B → pick if response=='b'
    For suppress_a: steered task is A (we suppressed it, so direction is negative, but track same task)
    For diff_ab: steered task is A
    For diff_ba: steered task is B

    In "swapped" ordering: task_a and task_b are physically swapped.
    We track the *original task A* (from pair construction) as the "steered" task.
    For boost_a in swapped ordering: we steered the task that is now in position A,
    which is original task B. But the condition tracks "task A in current ordering".

    The position-controlled approach: combine by looking at which task was steered
    (by position in prompt) rather than by which physical task.

    For boost_a at coef c: P(steered) = P(response=='a') in current ordering
    For boost_b at coef c: P(steered) = P(response=='b') in current ordering
    For suppress_a at coef c: equivalent to boost_a at -c → P(steered) = P(response=='a')
    For diff_ab: steered=A, P(steered) = P(response=='a')
    For diff_ba: steered=B, P(steered) = P(response=='b')

    Returns: {condition: {coefficient: [p_steered per trial]}}
    """
    from collections import defaultdict

    # Map condition to which response counts as "steered"
    condition_steered_choice = {
        "boost_a": "a",
        "boost_b": "b",
        "suppress_a": "a",
        "suppress_b": "b",
        "diff_ab": "a",
        "diff_ba": "b",
    }

    # For suppress conditions, coefficient is already negated in run_steering_batch
    # suppress_a uses -coef → so the coefficient stored is the *magnitude* coef
    # but the effective steering is negative. For the dose-response plot we want
    # suppress_a at coef c to appear at coefficient=-c.

    p_steered: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for trial in results:
        cond = trial["condition"]
        if cond == "control":
            continue
        coef = trial["coefficient"]
        steered_choice = condition_steered_choice.get(cond)
        if steered_choice is None:
            continue

        # For suppress conditions: negate the coefficient for the dose-response axis
        effective_coef = -coef if cond.startswith("suppress") else coef

        for resp in trial["responses"]:
            if resp == "parse_fail":
                continue
            p_val = 1.0 if resp == steered_choice else 0.0
            p_steered[cond][effective_coef].append(p_val)

    return dict(p_steered)


# ────────────────────────────────────────────────────────────────────────────────
# Phase 0: Calibration
# ────────────────────────────────────────────────────────────────────────────────

def phase0_calibration():
    """Load probe directions, compute calibrated coefficient ranges."""
    from src.steering.calibration import suggest_coefficient_range
    from src.probes.core.storage import load_probe_direction

    print("=== Phase 0: Calibration ===")

    calibration = {}
    for probe_id in PROBE_IDS:
        try:
            layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, probe_id)
            coefficients = suggest_coefficient_range(
                ACTIVATIONS_PATH, PROBE_MANIFEST_DIR, probe_id,
                multipliers=[-0.1, -0.05, 0.0, 0.05, 0.1]
            )
            calibration[probe_id] = {
                "layer": layer,
                "coefficients": coefficients,
            }
            print(f"  {probe_id} (L{layer}): coefficients = {[f'{c:.0f}' for c in coefficients]}")
        except Exception as e:
            print(f"  ERROR for {probe_id}: {e}")

    calib_path = RESULTS_DIR / "calibration.json"
    with open(calib_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"Saved calibration to {calib_path}")
    return calibration


def phase0_pilot(model, tokenizer, calibration: dict):
    """Run 20-pair pilot at calibrated coefficients to verify parse rates and dose-response."""
    from src.probes.core.storage import load_probe_direction

    print("=== Phase 0: Pilot ===")

    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PRIMARY_PROBE)
    coefs = calibration[PRIMARY_PROBE]["coefficients"]
    nonzero_coefs = [c for c in coefs if c != 0.0]

    print(f"Using {PRIMARY_PROBE} at layer {layer}, coefficients: {[f'{c:.0f}' for c in coefs]}")

    # Construct a small set of pairs for the pilot
    print("  Loading task data...")
    task_texts = load_task_texts()
    df = load_thurstonian()
    rng = random.Random(SEED)
    df_pilot = df[df["task_id"].isin(task_texts)].sample(n=min(100, len(df)), random_state=SEED)

    # Sample 20 diverse pairs
    df_pilot = df_pilot.sort_values("mu")
    step = max(1, len(df_pilot) // 20)
    task_list = list(df_pilot.itertuples(index=False))
    pairs = []
    for i in range(0, min(len(task_list) - 1, 20), 1):
        j = min(i + step, len(task_list) - 1)
        if i != j:
            pairs.append({
                "pair_id": f"pilot_{i:02d}",
                "task_a_id": task_list[i].task_id,
                "task_b_id": task_list[j].task_id,
                "task_a_text": task_texts[task_list[i].task_id],
                "task_b_text": task_texts[task_list[j].task_id],
                "mu_a": float(task_list[i].mu),
                "mu_b": float(task_list[j].mu),
                "delta_mu": float(abs(task_list[i].mu - task_list[j].mu)),
                "bin": "pilot",
            })
        if len(pairs) >= 20:
            break

    print(f"  Running pilot on {len(pairs)} pairs, 5 resamples per coef...")

    pilot_jsonl = RESULTS_DIR / "pilot_results.jsonl"
    pilot_results = run_steering_batch(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        direction=direction,
        layer=layer,
        coefficients=[c for c in nonzero_coefs],
        n_resamples=5,
        output_path=pilot_jsonl,
        label="Pilot",
    )

    # Check parse rates
    total = 0
    parsed = 0
    for trial in pilot_results:
        for resp in trial["responses"]:
            total += 1
            if resp != "parse_fail":
                parsed += 1

    parse_rate = parsed / total if total > 0 else 0
    print(f"  Pilot parse rate: {parse_rate:.1%} ({parsed}/{total})")

    pilot_path = RESULTS_DIR / "pilot_results.json"
    with open(pilot_path, "w") as f:
        json.dump({"pairs": pairs, "results": pilot_results, "parse_rate": parse_rate}, f, indent=2)
    print(f"  Saved pilot results to {pilot_path}")

    return pilot_results, parse_rate


# ────────────────────────────────────────────────────────────────────────────────
# Main phases
# ────────────────────────────────────────────────────────────────────────────────

def phase1_construct_and_screen(model):
    """Phase 1a: Construct 300 pairs and run screening."""
    print("=== Phase 1: Pair construction + screening ===")

    task_texts = load_task_texts()
    df = load_thurstonian()

    pairs = construct_pairs(df, task_texts, seed=SEED)

    pairs_path = RESULTS_DIR / "pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} pairs to {pairs_path}")

    print(f"\nRunning screening ({len(pairs)} pairs × 2 orderings × {SCREENING_RESAMPLES} resamples)...")
    screening_jsonl = RESULTS_DIR / "screening.jsonl"
    screening = run_screening(model, pairs, output_path=screening_jsonl)

    screening_path = RESULTS_DIR / "screening.json"
    with open(screening_path, "w") as f:
        json.dump(screening, f, indent=2)
    print(f"Saved screening results to {screening_path}")
    print(f"Borderline pairs: {screening['n_borderline']}/{screening['n_pairs']}")

    return pairs, screening


def phase1_steering(model, tokenizer, pairs: list[dict], screening: dict, calibration: dict):
    """Phase 1b: Run steering on borderline pairs."""
    from src.probes.core.storage import load_probe_direction

    print("=== Phase 1: Steering on borderline pairs ===")

    borderline_ids = set(screening["borderline_pair_ids"])
    borderline_pairs = [p for p in pairs if p["pair_id"] in borderline_ids]
    print(f"Running steering on {len(borderline_pairs)} borderline pairs")

    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PRIMARY_PROBE)
    coefs = calibration[PRIMARY_PROBE]["coefficients"]
    nonzero_coefs = [c for c in coefs if c != 0.0]
    print(f"Coefficients: {[f'{c:.0f}' for c in coefs]}")
    print(f"Non-zero: {[f'{c:.0f}' for c in nonzero_coefs]}")

    phase1_jsonl = RESULTS_DIR / "steering_phase1.jsonl"
    steering_results = run_steering_batch(
        model=model,
        tokenizer=tokenizer,
        pairs=borderline_pairs,
        direction=direction,
        layer=layer,
        coefficients=nonzero_coefs,
        n_resamples=STEERING_RESAMPLES,
        output_path=phase1_jsonl,
        label="Phase1-Steering",
    )

    steering_path = RESULTS_DIR / "steering_phase1.json"
    with open(steering_path, "w") as f:
        json.dump({
            "n_borderline_pairs": len(borderline_pairs),
            "layer": layer,
            "probe_id": PRIMARY_PROBE,
            "coefficients": coefs,
            "results": steering_results,
        }, f, indent=2)
    print(f"Saved Phase 1 steering results to {steering_path}")

    return steering_results


def phase2_decisive_pairs(model, tokenizer, pairs: list[dict], screening: dict, calibration: dict):
    """Phase 2: Run steering on decisive pairs stratified by |Δmu|."""
    from src.probes.core.storage import load_probe_direction

    print("=== Phase 2: Decisive pairs steering ===")

    borderline_ids = set(screening["borderline_pair_ids"])
    decisive_pairs = [p for p in pairs if p["pair_id"] not in borderline_ids]
    print(f"Total decisive pairs: {len(decisive_pairs)}")

    # Compute |Δmu| terciles
    delta_mus = np.array([p["delta_mu"] for p in decisive_pairs])
    t1, t2 = np.percentile(delta_mus, [33.3, 66.7])
    print(f"Δmu tercile thresholds: {t1:.3f}, {t2:.3f}")

    terciles = {
        "small": [p for p in decisive_pairs if p["delta_mu"] <= t1],
        "medium": [p for p in decisive_pairs if t1 < p["delta_mu"] <= t2],
        "large": [p for p in decisive_pairs if p["delta_mu"] > t2],
    }
    for name, ps in terciles.items():
        print(f"  {name}: {len(ps)} pairs")

    # Sample 30 from each tercile
    rng = random.Random(SEED + 1)
    sampled = []
    for name, ps in terciles.items():
        sample = ps[:30] if len(ps) <= 30 else rng.sample(ps, 30)
        for p in sample:
            p["delta_mu_bin"] = name
        sampled.extend(sample)

    print(f"Running steering on {len(sampled)} decisive pairs (30 per tercile)")

    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PRIMARY_PROBE)
    coefs = calibration[PRIMARY_PROBE]["coefficients"]
    nonzero_coefs = [c for c in coefs if c != 0.0]

    # For decisive pairs, run boost_a only (most informative condition)
    phase2_jsonl = RESULTS_DIR / "steering_phase2.jsonl"
    results = run_steering_batch(
        model=model,
        tokenizer=tokenizer,
        pairs=sampled,
        direction=direction,
        layer=layer,
        coefficients=nonzero_coefs,
        n_resamples=STEERING_RESAMPLES,
        output_path=phase2_jsonl,
        conditions=["boost_a"],
        extra_fields_fn=lambda p: {"delta_mu": p["delta_mu"], "delta_mu_bin": p["delta_mu_bin"]},
        label="Phase2",
    )
    phase2_path = RESULTS_DIR / "steering_phase2.json"
    with open(phase2_path, "w") as f:
        json.dump({
            "n_sampled": len(sampled),
            "tercile_thresholds": [float(t1), float(t2)],
            "results": results,
        }, f, indent=2)
    print(f"Saved Phase 2 results to {phase2_path}")

    return results


def phase3_multi_layer(model, tokenizer, pairs: list[dict], screening: dict, calibration: dict):
    """Phase 3: Multi-layer steering on borderline pairs."""
    from src.probes.core.storage import load_probe_direction

    print("=== Phase 3: Multi-layer steering ===")

    borderline_ids = set(screening["borderline_pair_ids"])
    borderline_pairs = [p for p in pairs if p["pair_id"] in borderline_ids]
    print(f"Running multi-layer steering on {len(borderline_pairs)} borderline pairs")

    # Load directions for all layers
    directions = {}
    layers = {}
    for probe_id in ["ridge_L31", "ridge_L37", "ridge_L43"]:
        layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, probe_id)
        directions[probe_id] = direction
        layers[probe_id] = layer
        print(f"  Loaded {probe_id}: layer={layer}")

    coefs = calibration[PRIMARY_PROBE]["coefficients"]
    nonzero_coefs = [c for c in coefs if c != 0.0]

    phase3_jsonl = RESULTS_DIR / "steering_phase3.jsonl"
    results = run_multi_layer_steering(
        model=model,
        tokenizer=tokenizer,
        pairs=borderline_pairs,
        directions=directions,
        layers=layers,
        coefficients=nonzero_coefs,
        n_resamples=STEERING_RESAMPLES,
        output_path=phase3_jsonl,
    )

    phase3_path = RESULTS_DIR / "steering_phase3.json"
    with open(phase3_path, "w") as f:
        json.dump({
            "n_borderline_pairs": len(borderline_pairs),
            "coefficients": coefs,
            "results": results,
        }, f, indent=2)
    print(f"Saved Phase 3 results to {phase3_path}")

    return results


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def load_model():
    from src.models.huggingface_model import HuggingFaceModel
    print("Loading model: gemma-3-27b...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded. Layers: {model.n_layers}, hidden_dim: {model.hidden_dim}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[0, 1, 2, 3],
                        help="Which phase to run (0=calibration, 1=screening+steering, 2=decisive, 3=multi-layer)")
    args = parser.parse_args()

    if args.phase == 0:
        calibration = phase0_calibration()
        model = load_model()
        phase0_pilot(model, model.tokenizer, calibration)

    elif args.phase == 1:
        # Load calibration
        calib_path = RESULTS_DIR / "calibration.json"
        if not calib_path.exists():
            print("ERROR: Run phase 0 first to generate calibration.json")
            return
        with open(calib_path) as f:
            calibration = json.load(f)

        model = load_model()

        # Check for existing pairs/screening
        pairs_path = RESULTS_DIR / "pairs.json"
        screening_path = RESULTS_DIR / "screening.json"

        if pairs_path.exists() and screening_path.exists():
            print("Loading existing pairs and screening results...")
            with open(pairs_path) as f:
                pairs = json.load(f)
            with open(screening_path) as f:
                screening = json.load(f)
            print(f"Loaded {len(pairs)} pairs, {screening['n_borderline']} borderline")
        else:
            pairs, screening = phase1_construct_and_screen(model)

        phase1_steering(model, model.tokenizer, pairs, screening, calibration)

    elif args.phase == 2:
        calib_path = RESULTS_DIR / "calibration.json"
        pairs_path = RESULTS_DIR / "pairs.json"
        screening_path = RESULTS_DIR / "screening.json"

        for p in [calib_path, pairs_path, screening_path]:
            if not p.exists():
                print(f"ERROR: Missing {p}. Run phases 0 and 1 first.")
                return

        with open(calib_path) as f:
            calibration = json.load(f)
        with open(pairs_path) as f:
            pairs = json.load(f)
        with open(screening_path) as f:
            screening = json.load(f)

        model = load_model()
        phase2_decisive_pairs(model, model.tokenizer, pairs, screening, calibration)

    elif args.phase == 3:
        calib_path = RESULTS_DIR / "calibration.json"
        pairs_path = RESULTS_DIR / "pairs.json"
        screening_path = RESULTS_DIR / "screening.json"

        for p in [calib_path, pairs_path, screening_path]:
            if not p.exists():
                print(f"ERROR: Missing {p}. Run phases 0 and 1 first.")
                return

        with open(calib_path) as f:
            calibration = json.load(f)
        with open(pairs_path) as f:
            pairs = json.load(f)
        with open(screening_path) as f:
            screening = json.load(f)

        model = load_model()
        phase3_multi_layer(model, model.tokenizer, pairs, screening, calibration)


if __name__ == "__main__":
    main()
