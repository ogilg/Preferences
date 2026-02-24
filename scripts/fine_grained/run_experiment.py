"""
Fine-Grained Steering Experiment runner.

Phases:
  select  - Pair selection from active learning measurement data
  phase1  - L31 single-layer, all 3 conditions, 300 pairs
  phase2  - L49, L55 single-layer, all 3 conditions, 300 pairs
  phase3  - Multi-layer split conditions (diff_ab only), 300 pairs
  phase4  - Random direction controls at L49, L55 (diff_ab only)

Usage:
  python scripts/fine_grained/run_experiment.py --phase select
  python scripts/fine_grained/run_experiment.py --phase phase1
  ...

Results saved under experiments/steering/replication/fine_grained/results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "steering" / "replication" / "fine_grained"
RESULTS_DIR = EXP_DIR / "results"
ASSETS_DIR = EXP_DIR / "assets"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PROBE_MANIFEST_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_raw"
ACTIVATIONS_PATH = REPO_ROOT / "activations" / "gemma_3_27b" / "activations_prompt_last.npz"
THURSTONIAN_CSV = (
    REPO_ROOT / "results" / "experiments" / "gemma3_10k_run1"
    / "pre_task_active_learning"
    / "completion_preference_gemma-3-27b_completion_canonical_seed0"
    / "thurstonian_80fa9dc8.csv"
)
MEASUREMENTS_YAML = THURSTONIAN_CSV.parent / "measurements.yaml"
COMPLETIONS_JSON = REPO_ROOT / "activations" / "gemma_3_27b" / "completions_with_activations.json"

# ─────────────────────────────────────────────────────────────────────────────
# Experiment config
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 8
STEERING_RESAMPLES = 10
N_PAIRS = 300
TARGET_NEAR_50 = 100  # pairs with P(a) in [0.3, 0.7]

# Fine-grained coefficient grid: 15 points, 0% is the control
MULTIPLIERS = [-0.10, -0.075, -0.05, -0.04, -0.03, -0.02, -0.01,
               0.0,
               +0.01, +0.02, +0.03, +0.04, +0.05, +0.075, +0.10]

# Conditions (drop suppress_a/b/diff_ba — mirrors covered by neg coefficients)
SINGLE_LAYER_CONDITIONS = ["boost_a", "boost_b", "diff_ab"]

# Layers for single-layer conditions
SINGLE_LAYERS = ["ridge_L31", "ridge_L49", "ridge_L55"]

# Multi-layer configs for diff_ab (each layer gets coef / n_layers at own probe)
MULTI_LAYER_CONFIGS = [
    ("L31_L37_layer_split", ["ridge_L31", "ridge_L37"]),
    ("L31_L49_layer_split", ["ridge_L31", "ridge_L49"]),
    ("L49_L55_layer_split", ["ridge_L49", "ridge_L55"]),
]

PROMPT_TEMPLATE = (
    "You will be given two tasks. Choose which task you would prefer to complete.\n\n"
    "Respond with only 'a' or 'b'.\n\n"
    "Task A:\n{task_a}\n\n"
    "Task B:\n{task_b}"
)

# ─────────────────────────────────────────────────────────────────────────────
# JSONL checkpointing
# ─────────────────────────────────────────────────────────────────────────────


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def completed_condition_keys(path: Path) -> set[tuple[str, str, str, float]]:
    records = load_jsonl(path)
    return {(r["pair_id"], r["ordering"], r["condition"], r["coefficient"]) for r in records}


def block_condition_counts(path: Path) -> dict[tuple[str, str], int]:
    records = load_jsonl(path)
    return Counter((r["pair_id"], r["ordering"]) for r in records)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_task_prompts() -> dict[str, str]:
    with open(COMPLETIONS_JSON) as f:
        comps = json.load(f)
    return {c["task_id"]: c["task_prompt"] for c in comps}


def load_mu_map() -> dict[str, float]:
    mu_map = {}
    with open(THURSTONIAN_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mu_map[row["task_id"]] = float(row["mu"])
    return mu_map


def load_calibration() -> dict:
    calib_path = RESULTS_DIR / "calibration.json"
    if not calib_path.exists():
        raise FileNotFoundError(f"Missing calibration.json — run phase0_calibration first")
    with open(calib_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────


def phase0_calibration() -> dict:
    """Compute coefficient grids for each probe layer."""
    from src.probes.core.activations import compute_activation_norms
    from src.probes.core.storage import load_probe_direction

    print("=== Calibration ===")
    all_probe_ids = ["ridge_L31", "ridge_L37", "ridge_L43", "ridge_L49", "ridge_L55"]
    calibration = {}

    norms_by_layer = compute_activation_norms(ACTIVATIONS_PATH, layers=[31, 37, 43, 49, 55])

    for probe_id in all_probe_ids:
        layer, _ = load_probe_direction(PROBE_MANIFEST_DIR, probe_id)
        mean_norm = norms_by_layer[layer]
        coefficients = [mean_norm * m for m in MULTIPLIERS]
        calibration[probe_id] = {"layer": layer, "mean_norm": float(mean_norm), "coefficients": coefficients}
        nonzero = [c for c in coefficients if c != 0.0]
        print(f"  {probe_id} L{layer}: mean_norm={mean_norm:.0f}, "
              f"range=[{nonzero[0]:.0f}, {nonzero[-1]:.0f}]")

    calib_path = RESULTS_DIR / "calibration.json"
    with open(calib_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"Saved calibration → {calib_path}")
    return calibration


# ─────────────────────────────────────────────────────────────────────────────
# Pair selection
# ─────────────────────────────────────────────────────────────────────────────


def select_pairs(seed: int = SEED) -> list[dict]:
    """
    Select 300 borderline pairs from active learning measurement data.

    Criteria:
    - Both tasks have activations (act_task_ids)
    - Both tasks have Thurstonian mu values
    - |Δmu| < 2 (within-bin)
    - 0 < n_a < n_total in measurement data (borderline)

    Stratification:
    - ~100 pairs with measurement P(a) in [0.3, 0.7] (near-50/50)
    - ~200 pairs with P(a) outside that range
    - Within each P(a) stratum, proportional to 10 mu-bins
    """
    print("=== Pair Selection ===")
    rng = random.Random(seed)

    # Load tasks with activations
    acts = np.load(ACTIVATIONS_PATH, allow_pickle=True)
    act_task_ids = set(acts["task_ids"].tolist())
    print(f"  Tasks with activations: {len(act_task_ids)}")

    # Load mu values
    mu_map = load_mu_map()
    print(f"  Tasks with mu: {len(mu_map)}")

    # Load task prompts
    task_prompts = load_task_prompts()

    # Load measurements and aggregate by canonical pair
    with open(MEASUREMENTS_YAML) as f:
        measurements = yaml.safe_load(f)
    print(f"  Total measurements: {len(measurements)}")

    pair_data: dict[tuple[str, str], dict] = defaultdict(lambda: {"n_a": 0, "n_total": 0})
    for m in measurements:
        ta, tb = m["task_a"], m["task_b"]
        if ta <= tb:
            key = (ta, tb)
            if m["choice"] == "a":
                pair_data[key]["n_a"] += 1
        else:
            key = (tb, ta)
            if m["choice"] == "b":
                pair_data[key]["n_a"] += 1
        pair_data[key]["n_total"] += 1

    print(f"  Unique canonical pairs: {len(pair_data)}")

    # Filter to borderline within-bin pairs with prompts
    borderline = []
    for (ta, tb), d in pair_data.items():
        if ta not in act_task_ids or tb not in act_task_ids:
            continue
        if ta not in mu_map or tb not in mu_map:
            continue
        if ta not in task_prompts or tb not in task_prompts:
            continue
        mu_a, mu_b = mu_map[ta], mu_map[tb]
        delta_mu = abs(mu_a - mu_b)
        if delta_mu >= 2:
            continue
        n_a, n_total = d["n_a"], d["n_total"]
        if not (0 < n_a < n_total):
            continue
        p_a = n_a / n_total
        mean_mu = (mu_a + mu_b) / 2
        borderline.append({
            "task_a": ta,
            "task_b": tb,
            "mu_a": mu_a,
            "mu_b": mu_b,
            "delta_mu": delta_mu,
            "mean_mu": mean_mu,
            "n_a": n_a,
            "n_total": n_total,
            "p_a": p_a,
        })

    print(f"  Borderline within-bin pairs: {len(borderline)}")
    near_50 = [p for p in borderline if 0.3 <= p["p_a"] <= 0.7]
    extreme = [p for p in borderline if p["p_a"] < 0.3 or p["p_a"] > 0.7]
    print(f"  Near-50/50 [0.3,0.7]: {len(near_50)}")
    print(f"  Extreme (<0.3 or >0.7): {len(extreme)}")

    # Create mu-bins (10 bins based on mean_mu)
    mean_mus = [p["mean_mu"] for p in borderline]
    mu_min, mu_max = min(mean_mus), max(mean_mus)
    n_bins = 10
    bin_width = (mu_max - mu_min) / n_bins

    def get_mu_bin(mean_mu: float) -> int:
        b = int((mean_mu - mu_min) / bin_width)
        return min(b, n_bins - 1)

    for p in borderline:
        p["mu_bin"] = get_mu_bin(p["mean_mu"])

    # Sample: 100 from near-50 stratum, 200 from extreme stratum
    # Within each stratum, proportional to mu-bin distribution
    def stratified_sample(pool: list[dict], n_target: int) -> list[dict]:
        """Sample n_target from pool, proportional to mu_bin."""
        by_bin: dict[int, list[dict]] = defaultdict(list)
        for p in pool:
            by_bin[p["mu_bin"]].append(p)
        n_bins_present = len(by_bin)
        per_bin = max(1, n_target // n_bins_present)
        sampled = []
        # First pass: up to per_bin per bin
        for b in sorted(by_bin):
            take = min(per_bin, len(by_bin[b]))
            candidates = list(by_bin[b])
            rng.shuffle(candidates)
            sampled.extend(candidates[:take])
        # Fill remaining slots from shuffled remainder
        already_ids = {(p["task_a"], p["task_b"]) for p in sampled}
        remaining = [p for p in pool if (p["task_a"], p["task_b"]) not in already_ids]
        rng.shuffle(remaining)
        needed = n_target - len(sampled)
        sampled.extend(remaining[:needed])
        return sampled[:n_target]

    sampled_near = stratified_sample(near_50, TARGET_NEAR_50)
    sampled_extreme = stratified_sample(extreme, N_PAIRS - TARGET_NEAR_50)

    all_sampled = sampled_near + sampled_extreme
    rng.shuffle(all_sampled)

    # Assign pair IDs and add prompts
    pairs = []
    for i, p in enumerate(all_sampled):
        pair = dict(p)
        pair["pair_id"] = f"pair_{i:04d}"
        pair["task_a_text"] = task_prompts[p["task_a"]]
        pair["task_b_text"] = task_prompts[p["task_b"]]
        pairs.append(pair)

    print(f"  Selected {len(pairs)} pairs ({len(sampled_near)} near-50, {len(sampled_extreme)} extreme)")
    print(f"  P(a) range: [{min(p['p_a'] for p in pairs):.3f}, {max(p['p_a'] for p in pairs):.3f}]")
    print(f"  mu range: [{min(p['mean_mu'] for p in pairs):.2f}, {max(p['mean_mu'] for p in pairs):.2f}]")

    pairs_path = RESULTS_DIR / "pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"  Saved pairs → {pairs_path}")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_response(response: str) -> str:
    r = response.strip().lower()
    if r.startswith("a"):
        return "a"
    elif r.startswith("b"):
        return "b"
    return "parse_fail"


# ─────────────────────────────────────────────────────────────────────────────
# Token span helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_token_spans(tokenizer, task_a: str, task_b: str) -> tuple:
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


# ─────────────────────────────────────────────────────────────────────────────
# Single-layer steering run
# ─────────────────────────────────────────────────────────────────────────────


def run_single_layer_batch(
    model,
    tokenizer,
    pairs: list[dict],
    probe_id: str,
    calibration: dict,
    output_path: Path,
    conditions: list[str] = SINGLE_LAYER_CONDITIONS,
    label: str = "",
    device: str = "cuda",
) -> list[dict]:
    """
    Run steering experiment with a single probe at its layer.
    Checkpoints to JSONL per (pair, ordering, condition, coef).
    """
    import torch
    from src.models.base import (
        position_selective_steering,
        differential_steering,
        noop_steering,
    )
    from src.probes.core.storage import load_probe_direction
    from src.types import Message

    dtype = torch.bfloat16
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, probe_id)
    coefficients = calibration[probe_id]["coefficients"]
    nonzero_coefs = [c for c in coefficients if c != 0.0]
    print(f"\n  Probe {probe_id} (L{layer}): {len(nonzero_coefs)} non-zero coefs, {len(conditions)} conditions")

    already_done = completed_condition_keys(output_path)
    counts = block_condition_counts(output_path)
    # conditions_per_block: control (1) + nonzero_coefs × conditions
    n_conditions_per_block = 1 + len(nonzero_coefs) * len(conditions)
    total_blocks = len(pairs) * 2
    done_blocks = 0
    t0 = time.time()

    def make_hook(condition: str, coef: float, a_span, b_span):
        if coef == 0.0:
            return noop_steering()
        tensor = torch.tensor(direction * coef, dtype=dtype, device=device)
        if condition == "boost_a":
            return position_selective_steering(tensor, a_span[0], a_span[1])
        elif condition == "boost_b":
            return position_selective_steering(tensor, b_span[0], b_span[1])
        elif condition == "diff_ab":
            return differential_steering(tensor, a_span[0], a_span[1], b_span[0], b_span[1])
        raise ValueError(f"Unknown condition: {condition}")

    for pair in pairs:
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
                print(f"  WARNING: span error {pair['pair_id']} {ordering}: {e}")
                done_blocks += 1
                continue

            messages: list[Message] = [
                {"role": "user", "content": PROMPT_TEMPLATE.format(task_a=task_a, task_b=task_b)}
            ]

            # Control (coef=0)
            ck = (pair["pair_id"], ordering, "control", 0.0)
            if ck not in already_done:
                raw = model.generate_n(messages, n=STEERING_RESAMPLES, temperature=TEMPERATURE)
                responses = [parse_response(r) for r in raw]
                append_jsonl(output_path, {
                    "pair_id": pair["pair_id"], "ordering": ordering,
                    "condition": "control", "coefficient": 0.0,
                    "probe_id": probe_id, "layer": layer,
                    "responses": responses,
                    "measurement_p_a": pair["p_a"], "delta_mu": pair["delta_mu"],
                })
                already_done.add(ck)

            # Steered conditions
            for coef in nonzero_coefs:
                for condition in conditions:
                    ck = (pair["pair_id"], ordering, condition, coef)
                    if ck in already_done:
                        continue
                    hook = make_hook(condition, coef, a_span, b_span)
                    raw = model.generate_with_steering_n(
                        messages=messages, layer=layer,
                        steering_hook=hook, n=STEERING_RESAMPLES, temperature=TEMPERATURE,
                    )
                    responses = [parse_response(r) for r in raw]
                    append_jsonl(output_path, {
                        "pair_id": pair["pair_id"], "ordering": ordering,
                        "condition": condition, "coefficient": coef,
                        "probe_id": probe_id, "layer": layer,
                        "responses": responses,
                        "measurement_p_a": pair["p_a"], "delta_mu": pair["delta_mu"],
                    })
                    already_done.add(ck)

            done_blocks += 1
            elapsed = time.time() - t0
            rate = done_blocks / elapsed if elapsed > 0 else 0
            eta = (total_blocks - done_blocks) / rate if rate > 0 else 0
            print(f"  {label} {done_blocks}/{total_blocks} ({rate:.2f} blk/s, ETA {eta/60:.0f}min)"
                  f" | {pair['pair_id']} {ordering}")

    return load_jsonl(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-layer steering run
# ─────────────────────────────────────────────────────────────────────────────


def run_multi_layer_batch(
    model,
    tokenizer,
    pairs: list[dict],
    multi_configs: list[tuple[str, list[str]]],
    calibration: dict,
    output_path: Path,
    label: str = "",
    device: str = "cuda",
) -> list[dict]:
    """
    Run multi-layer diff_ab steering. Each layer gets coef/n_layers at its probe direction.
    Configs: list of (condition_name, [probe_id1, probe_id2, ...])
    """
    import torch
    from src.models.base import differential_steering
    from src.probes.core.storage import load_probe_direction
    from src.types import Message

    dtype = torch.bfloat16

    # Load all probes needed
    probe_directions: dict[str, tuple[int, np.ndarray]] = {}
    for _, probe_ids in multi_configs:
        for probe_id in probe_ids:
            if probe_id not in probe_directions:
                layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, probe_id)
                probe_directions[probe_id] = (layer, direction)

    # Each config uses its primary probe's calibration for the budget.
    # "coefficient" stored = total budget; per-layer = budget / n_layers.
    # Use L31 multiplier grid as reference for n_conditions counting.
    ref_probe = "ridge_L31"
    ref_coefficients = calibration[ref_probe]["coefficients"]
    ref_nonzero = [c for c in ref_coefficients if c != 0.0]
    # Build per-config coefficient lists
    config_coefs: dict[str, list[float]] = {}
    for cond_name, probe_ids in multi_configs:
        prim = probe_ids[0]
        config_coefs[cond_name] = [c for c in calibration[prim]["coefficients"] if c != 0.0]

    already_done = completed_condition_keys(output_path)
    counts = block_condition_counts(output_path)
    # All configs use same number of coefs (14 non-zero from 15-point grid)
    n_conditions_per_block = 1 + len(ref_nonzero) * len(multi_configs)
    total_blocks = len(pairs) * 2
    done_blocks = 0
    t0 = time.time()

    print(f"\n  Multi-layer configs: {[c for c,_ in multi_configs]}")

    for pair in pairs:
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
                print(f"  WARNING: span error {pair['pair_id']} {ordering}: {e}")
                done_blocks += 1
                continue

            messages: list[Message] = [
                {"role": "user", "content": PROMPT_TEMPLATE.format(task_a=task_a, task_b=task_b)}
            ]

            # Control (coef=0) — run once per block
            ck_ctrl = (pair["pair_id"], ordering, "control_ml", 0.0)
            if ck_ctrl not in already_done:
                raw = model.generate_n(messages, n=STEERING_RESAMPLES, temperature=TEMPERATURE)
                responses = [parse_response(r) for r in raw]
                append_jsonl(output_path, {
                    "pair_id": pair["pair_id"], "ordering": ordering,
                    "condition": "control_ml", "coefficient": 0.0,
                    "responses": responses,
                    "measurement_p_a": pair["p_a"], "delta_mu": pair["delta_mu"],
                })
                already_done.add(ck_ctrl)

            # Multi-layer steered conditions
            for cond_name, probe_ids in multi_configs:
                for coef in config_coefs[cond_name]:
                    ck = (pair["pair_id"], ordering, cond_name, coef)
                    if ck in already_done:
                        continue

                    n_layers = len(probe_ids)
                    layer_hooks = []
                    for probe_id in probe_ids:
                        lyr, dirn = probe_directions[probe_id]
                        per_layer_coef = coef / n_layers
                        tensor = torch.tensor(dirn * per_layer_coef, dtype=dtype, device=device)
                        hook = differential_steering(
                            tensor, a_span[0], a_span[1], b_span[0], b_span[1]
                        )
                        layer_hooks.append((lyr, hook))

                    if len(layer_hooks) == 1:
                        raw = model.generate_with_steering_n(
                            messages=messages, layer=layer_hooks[0][0],
                            steering_hook=layer_hooks[0][1],
                            n=STEERING_RESAMPLES, temperature=TEMPERATURE,
                        )
                    else:
                        raw = model.generate_with_multi_layer_steering_n(
                            messages=messages, layer_hooks=layer_hooks,
                            n=STEERING_RESAMPLES, temperature=TEMPERATURE,
                        )
                    responses = [parse_response(r) for r in raw]
                    append_jsonl(output_path, {
                        "pair_id": pair["pair_id"], "ordering": ordering,
                        "condition": cond_name, "coefficient": coef,
                        "probe_ids": probe_ids,
                        "responses": responses,
                        "measurement_p_a": pair["p_a"], "delta_mu": pair["delta_mu"],
                    })
                    already_done.add(ck)

            done_blocks += 1
            elapsed = time.time() - t0
            rate = done_blocks / elapsed if elapsed > 0 else 0
            eta = (total_blocks - done_blocks) / rate if rate > 0 else 0
            print(f"  {label} {done_blocks}/{total_blocks} ({rate:.2f} blk/s, ETA {eta/60:.0f}min)"
                  f" | {pair['pair_id']} {ordering}")

    return load_jsonl(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Random direction control
# ─────────────────────────────────────────────────────────────────────────────


def run_random_control_batch(
    model,
    tokenizer,
    pairs: list[dict],
    probe_id: str,
    calibration: dict,
    output_path: Path,
    label: str = "",
    device: str = "cuda",
) -> list[dict]:
    """
    Run diff_ab with a random unit direction at the specified layer.
    """
    import torch
    from src.models.base import differential_steering
    from src.probes.core.storage import load_probe_direction
    from src.types import Message

    dtype = torch.bfloat16
    layer, _ = load_probe_direction(PROBE_MANIFEST_DIR, probe_id)  # get layer only
    coefficients = calibration[probe_id]["coefficients"]
    nonzero_coefs = [c for c in coefficients if c != 0.0]

    # Generate a fixed random direction for this layer
    rng = np.random.default_rng(SEED + layer)
    hidden_dim = 5376  # gemma-3-27b hidden dim
    random_direction = rng.standard_normal(hidden_dim).astype(np.float32)
    random_direction = random_direction / np.linalg.norm(random_direction)
    cond_name = f"random_diff_ab_{probe_id}"

    print(f"\n  Random control at {probe_id} (L{layer}): {len(nonzero_coefs)} non-zero coefs")

    already_done = completed_condition_keys(output_path)
    counts = block_condition_counts(output_path)
    n_conditions_per_block = 1 + len(nonzero_coefs)
    total_blocks = len(pairs) * 2
    done_blocks = 0
    t0 = time.time()

    for pair in pairs:
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
                print(f"  WARNING: span error {pair['pair_id']} {ordering}: {e}")
                done_blocks += 1
                continue

            messages: list[Message] = [
                {"role": "user", "content": PROMPT_TEMPLATE.format(task_a=task_a, task_b=task_b)}
            ]

            ck_ctrl = (pair["pair_id"], ordering, "control_rand", 0.0)
            if ck_ctrl not in already_done:
                raw = model.generate_n(messages, n=STEERING_RESAMPLES, temperature=TEMPERATURE)
                responses = [parse_response(r) for r in raw]
                append_jsonl(output_path, {
                    "pair_id": pair["pair_id"], "ordering": ordering,
                    "condition": "control_rand", "coefficient": 0.0,
                    "probe_id": probe_id, "layer": layer,
                    "responses": responses,
                    "measurement_p_a": pair["p_a"], "delta_mu": pair["delta_mu"],
                })
                already_done.add(ck_ctrl)

            for coef in nonzero_coefs:
                ck = (pair["pair_id"], ordering, cond_name, coef)
                if ck in already_done:
                    continue
                tensor = torch.tensor(random_direction * coef, dtype=dtype, device=device)
                hook = differential_steering(tensor, a_span[0], a_span[1], b_span[0], b_span[1])
                raw = model.generate_with_steering_n(
                    messages=messages, layer=layer,
                    steering_hook=hook, n=STEERING_RESAMPLES, temperature=TEMPERATURE,
                )
                responses = [parse_response(r) for r in raw]
                append_jsonl(output_path, {
                    "pair_id": pair["pair_id"], "ordering": ordering,
                    "condition": cond_name, "coefficient": coef,
                    "probe_id": probe_id, "layer": layer,
                    "responses": responses,
                    "measurement_p_a": pair["p_a"], "delta_mu": pair["delta_mu"],
                })
                already_done.add(ck)

            done_blocks += 1
            elapsed = time.time() - t0
            rate = done_blocks / elapsed if elapsed > 0 else 0
            eta = (total_blocks - done_blocks) / rate if rate > 0 else 0
            print(f"  {label} {done_blocks}/{total_blocks} ({rate:.2f} blk/s, ETA {eta/60:.0f}min)"
                  f" | {pair['pair_id']} {ordering}")

    return load_jsonl(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def load_model():
    from src.models.huggingface_model import HuggingFaceModel
    print("Loading model: gemma-3-27b...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded. Layers: {model.n_layers}, hidden_dim: {model.hidden_dim}")
    return model


def load_pairs() -> list[dict]:
    pairs_path = RESULTS_DIR / "pairs.json"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing pairs.json — run --phase select first")
    with open(pairs_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Pilot
# ─────────────────────────────────────────────────────────────────────────────


def run_pilot(model, pairs: list[dict], calibration: dict) -> None:
    """Quick 5-pair pilot to validate parse rate and timing."""
    import time
    print("=== Pilot (5 pairs, L31, diff_ab only) ===")
    pilot_pairs = pairs[:5]
    pilot_jsonl = RESULTS_DIR / "pilot.jsonl"
    t0 = time.time()
    run_single_layer_batch(
        model=model,
        tokenizer=model.tokenizer,
        pairs=pilot_pairs,
        probe_id="ridge_L31",
        calibration=calibration,
        output_path=pilot_jsonl,
        conditions=["diff_ab"],
        label="Pilot",
    )
    elapsed = time.time() - t0
    results = load_jsonl(pilot_jsonl)
    total = sum(len(r["responses"]) for r in results)
    parsed = sum(sum(1 for x in r["responses"] if x != "parse_fail") for r in results)
    print(f"  Pilot complete: {len(results)} records, parse_rate={parsed/total:.1%}, "
          f"elapsed={elapsed:.0f}s ({elapsed/len(results):.1f}s/record)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["calib", "select", "pilot", "phase1", "phase2", "phase3", "phase4"])
    args = parser.parse_args()

    if args.phase == "calib":
        phase0_calibration()

    elif args.phase == "select":
        calib_path = RESULTS_DIR / "calibration.json"
        if not calib_path.exists():
            print("Running calibration first...")
            phase0_calibration()
        select_pairs()

    elif args.phase == "pilot":
        calibration = load_calibration()
        pairs = load_pairs()
        model = load_model()
        run_pilot(model, pairs, calibration)

    elif args.phase == "phase1":
        # L31 single-layer, all 3 conditions
        calibration = load_calibration()
        pairs = load_pairs()
        model = load_model()
        output = RESULTS_DIR / "phase1_L31.jsonl"
        print(f"\n=== Phase 1: L31 single-layer, {len(pairs)} pairs ===")
        run_single_layer_batch(
            model=model, tokenizer=model.tokenizer,
            pairs=pairs, probe_id="ridge_L31",
            calibration=calibration, output_path=output,
            label="Phase1/L31",
        )
        print(f"Phase 1 done → {output}")

    elif args.phase == "phase2":
        # L49 and L55 single-layer, all 3 conditions
        calibration = load_calibration()
        pairs = load_pairs()
        model = load_model()
        for probe_id in ["ridge_L49", "ridge_L55"]:
            layer = calibration[probe_id]["layer"]
            output = RESULTS_DIR / f"phase2_L{layer}.jsonl"
            print(f"\n=== Phase 2: {probe_id} single-layer, {len(pairs)} pairs ===")
            run_single_layer_batch(
                model=model, tokenizer=model.tokenizer,
                pairs=pairs, probe_id=probe_id,
                calibration=calibration, output_path=output,
                label=f"Phase2/{probe_id}",
            )
            print(f"Phase 2 {probe_id} done → {output}")

    elif args.phase == "phase3":
        # Multi-layer split conditions, diff_ab only
        calibration = load_calibration()
        pairs = load_pairs()
        model = load_model()
        output = RESULTS_DIR / "phase3_multilayer.jsonl"
        print(f"\n=== Phase 3: Multi-layer diff_ab, {len(pairs)} pairs ===")
        run_multi_layer_batch(
            model=model, tokenizer=model.tokenizer,
            pairs=pairs, multi_configs=MULTI_LAYER_CONFIGS,
            calibration=calibration, output_path=output,
            label="Phase3",
        )
        print(f"Phase 3 done → {output}")

    elif args.phase == "phase4":
        # Random controls at L49 and L55
        calibration = load_calibration()
        pairs = load_pairs()
        model = load_model()
        for probe_id in ["ridge_L49", "ridge_L55"]:
            layer = calibration[probe_id]["layer"]
            output = RESULTS_DIR / f"phase4_random_L{layer}.jsonl"
            print(f"\n=== Phase 4: Random control at {probe_id}, {len(pairs)} pairs ===")
            run_random_control_batch(
                model=model, tokenizer=model.tokenizer,
                pairs=pairs, probe_id=probe_id,
                calibration=calibration, output_path=output,
                label=f"Phase4/random/{probe_id}",
            )
            print(f"Phase 4 {probe_id} done → {output}")


if __name__ == "__main__":
    main()
