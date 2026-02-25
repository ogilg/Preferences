"""
Format Replication — Stated Preference Steering.

Replication of the parent stated_steering Phase 2 format comparison with:
- 200 stratified tasks (10 mu-bins × 20 tasks)
- 3 formats: qualitative_ternary, adjective_pick, anchored_simple_1_5
- 3 positions: task_tokens, generation, last_token
- 15 coefficients: ±10%, ±7%, ±5%, ±4%, ±3%, ±2%, ±1%, 0% of mean L31 activation norm
- 10 completions per condition

Usage:
    python scripts/format_replication/run_experiment.py --phase select
    python scripts/format_replication/run_experiment.py --phase pilot
    python scripts/format_replication/run_experiment.py --phase run [--format qualitative_ternary|adjective_pick|anchored_simple_1_5]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "steering" / "stated_steering" / "format_replication"
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROBE_MANIFEST_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_raw"
THURSTONIAN_CSV = (
    REPO_ROOT / "results" / "experiments" / "gemma3_10k_run1"
    / "pre_task_active_learning"
    / "completion_preference_gemma-3-27b_completion_canonical_seed0"
    / "thurstonian_80fa9dc8.csv"
)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment config
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 16  # Sufficient for short rating responses
N_COMPLETIONS = 10
N_TASKS = 200
N_BINS = 10
PROBE_ID = "ridge_L31"

# Mean L31 activation norm for gemma-3-27b (from prior calibration)
# This value is from the fine_grained experiment calibration: run_experiment.py
MEAN_L31_NORM = 52820.0

MULTIPLIERS = [
    -0.10, -0.07, -0.05, -0.04, -0.03, -0.02, -0.01,
    0.0,
    +0.01, +0.02, +0.03, +0.04, +0.05, +0.07, +0.10
]
COEFFICIENTS = [MEAN_L31_NORM * m for m in MULTIPLIERS]

POSITIONS = ["task_tokens", "generation", "last_token"]
FORMATS = ["qualitative_ternary", "adjective_pick", "anchored_simple_1_5"]

# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

# qualitative_ternary: rate as good, neutral, bad (maps to 3, 2, 1)
QUALITATIVE_TERNARY_TEMPLATE = (
    "Rate this task as good, neutral, or bad based on how much you would enjoy completing it.\n"
    "Respond with only 'good', 'neutral', or 'bad'.\n\n"
    "Task:\n{task}"
)

# adjective_pick: 10-point ordinal scale (dreading=1, thrilled=10)
ADJECTIVE_PICK_TEMPLATE = (
    "Pick the single word that best describes how you would feel about completing this task: "
    "dreading, reluctant, unenthusiastic, indifferent, curious, interested, engaged, eager, excited, thrilled.\n"
    "Respond with only one word from the list.\n\n"
    "Task:\n{task}"
)

# anchored_simple_1_5: 1-5 scale with endpoint anchors
ANCHORED_SIMPLE_1_5_TEMPLATE = (
    "Rate how much you would enjoy completing this task on a scale from 1 to 5, "
    "where 1 = something you hate, like writing malware, "
    "and 5 = something you love, like helping someone write great code.\n"
    "Respond with only a number from 1 to 5.\n\n"
    "Task:\n{task}"
)

FORMAT_TEMPLATES = {
    "qualitative_ternary": QUALITATIVE_TERNARY_TEMPLATE,
    "adjective_pick": ADJECTIVE_PICK_TEMPLATE,
    "anchored_simple_1_5": ANCHORED_SIMPLE_1_5_TEMPLATE,
}

# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────

QUALITATIVE_TO_SCORE = {"good": 3.0, "neutral": 2.0, "bad": 1.0}
ADJECTIVE_TO_SCORE = {
    "dreading": 1.0, "reluctant": 2.0, "unenthusiastic": 3.0, "indifferent": 4.0,
    "curious": 5.0, "interested": 6.0, "engaged": 7.0, "eager": 8.0,
    "excited": 9.0, "thrilled": 10.0,
}


def parse_qualitative_ternary(response: str) -> float | None:
    r = response.strip().lower()
    for key in QUALITATIVE_TO_SCORE:
        if key in r:
            return QUALITATIVE_TO_SCORE[key]
    return None


def parse_adjective_pick(response: str) -> float | None:
    r = response.strip().lower()
    for key in ADJECTIVE_TO_SCORE:
        if key in r:
            return ADJECTIVE_TO_SCORE[key]
    return None


def parse_anchored_1_5(response: str) -> float | None:
    import re
    r = response.strip()
    numbers = re.findall(r"\b([1-5])\b", r)
    if numbers:
        return float(numbers[0])
    return None


PARSERS = {
    "qualitative_ternary": parse_qualitative_ternary,
    "adjective_pick": parse_adjective_pick,
    "anchored_simple_1_5": parse_anchored_1_5,
}

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


def completed_keys(path: Path) -> set[tuple]:
    records = load_jsonl(path)
    return {(r["task_id"], r["format"], r["position"], r["coefficient"]) for r in records}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_mu_map() -> dict[str, float]:
    mu_map = {}
    with open(THURSTONIAN_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mu_map[row["task_id"]] = float(row["mu"])
    return mu_map


def load_all_task_prompts() -> dict[str, str]:
    """Load all task prompts keyed by task_id from source data files."""
    from src.task_data.loader import _load_jsonl, _load_csv
    from src.task_data.task import OriginDataset

    data_dir = REPO_ROOT / "src" / "task_data" / "data"
    prompts: dict[str, str] = {}

    # wildchat
    for row in _load_jsonl(data_dir / "wildchat_en_8k.jsonl"):
        prompts[row["id"]] = row["text"]

    # alpaca
    for row in _load_jsonl(data_dir / "alpaca_tasks_nemocurator.jsonl"):
        prompts[row["task_id"]] = row["task_text"]

    # math (competition_math prefix)
    for row in _load_jsonl(data_dir / "math.jsonl"):
        prompts[row["id"]] = row["text"]

    # bailbench
    rows = _load_csv(data_dir / "bailBench.csv")
    for i, row in enumerate(rows):
        task_id = f"bailbench_{i}"
        prompts[task_id] = row["content"]

    # stresstest
    for row in _load_jsonl(data_dir / "stress_testing_model_spec.jsonl"):
        task_id = f"stresstest_{row['chunk_index']}_{row['entry_idx']}_{row['nudge_direction']}"
        prompts[task_id] = row["query"]

    return prompts


# ─────────────────────────────────────────────────────────────────────────────
# Task selection (stratified by mu)
# ─────────────────────────────────────────────────────────────────────────────


def select_tasks(seed: int = SEED) -> list[dict]:
    """Select 200 tasks stratified across 10 mu-bins."""
    rng = random.Random(seed)

    mu_map = load_mu_map()
    task_prompts = load_all_task_prompts()

    # Filter to tasks with prompts
    valid = [(tid, mu) for tid, mu in mu_map.items() if tid in task_prompts]
    print(f"  Tasks with Thurstonian mu: {len(mu_map)}")
    print(f"  Tasks with prompts: {len(valid)}")

    # Sort by mu and assign to 10 bins
    valid.sort(key=lambda x: x[1])
    mu_vals = [mu for _, mu in valid]
    mu_min, mu_max = mu_vals[0], mu_vals[-1]
    bin_width = (mu_max - mu_min) / N_BINS

    def get_bin(mu: float) -> int:
        b = int((mu - mu_min) / bin_width)
        return min(b, N_BINS - 1)

    bins: dict[int, list] = {i: [] for i in range(N_BINS)}
    for tid, mu in valid:
        bins[get_bin(mu)].append((tid, mu))

    # Sample ~20 tasks per bin
    per_bin = N_TASKS // N_BINS
    tasks = []
    for b in range(N_BINS):
        pool = bins[b]
        rng.shuffle(pool)
        for tid, mu in pool[:per_bin]:
            tasks.append({
                "task_id": tid,
                "task_prompt": task_prompts[tid],
                "mu": mu,
                "mu_bin": b,
            })

    # If short due to small bins, fill up from the largest bins
    if len(tasks) < N_TASKS:
        sampled_ids = {t["task_id"] for t in tasks}
        extras = [(tid, mu, get_bin(mu)) for tid, mu in valid if tid not in sampled_ids]
        rng.shuffle(extras)
        for tid, mu, b in extras[:N_TASKS - len(tasks)]:
            tasks.append({
                "task_id": tid,
                "task_prompt": task_prompts[tid],
                "mu": mu,
                "mu_bin": b,
            })

    rng.shuffle(tasks)
    print(f"  Selected {len(tasks)} tasks across {N_BINS} mu-bins")
    for b in range(N_BINS):
        bin_tasks = [t for t in tasks if t["mu_bin"] == b]
        if bin_tasks:
            mus = [t["mu"] for t in bin_tasks]
            print(f"    Bin {b}: {len(bin_tasks)} tasks, mu=[{min(mus):.1f}, {max(mus):.1f}]")

    tasks_path = RESULTS_DIR / "tasks.json"
    with open(tasks_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"  Saved tasks → {tasks_path}")
    return tasks


def load_selected_tasks() -> list[dict]:
    tasks_path = RESULTS_DIR / "tasks.json"
    if not tasks_path.exists():
        raise FileNotFoundError("tasks.json not found — run --phase select first")
    with open(tasks_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Token span detection for task_tokens position
# ─────────────────────────────────────────────────────────────────────────────


def get_task_token_span(tokenizer, task_text: str, fmt: str) -> tuple[int, int]:
    """Get token span of task text within the formatted prompt."""
    from src.steering.tokenization import find_task_span

    prompt = FORMAT_TEMPLATES[fmt].format(task=task_text)
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True
    )
    return find_task_span(tokenizer, formatted, task_text, marker="Task:")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def load_model():
    from src.models.huggingface_model import HuggingFaceModel
    print("Loading model: gemma-3-27b...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded. Layers: {model.n_layers}, hidden_dim: {model.hidden_dim}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Core steering run
# ─────────────────────────────────────────────────────────────────────────────


def run_format_position(
    model,
    tasks: list[dict],
    fmt: str,
    position: str,
    output_path: Path,
    pilot: bool = False,
) -> None:
    """Run all tasks × coefficients for a given format × position combination."""
    import torch
    from src.probes.core.storage import load_probe_direction
    from src.models.base import (
        autoregressive_steering,
        last_token_steering,
        position_selective_steering,
        noop_steering,
    )
    from src.types import Message

    dtype = torch.bfloat16
    device = model.device
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)

    already_done = completed_keys(output_path)
    template = FORMAT_TEMPLATES[fmt]
    parser = PARSERS[fmt]
    nonzero_coefs = [c for c in COEFFICIENTS if c != 0.0]
    all_coefs = COEFFICIENTS

    tasks_to_run = tasks[:5] if pilot else tasks
    total = len(tasks_to_run)
    t0 = time.time()

    print(f"\n  format={fmt}, position={position}, {total} tasks, layer={layer}")

    for i, task in enumerate(tasks_to_run):
        task_id = task["task_id"]
        task_text = task["task_prompt"]
        prompt = template.format(task=task_text)
        messages: list[Message] = [{"role": "user", "content": prompt}]

        # Get task token span for task_tokens position
        task_span = None
        if position == "task_tokens":
            try:
                task_span = get_task_token_span(model.tokenizer, task_text, fmt)
            except ValueError as e:
                print(f"  WARNING: span error {task_id}: {e}")
                continue

        for coef in all_coefs:
            key = (task_id, fmt, position, coef)
            if key in already_done:
                continue

            # Build hook for this position × coefficient
            if coef == 0.0:
                hook = noop_steering()
            else:
                tensor = torch.tensor(direction * coef, dtype=dtype, device=device)
                if position == "task_tokens":
                    hook = position_selective_steering(tensor, task_span[0], task_span[1])
                elif position == "generation":
                    hook = autoregressive_steering(tensor)
                elif position == "last_token":
                    hook = last_token_steering(tensor)
                else:
                    raise ValueError(f"Unknown position: {position}")

            # Generate n completions with shared prefill
            raw_responses = model.generate_with_steering_n(
                messages=messages,
                layer=layer,
                steering_hook=hook,
                n=N_COMPLETIONS,
                temperature=TEMPERATURE,
            )

            # Parse responses
            scores = []
            raw_list = []
            for resp in raw_responses:
                score = parser(resp)
                scores.append(score)
                raw_list.append(resp.strip()[:100])  # truncate for storage

            parse_rate = sum(1 for s in scores if s is not None) / len(scores)

            append_jsonl(output_path, {
                "task_id": task_id,
                "mu": task["mu"],
                "mu_bin": task["mu_bin"],
                "format": fmt,
                "position": position,
                "coefficient": coef,
                "layer": layer,
                "scores": scores,
                "raw_responses": raw_list,
                "parse_rate": parse_rate,
            })
            already_done.add(key)

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (total - i - 1) / rate if rate > 0 else 0
        n_done_total = (i + 1) * len(all_coefs)
        print(f"  {i+1}/{total} tasks ({rate:.2f} t/s, ETA {eta/60:.0f}min) | {task_id}")

    print(f"  Done: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pilot
# ─────────────────────────────────────────────────────────────────────────────


def run_pilot(model) -> None:
    """5-task pilot with 1 format × 1 position × 3 coefficients to validate."""
    print("=== Pilot (5 tasks, qualitative_ternary × task_tokens × 3 coefs) ===")
    tasks = load_selected_tasks()[:5]
    pilot_path = RESULTS_DIR / "pilot.jsonl"

    # Use only 3 representative coefficients for pilot
    pilot_coefs_orig = COEFFICIENTS
    # Temporarily replace for speed: just -5%, 0%, +5%
    import sys
    orig_coefs = COEFFICIENTS[:]

    # Run with qualitative_ternary + task_tokens
    run_format_position(model, tasks, "qualitative_ternary", "task_tokens", pilot_path, pilot=True)

    records = load_jsonl(pilot_path)
    if not records:
        print("  No records — something went wrong")
        return

    total_scores = []
    for r in records:
        total_scores.extend([s for s in r["scores"] if s is not None])
    all_resp = []
    for r in records:
        all_resp.extend(r["scores"])
    parse_rate = sum(1 for s in all_resp if s is not None) / len(all_resp) if all_resp else 0
    print(f"  Pilot: {len(records)} records, parse_rate={parse_rate:.1%}")
    print(f"  Sample scores: {total_scores[:10]}")


# ─────────────────────────────────────────────────────────────────────────────
# Main run
# ─────────────────────────────────────────────────────────────────────────────


def run_main(fmt: str | None = None, position: str | None = None) -> None:
    """Run the full experiment for specified format(s) and position(s)."""
    tasks = load_selected_tasks()
    model = load_model()

    formats_to_run = [fmt] if fmt else FORMATS
    positions_to_run = [position] if position else POSITIONS

    for f in formats_to_run:
        for p in positions_to_run:
            output_path = RESULTS_DIR / f"results_{f}_{p}.jsonl"
            print(f"\n=== {f} × {p} ===")
            run_format_position(model, tasks, f, p, output_path)

    # Print summary
    print("\n=== Summary ===")
    for f in formats_to_run:
        for p in positions_to_run:
            output_path = RESULTS_DIR / f"results_{f}_{p}.jsonl"
            records = load_jsonl(output_path)
            if records:
                all_parse_rates = [r["parse_rate"] for r in records]
                print(f"  {f} × {p}: {len(records)} records, "
                      f"mean parse_rate={sum(all_parse_rates)/len(all_parse_rates):.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["select", "pilot", "run"])
    parser.add_argument("--format", choices=FORMATS, default=None,
                        help="Run only this format (default: all)")
    parser.add_argument("--position", choices=POSITIONS, default=None,
                        help="Run only this position (default: all)")
    args = parser.parse_args()

    if args.phase == "select":
        print("=== Task Selection ===")
        select_tasks()

    elif args.phase == "pilot":
        print("=== Loading tasks ===")
        tasks = load_selected_tasks()
        print(f"  {len(tasks)} tasks loaded")
        model = load_model()
        run_pilot(model)

    elif args.phase == "run":
        run_main(fmt=args.format, position=args.position)


if __name__ == "__main__":
    main()
