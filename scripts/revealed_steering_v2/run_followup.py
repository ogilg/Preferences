"""Revealed steering v2 follow-up — 500 pairs, 20-trial baseline, 15 multipliers.

Phase 1: Baseline recompute (coef=0, 20 trials per pair, 10 per ordering)
Phase 2: Steering sweep (15 multipliers, 10 trials per pair, 5 per ordering)

Reuses infrastructure from run_experiment.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from run_experiment import (
    load_hf_model, load_probe, compute_mean_norm,
    load_template, build_prompt_builder,
    parse_choice_local, append_checkpoint, load_checkpoint,
    MANIFEST_DIR, ACTIVATIONS_PATH, PROBE_ID,
    MULTIPLIERS, TEMPERATURE, MAX_NEW_TOKENS,
)

# ── Follow-up paths ─────────────────────────────────────────────────────────
FOLLOWUP_DIR = Path("experiments/revealed_steering_v2/followup")
OLD_PAIRS_PATH = Path("experiments/steering/replication/fine_grained/results/pairs.json")
NEW_PAIRS_PATH = FOLLOWUP_DIR / "pairs_200_new.json"
PAIRS_500_PATH = FOLLOWUP_DIR / "pairs_500.json"

# Override checkpoint path for follow-up
import run_experiment
CHECKPOINT_PATH = FOLLOWUP_DIR / "checkpoint.jsonl"
run_experiment.CHECKPOINT_PATH = CHECKPOINT_PATH

BASELINE_TRIALS_PER_ORDERING = 10  # 20 total per pair
STEERING_TRIALS_PER_ORDERING = 5   # 10 total per pair


def load_500_pairs():
    """Load 300 old + 200 new pairs as Task objects."""
    from src.task_data import Task, OriginDataset

    old_raw = json.loads(OLD_PAIRS_PATH.read_text())
    new_raw = json.loads(NEW_PAIRS_PATH.read_text())

    all_raw = old_raw + new_raw
    pairs = []
    for p in all_raw:
        task_a = Task(prompt=p["task_a_text"], origin=OriginDataset.ALPACA, id=p["task_a"], metadata={})
        task_b = Task(prompt=p["task_b_text"], origin=OriginDataset.ALPACA, id=p["task_b"], metadata={})
        pairs.append({
            "pair_id": p["pair_id"],
            "task_a": task_a,
            "task_b": task_b,
            "delta_mu": p["delta_mu"],
            "mu_a": p["mu_a"],
            "mu_b": p["mu_b"],
        })
    print(f"Loaded {len(pairs)} pairs (300 old + 200 new)")
    return pairs


def run_baseline(hf_model, layer, direction, pairs, builder):
    """Phase 1: Baseline measurement at coef=0, 20 trials per pair."""
    from src.steering.client import SteeredHFClient
    import torch
    from src.models.base import all_tokens_steering

    checkpoint = load_checkpoint()
    client = SteeredHFClient(hf_model, layer, direction, coefficient=0, steering_mode="differential")
    all_records = []
    fallback_count = 0

    total_calls = len(pairs) * 2
    call_count = 0
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"PHASE 1: BASELINE ({len(pairs)} pairs × 20 trials)")
    print(f"{'='*60}")

    block_records = []
    for pair_idx, pair in enumerate(pairs):
        for ordering in [0, 1]:
            ckpt_key = f"{pair['pair_id']}_0.0_baseline_{ordering}"
            existing = checkpoint.get(ckpt_key, [])
            n_existing = len(existing)
            n_needed = BASELINE_TRIALS_PER_ORDERING - n_existing
            call_count += 1

            if n_needed <= 0:
                continue

            if ordering == 0:
                prompt = builder.build(pair["task_a"], pair["task_b"])
            else:
                prompt = builder.build(pair["task_b"], pair["task_a"])

            task_prompts = [t.prompt for t in prompt.tasks]

            try:
                resps = client.generate_n(
                    prompt.messages, n=n_needed, temperature=TEMPERATURE, task_prompts=task_prompts
                )
            except ValueError:
                fallback_count += 1
                resps = hf_model.generate_n(prompt.messages, n=n_needed, temperature=TEMPERATURE)

            for sample_idx, resp in enumerate(resps):
                choice_presented = parse_choice_local(resp)
                if choice_presented is None:
                    choice_original = None
                elif ordering == 0:
                    choice_original = choice_presented
                else:
                    choice_original = "b" if choice_presented == "a" else "a"

                record = {
                    "pair_id": pair["pair_id"],
                    "task_a_id": pair["task_a"].id,
                    "task_b_id": pair["task_b"].id,
                    "coefficient": 0.0,
                    "multiplier": 0.0,
                    "condition": "baseline",
                    "sample_idx": n_existing + sample_idx,
                    "ordering": ordering,
                    "choice_original": choice_original,
                    "choice_presented": choice_presented,
                    "raw_response": resp[:500],
                    "delta_mu": pair["delta_mu"],
                    "steering_fallback": False,
                }
                block_records.append(record)
                all_records.append(record)

        if (pair_idx + 1) % 20 == 0:
            if block_records:
                append_checkpoint(block_records)
                block_records = []
            elapsed = time.time() - t_start
            rate = call_count / elapsed if elapsed > 0 else 0
            remaining = (total_calls - call_count) / rate if rate > 0 else 0
            print(f"  Pair {pair_idx+1}/{len(pairs)} | "
                  f"{elapsed/60:.0f}m elapsed, ~{remaining/60:.0f}m remaining")

    if block_records:
        append_checkpoint(block_records)

    if fallback_count > 0:
        print(f"  NOTE: {fallback_count} pair/ordering combos used all_tokens fallback")

    print(f"Phase 1 done — {len(all_records)} records")
    return all_records


def run_steering_sweep(hf_model, layer, direction, mean_norm, pairs, builder):
    """Phase 2: Full 15-multiplier sweep, 10 trials per pair."""
    from src.steering.client import SteeredHFClient
    from src.models.base import all_tokens_steering
    import torch

    checkpoint = load_checkpoint()
    coefficients = [mean_norm * m for m in MULTIPLIERS]
    all_records = []
    fallback_count = 0

    total_calls = len(pairs) * 2 * len(coefficients)
    call_count = 0
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"PHASE 2: STEERING SWEEP ({len(MULTIPLIERS)} multipliers × {len(pairs)} pairs)")
    print(f"{'='*60}")

    for coef_idx, (mult, coef) in enumerate(zip(MULTIPLIERS, coefficients)):
        print(f"\n=== probe [{coef_idx+1}/{len(MULTIPLIERS)}]: mult={mult}, coef={coef:.1f} ===")
        client = SteeredHFClient(hf_model, layer, direction, coefficient=coef, steering_mode="differential")
        block_records = []

        for pair_idx, pair in enumerate(pairs):
            for ordering in [0, 1]:
                ckpt_key = f"{pair['pair_id']}_{coef}_probe_{ordering}"
                existing = checkpoint.get(ckpt_key, [])
                n_existing = len(existing)
                n_needed = STEERING_TRIALS_PER_ORDERING - n_existing
                call_count += 1

                if n_needed <= 0:
                    continue

                if ordering == 0:
                    prompt = builder.build(pair["task_a"], pair["task_b"])
                else:
                    prompt = builder.build(pair["task_b"], pair["task_a"])

                task_prompts = [t.prompt for t in prompt.tasks]
                steering_fallback = False

                try:
                    resps = client.generate_n(
                        prompt.messages, n=n_needed, temperature=TEMPERATURE, task_prompts=task_prompts
                    )
                except ValueError:
                    steering_fallback = True
                    fallback_count += 1
                    if coef == 0:
                        resps = hf_model.generate_n(prompt.messages, n=n_needed, temperature=TEMPERATURE)
                    else:
                        scaled = direction * coef
                        steering_tensor = torch.tensor(scaled, dtype=torch.bfloat16, device=hf_model.device)
                        hook = all_tokens_steering(steering_tensor)
                        resps = hf_model.generate_with_steering_n(
                            prompt.messages, layer=client.layer, steering_hook=hook,
                            n=n_needed, temperature=TEMPERATURE,
                        )

                for sample_idx, resp in enumerate(resps):
                    choice_presented = parse_choice_local(resp)
                    if choice_presented is None:
                        choice_original = None
                    elif ordering == 0:
                        choice_original = choice_presented
                    else:
                        choice_original = "b" if choice_presented == "a" else "a"

                    record = {
                        "pair_id": pair["pair_id"],
                        "task_a_id": pair["task_a"].id,
                        "task_b_id": pair["task_b"].id,
                        "coefficient": coef,
                        "multiplier": mult,
                        "condition": "probe",
                        "sample_idx": n_existing + sample_idx,
                        "ordering": ordering,
                        "choice_original": choice_original,
                        "choice_presented": choice_presented,
                        "raw_response": resp[:500],
                        "delta_mu": pair["delta_mu"],
                        "steering_fallback": steering_fallback,
                    }
                    block_records.append(record)
                    all_records.append(record)

            if (pair_idx + 1) % 20 == 0:
                if block_records:
                    append_checkpoint(block_records)
                    block_records = []
                elapsed = time.time() - t_start
                rate = call_count / elapsed if elapsed > 0 else 0
                remaining = (total_calls - call_count) / rate if rate > 0 else 0
                print(f"  Pair {pair_idx+1}/{len(pairs)} | {call_count}/{total_calls} | "
                      f"{elapsed/60:.0f}m elapsed, ~{remaining/60:.0f}m remaining")

        if block_records:
            append_checkpoint(block_records)
            block_records = []

    if fallback_count > 0:
        print(f"  NOTE: {fallback_count} pair/ordering combos used all_tokens fallback")

    print(f"Phase 2 done — {len(all_records)} records")
    return all_records


def compile_baseline_into_pairs(pairs_raw: list[dict]) -> list[dict]:
    """Read checkpoint and add p_a_baseline_20 to each pair."""
    from collections import defaultdict

    records = []
    if CHECKPOINT_PATH.exists():
        for line in CHECKPOINT_PATH.read_text().strip().split("\n"):
            if line.strip():
                rec = json.loads(line)
                if rec["condition"] == "baseline":
                    records.append(rec)

    pair_choices = defaultdict(lambda: {"a": 0, "total": 0})
    for rec in records:
        if rec["choice_original"] is not None:
            pair_choices[rec["pair_id"]]["total"] += 1
            if rec["choice_original"] == "a":
                pair_choices[rec["pair_id"]]["a"] += 1

    for p in pairs_raw:
        pid = p["pair_id"]
        if pid in pair_choices and pair_choices[pid]["total"] > 0:
            p["p_a_baseline_20"] = pair_choices[pid]["a"] / pair_choices[pid]["total"]
            p["n_baseline_20"] = pair_choices[pid]["total"]

    return pairs_raw


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0, help="0=both, 1=baseline, 2=steering")
    args = parser.parse_args()

    FOLLOWUP_DIR.mkdir(parents=True, exist_ok=True)

    hf_model = load_hf_model()
    layer, direction = load_probe()
    mean_norm = compute_mean_norm(layer)
    print(f"Mean activation norm at L{layer}: {mean_norm:.2f}")

    pairs = load_500_pairs()
    template = load_template()
    builder = build_prompt_builder(template)

    t_total = time.time()

    if args.phase in [0, 1]:
        run_baseline(hf_model, layer, direction, pairs, builder)

        # Save pairs_500.json with baseline P(A)
        old_raw = json.loads(OLD_PAIRS_PATH.read_text())
        new_raw = json.loads(NEW_PAIRS_PATH.read_text())
        all_raw = old_raw + new_raw
        all_raw = compile_baseline_into_pairs(all_raw)
        PAIRS_500_PATH.write_text(json.dumps(all_raw, indent=2))
        print(f"Saved {PAIRS_500_PATH} with baseline P(A)")

    if args.phase in [0, 2]:
        run_steering_sweep(hf_model, layer, direction, mean_norm, pairs, builder)

    total_min = (time.time() - t_total) / 60
    print(f"\nAll phases complete in {total_min:.0f}m")

    # Count total records
    if CHECKPOINT_PATH.exists():
        n_records = sum(1 for line in CHECKPOINT_PATH.read_text().strip().split("\n") if line.strip())
        print(f"Total checkpoint records: {n_records}")


if __name__ == "__main__":
    main()
