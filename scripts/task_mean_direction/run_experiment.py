"""Task-mean direction steering experiment.

Tests whether the task_mean probe direction produces effective preference
steering when applied to task-token positions via differential steering.
"""

import asyncio
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.activations import compute_activation_norms
from src.probes.core.storage import load_probe_direction
from src.steering.client import SteeredHFClient
from src.measurement.elicitation.prompt_templates import (
    TEMPLATES_DATA_DIR,
    load_templates_from_yaml,
)
from src.measurement.runners.runners import build_revealed_builder
from src.task_data import OriginDataset, Task

# --- Config ---
MANIFEST_DIR = Path("results/probes/heldout_eval_gemma3_task_mean")
ACTIVATIONS_PATH = Path(
    "activations/gemma_3_27b_turn_boundary_sweep/activations_task_mean.npz"
)
PAIRS_PATH = Path("experiments/revealed_steering_v2/followup/pairs_500.json")
CHECKPOINT_PATH = Path("experiments/steering/task_mean_direction/checkpoint.jsonl")

LAYERS = [25, 32]
MULTIPLIERS = [-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05]
N_PER_ORDERING = 5
CONDITION = "task_mean"


def load_pairs(limit: int = 0) -> list[dict]:
    with open(PAIRS_PATH) as f:
        raw = json.load(f)
    if limit > 0:
        raw = raw[:limit]
    pairs = []
    for p in raw:
        pairs.append(
            {
                "pair_id": p["pair_id"],
                "task_a": Task(
                    prompt=p["task_a_text"],
                    origin=OriginDataset.WILDCHAT,
                    id=p["task_a"],
                    metadata={},
                ),
                "task_b": Task(
                    prompt=p["task_b_text"],
                    origin=OriginDataset.WILDCHAT,
                    id=p["task_b"],
                    metadata={},
                ),
                "delta_mu": p["delta_mu"],
            }
        )
    return pairs


def load_done_keys() -> set[tuple]:
    """Load checkpoint, return set of (pair_id, layer, multiplier, ordering) keys with >= 5 records."""
    counts: Counter[tuple] = Counter()
    if not CHECKPOINT_PATH.exists():
        return set()
    for line in CHECKPOINT_PATH.read_text().strip().split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)
        key = (rec["pair_id"], rec["layer"], rec["multiplier"], rec["ordering"])
        counts[key] += 1
    return {k for k, v in counts.items() if v >= N_PER_ORDERING}


def append_checkpoint(records: list[dict]) -> None:
    with open(CHECKPOINT_PATH, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def prefix_parse(response: str) -> str | None:
    """Fast prefix match: returns 'a', 'b', or None."""
    cleaned = re.sub(r"^[\s*#_`>]+", "", response).lower()
    if cleaned.startswith("task a"):
        return "a"
    if cleaned.startswith("task b"):
        return "b"
    return None


async def semantic_parse_batch(
    items: list[tuple[str, str, str]],
) -> list[str]:
    """Semantic parse a batch of (response, task_a_prompt, task_b_prompt) tuples.

    Returns list of 'a', 'b', 'refusal', or 'parse_fail'.
    """
    from src.measurement.elicitation.semantic_parser import (
        parse_completion_choice_async,
    )

    async def _parse_one(response: str, ta: str, tb: str) -> str:
        try:
            return await parse_completion_choice_async(response, ta, tb)
        except Exception:
            return "parse_fail"

    tasks = [_parse_one(r, ta, tb) for r, ta, tb in items]
    return await asyncio.gather(*tasks)


def flip_choice(choice: str) -> str:
    if choice == "a":
        return "b"
    if choice == "b":
        return "a"
    return choice


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", type=int, default=0, help="Pilot with N pairs")
    args = parser.parse_args()

    pairs = load_pairs(limit=args.pilot)
    done_keys = load_done_keys() if args.resume else set()

    # Probe directions and norms
    print("Loading probe directions and activation norms...")
    norms = compute_activation_norms(ACTIVATIONS_PATH, layers=LAYERS)
    probe_directions: dict[int, np.ndarray] = {}
    for layer in LAYERS:
        _, direction = load_probe_direction(MANIFEST_DIR, f"ridge_L{layer}")
        probe_directions[layer] = direction
        print(f"  L{layer}: mean_norm={norms[layer]:.0f}")

    # Template and prompt builder
    templates = load_templates_from_yaml(
        TEMPLATES_DATA_DIR / "completion_preference.yaml"
    )
    template = next(t for t in templates if t.name == "completion_preference")
    builder = build_revealed_builder(template, response_format_name="completion")

    # Model
    print("Loading model...")
    hf_model = HuggingFaceModel("gemma-3-27b", max_new_tokens=256)

    # Stats
    total_gen = 0
    total_skip = 0
    n_prefix_parsed = 0
    n_semantic_parsed = 0
    n_fallback = 0
    t_start = time.time()

    # Count total work
    total_conditions = len(LAYERS) * len(MULTIPLIERS) * len(pairs) * 2
    done_conditions = 0

    for layer in LAYERS:
        direction = probe_directions[layer]
        mean_norm = norms[layer]

        base_client = SteeredHFClient(
            hf_model=hf_model,
            layer=layer,
            steering_direction=direction,
            coefficient=0.0,
            steering_mode="differential",
        )

        for mult in MULTIPLIERS:
            coef = mean_norm * mult

            for pair in pairs:
                for ordering in [0, 1]:
                    done_conditions += 1
                    key = (pair["pair_id"], layer, mult, ordering)
                    if key in done_keys:
                        total_skip += N_PER_ORDERING
                        continue

                    # Present tasks in correct ordering
                    if ordering == 0:
                        presented_a = pair["task_a"]
                        presented_b = pair["task_b"]
                    else:
                        presented_a = pair["task_b"]
                        presented_b = pair["task_a"]

                    prompt_obj = builder.build(presented_a, presented_b)
                    task_prompts = [presented_a.prompt, presented_b.prompt]

                    # Generate
                    client = base_client.with_coefficient(coef)
                    steering_fallback = False
                    try:
                        responses = client.generate_n(
                            prompt_obj.messages,
                            n=N_PER_ORDERING,
                            temperature=1.0,
                            task_prompts=task_prompts,
                        )
                    except Exception as e:
                        print(
                            f"  WARNING: differential steering failed for "
                            f"{pair['pair_id']} L{layer} m={mult} o={ordering}: {e}"
                        )
                        fb_client = SteeredHFClient(
                            hf_model=hf_model,
                            layer=layer,
                            steering_direction=direction,
                            coefficient=coef,
                            steering_mode="all_tokens",
                        )
                        responses = fb_client.generate_n(
                            prompt_obj.messages,
                            n=N_PER_ORDERING,
                            temperature=1.0,
                        )
                        steering_fallback = True
                        n_fallback += 1

                    # Parse responses
                    records = []
                    need_semantic: list[tuple[int, dict, str, str, str]] = []

                    for idx, resp in enumerate(responses):
                        choice_presented = prefix_parse(resp)
                        rec = {
                            "pair_id": pair["pair_id"],
                            "task_a_id": pair["task_a"].id,
                            "task_b_id": pair["task_b"].id,
                            "coefficient": round(coef, 1),
                            "multiplier": mult,
                            "layer": layer,
                            "condition": CONDITION,
                            "sample_idx": idx,
                            "ordering": ordering,
                            "choice_original": None,
                            "choice_presented": choice_presented,
                            "raw_response": resp,
                            "delta_mu": pair["delta_mu"],
                            "steering_fallback": steering_fallback,
                        }

                        if choice_presented is not None:
                            n_prefix_parsed += 1
                            rec["choice_original"] = (
                                choice_presented
                                if ordering == 0
                                else flip_choice(choice_presented)
                            )
                            records.append(rec)
                        else:
                            need_semantic.append(
                                (
                                    idx,
                                    rec,
                                    resp,
                                    presented_a.prompt,
                                    presented_b.prompt,
                                )
                            )

                    # Semantic parsing fallback
                    if need_semantic:
                        sem_inputs = [
                            (resp, ta, tb) for _, _, resp, ta, tb in need_semantic
                        ]
                        sem_results = asyncio.run(semantic_parse_batch(sem_inputs))
                        for (_, rec, _, _, _), result in zip(
                            need_semantic, sem_results
                        ):
                            n_semantic_parsed += 1
                            rec["choice_presented"] = result
                            if result in ("a", "b"):
                                rec["choice_original"] = (
                                    result
                                    if ordering == 0
                                    else flip_choice(result)
                                )
                            else:
                                rec["choice_original"] = result
                            records.append(rec)

                    append_checkpoint(records)
                    total_gen += len(records)

                    elapsed = time.time() - t_start
                    rate = total_gen / elapsed if elapsed > 0 else 0
                    pct = done_conditions / total_conditions * 100
                    if done_conditions % 20 == 0 or done_conditions <= 5:
                        print(
                            f"  L{layer} m={mult:+.2f} {pair['pair_id']} o={ordering} | "
                            f"gen={total_gen} skip={total_skip} | "
                            f"{rate:.1f}/s | {pct:.1f}% | "
                            f"prefix={n_prefix_parsed} sem={n_semantic_parsed} fb={n_fallback}"
                        )

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed / 3600:.1f}h")
    print(f"Generated: {total_gen}, Skipped: {total_skip}")
    print(f"Prefix parsed: {n_prefix_parsed}, Semantic parsed: {n_semantic_parsed}")
    print(f"Steering fallbacks: {n_fallback}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
