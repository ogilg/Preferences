"""EOT direction steering experiment.

Runs differential steering using the EOT probe direction on 500 pairs
at multipliers +/-0.03, collecting 10 trials per pair per multiplier
(5 per ordering). Checkpoints to JSONL for resumability.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np

from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.storage import load_probe_direction
from src.steering.client import SteeredHFClient
from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.measurement.elicitation.semantic_parser import parse_completion_choice_async, ParseError
from src.task_data import Task, OriginDataset

# --- Config ---
PAIRS_PATH = Path("experiments/revealed_steering_v2/followup/pairs_500.json")
EOT_PROBE_DIR = Path("results/probes/heldout_eval_gemma3_eot")
PROBE_ID = "ridge_L31"
MEAN_NORM = 52823.0
MULTIPLIERS = [0.03, -0.03]
N_PER_ORDERING = 5  # 5 trials per ordering = 10 total per pair/multiplier
CHECKPOINT_PATH = Path("experiments/steering/eot_direction/checkpoint.jsonl")
TEMPLATE_PATH = Path("src/measurement/elicitation/prompt_templates/data/completion_preference.yaml")


def load_pairs() -> list[dict]:
    with open(PAIRS_PATH) as f:
        return json.load(f)


def make_task(task_id: str, task_text: str) -> Task:
    if task_id.startswith("alpaca"):
        origin = OriginDataset.ALPACA
    elif task_id.startswith("competition_math") or task_id.startswith("math"):
        origin = OriginDataset.MATH
    elif task_id.startswith("wildchat"):
        origin = OriginDataset.WILDCHAT
    elif task_id.startswith("bailbench"):
        origin = OriginDataset.BAILBENCH
    else:
        origin = OriginDataset.WILDCHAT
    return Task(prompt=task_text, origin=origin, id=task_id, metadata={})


def build_prompt(template_str: str, task_a_text: str, task_b_text: str) -> str:
    fmt = CompletionChoiceFormat(task_a_label="Task A", task_b_label="Task B")
    return template_str.format(
        format_instruction=fmt.format_instruction(),
        task_a=task_a_text,
        task_b=task_b_text,
    )


def parse_prefix(response: str) -> str | None:
    """Try prefix match: 'Task A:' or 'Task B:'. Returns 'a', 'b', or None."""
    cleaned = re.sub(r"^[\s*#_`>]+", "", response).lower()
    if cleaned.startswith("task a"):
        return "a"
    if cleaned.startswith("task b"):
        return "b"
    return None


async def parse_with_fallback(
    response: str, task_a_text: str, task_b_text: str
) -> tuple[str, bool]:
    """Parse response. Returns (choice, used_fallback).
    choice is 'a', 'b', 'refusal', or 'parse_fail'.
    """
    prefix_result = parse_prefix(response)
    if prefix_result is not None:
        return prefix_result, False
    try:
        result = await parse_completion_choice_async(response, task_a_text, task_b_text)
        return result, True
    except ParseError:
        return "parse_fail", True


def load_checkpoint() -> list[dict]:
    if not CHECKPOINT_PATH.exists():
        return []
    records = []
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_completed_keys(records: list[dict]) -> set[tuple[str, float, int]]:
    """Build set of (pair_id, multiplier, ordering) keys that have N_PER_ORDERING records."""
    from collections import Counter
    counts = Counter(
        (r["pair_id"], r["multiplier"], r["ordering"]) for r in records
    )
    return {k for k, v in counts.items() if v >= N_PER_ORDERING}


def append_record(record: dict):
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


async def run_experiment(pilot_n: int | None = None, resume: bool = False):
    pairs = load_pairs()
    if pilot_n is not None:
        pairs = pairs[:pilot_n]
        print(f"PILOT MODE: using {pilot_n} pairs")

    # Load template
    templates = load_templates_from_yaml(TEMPLATE_PATH)
    template = templates[0]

    # Load probe
    layer, direction = load_probe_direction(EOT_PROBE_DIR, PROBE_ID)
    print(f"Loaded EOT probe: layer={layer}, direction shape={direction.shape}")

    # Compute coefficients
    coefficients = {mult: MEAN_NORM * mult for mult in MULTIPLIERS}
    print(f"Coefficients: {coefficients}")

    # Resume logic
    completed_keys = set()
    if resume and CHECKPOINT_PATH.exists():
        existing = load_checkpoint()
        completed_keys = get_completed_keys(existing)
        print(f"Resuming: {len(completed_keys)} (pair, mult, ordering) combos already complete")

    # Load model
    print("Loading model...", flush=True)
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=256)
    client = SteeredHFClient(
        model, layer, direction, coefficient=0.0,
        steering_mode="differential",
        a_marker="Task A:", b_marker="Task B:",
    )
    print("Model loaded.", flush=True)

    total_pairs = len(pairs)
    total_generations = total_pairs * len(MULTIPLIERS) * 2 * N_PER_ORDERING
    done_count = 0
    fallback_count = 0
    parse_fail_count = 0
    start_time = time.time()

    for pi, pair in enumerate(pairs):
        pair_id = pair["pair_id"]
        task_a = make_task(pair["task_a"], pair["task_a_text"])
        task_b = make_task(pair["task_b"], pair["task_b_text"])
        delta_mu = pair["delta_mu"]

        for mult in MULTIPLIERS:
            coef = coefficients[mult]
            steered = client.with_coefficient(coef)

            for ordering in [0, 1]:
                key = (pair_id, mult, ordering)
                if key in completed_keys:
                    done_count += N_PER_ORDERING
                    continue

                # ordering=0: A first (AB), ordering=1: B first (BA)
                if ordering == 0:
                    presented_a, presented_b = task_a, task_b
                else:
                    presented_a, presented_b = task_b, task_a

                prompt_text = build_prompt(
                    template.template,
                    presented_a.prompt,
                    presented_b.prompt,
                )
                messages = [{"role": "user", "content": prompt_text}]
                task_prompts = [presented_a.prompt, presented_b.prompt]

                try:
                    responses = steered.generate_n(
                        messages, n=N_PER_ORDERING, temperature=1.0,
                        task_prompts=task_prompts,
                    )
                    steering_fallback = False
                except ValueError:
                    # Span detection failed, fall back to all_tokens
                    fallback_client = SteeredHFClient(
                        model, layer, direction, coef,
                        steering_mode="all_tokens",
                    )
                    responses = fallback_client.generate_n(
                        messages, n=N_PER_ORDERING, temperature=1.0,
                    )
                    steering_fallback = True

                for si, resp in enumerate(responses):
                    choice_presented, used_semantic = await parse_with_fallback(
                        resp, presented_a.prompt, presented_b.prompt
                    )
                    if used_semantic:
                        fallback_count += 1
                    if choice_presented == "parse_fail":
                        parse_fail_count += 1

                    # Map back to original ordering
                    if ordering == 0:
                        choice_original = choice_presented
                    else:
                        if choice_presented == "a":
                            choice_original = "b"
                        elif choice_presented == "b":
                            choice_original = "a"
                        else:
                            choice_original = choice_presented

                    record = {
                        "pair_id": pair_id,
                        "task_a_id": task_a.id,
                        "task_b_id": task_b.id,
                        "coefficient": coef,
                        "multiplier": mult,
                        "condition": "eot",
                        "sample_idx": si,
                        "ordering": ordering,
                        "choice_original": choice_original,
                        "choice_presented": choice_presented,
                        "raw_response": resp[:500],
                        "delta_mu": delta_mu,
                        "steering_fallback": steering_fallback,
                    }
                    append_record(record)
                    done_count += 1

        if (pi + 1) % 10 == 0 or pi == 0:
            elapsed = time.time() - start_time
            rate = done_count / elapsed if elapsed > 0 else 0
            eta = (total_generations - done_count) / rate if rate > 0 else 0
            print(
                f"Pair {pi+1}/{total_pairs} | "
                f"Generations: {done_count}/{total_generations} | "
                f"Rate: {rate:.1f}/s | "
                f"ETA: {eta/60:.0f}min | "
                f"Semantic fallbacks: {fallback_count} | "
                f"Parse fails: {parse_fail_count}",
                flush=True,
            )

    elapsed = time.time() - start_time
    print(f"\nDone! {done_count} generations in {elapsed/60:.1f} minutes")
    print(f"Semantic fallbacks: {fallback_count}, Parse fails: {parse_fail_count}")
    print(f"Steering fallbacks (span detection): {sum(1 for r in load_checkpoint() if r.get('steering_fallback', False))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", type=int, default=None, help="Number of pairs for pilot run")
    args = parser.parse_args()
    asyncio.run(run_experiment(pilot_n=args.pilot, resume=args.resume))


if __name__ == "__main__":
    main()
