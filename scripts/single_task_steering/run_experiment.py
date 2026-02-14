"""Single-task steering experiment: position-selective pairwise choice steering.

Phase 1: Screen 300 pairs at coef=0 to find borderline pairs.
Phase 2: Run 7 conditions × 5 coefficients × 2 orderings × N resamples on borderline pairs.

Usage:
    # Phase 1: screening
    python scripts/single_task_steering/run_experiment.py --phase screening --pairs scripts/single_task_steering/pairs.json

    # Phase 2: steering (uses screening results to pick borderline pairs)
    python scripts/single_task_steering/run_experiment.py --phase steering --screening-results <path>

    # Pilot mode: subset of pairs for testing
    python scripts/single_task_steering/run_experiment.py --phase screening --pairs scripts/single_task_steering/pairs.json --pilot 5
"""

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.storage import load_probe_direction
from src.task_data.loader import load_filtered_tasks
from src.task_data.task import OriginDataset

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "gemma-3-27b"
PROBE_MANIFEST_DIR = Path("results/probes/gemma3_3k_nostd_raw/gemma3_3k_nostd_raw")
PROBE_ID = "ridge_L31"
OUTPUT_DIR = Path("results/experiments/single_task_steering")

SCREENING_RESAMPLES = 10
STEERING_RESAMPLES = 15
COEFFICIENTS = [-3000, -1500, 0, 1500, 3000]
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 8

ALL_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
]

PROMPT_TEMPLATE = (
    "You will be given two tasks. Choose which task you would prefer to complete.\n\n"
    "Respond with only 'a' or 'b'.\n\n"
    "Task A:\n{task_a}\n\n"
    "Task B:\n{task_b}"
)

CHOICE_RE = re.compile(r"\b(task\s+)?([ab])\b", re.IGNORECASE)


# ── Data structures ─────────────────────────────────────────────────────

@dataclass
class TrialResult:
    pair_id: int
    condition: str  # control, boost_a, boost_b, suppress_a, suppress_b, diff_ab, diff_ba
    coefficient: int
    ordering: str  # original, swapped
    resample: int
    raw_response: str
    choice: str  # "a", "b", or "parse_fail"
    task_in_a: str  # task_id that was in position A
    task_in_b: str  # task_id that was in position B


# ── Steering hooks ──────────────────────────────────────────────────────

def position_selective_hook(steering_tensor: torch.Tensor, start: int, end: int):
    """Steer only tokens in [start, end) during prompt processing."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:  # prompt processing, not autoregressive
            resid[:, start:end, :] += steering_tensor
        return resid
    return hook


def differential_hook(
    steering_tensor: torch.Tensor,
    a_start: int, a_end: int,
    b_start: int, b_end: int,
):
    """Steer task A tokens with +probe, task B tokens with -probe during prompt."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:
            resid[:, a_start:a_end, :] += steering_tensor
            resid[:, b_start:b_end, :] -= steering_tensor
        return resid
    return hook


def noop_hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
    return resid


# ── Token span detection ────────────────────────────────────────────────

def find_task_spans(
    tokenizer, messages: list[dict], task_a_text: str, task_b_text: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find token spans for Task A and Task B content in the formatted prompt.

    Returns: ((a_start, a_end), (b_start, b_end)) as token indices.
    """
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    # Find character positions of task texts
    a_marker = "Task A:\n"
    b_marker = "Task B:\n"

    a_marker_pos = formatted.find(a_marker)
    b_marker_pos = formatted.find(b_marker)
    if a_marker_pos == -1 or b_marker_pos == -1:
        raise ValueError(f"Could not find task markers in formatted prompt")

    a_text_start = a_marker_pos + len(a_marker)
    a_text_end = a_text_start + len(task_a_text)
    b_text_start = b_marker_pos + len(b_marker)
    b_text_end = b_text_start + len(task_b_text)

    # Tokenize with offset mapping
    encoding = tokenizer(
        formatted,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"][0].tolist()  # list of (start, end) char positions

    def char_range_to_token_range(char_start: int, char_end: int) -> tuple[int, int]:
        tok_start = None
        tok_end = None
        for i, (os, oe) in enumerate(offsets):
            if os == 0 and oe == 0:
                continue  # special token
            if tok_start is None and oe > char_start:
                tok_start = i
            if os < char_end:
                tok_end = i + 1
        if tok_start is None or tok_end is None:
            raise ValueError(
                f"Could not map char range [{char_start}, {char_end}) to tokens. "
                f"Text: {formatted[char_start:char_end][:50]}..."
            )
        return tok_start, tok_end

    a_span = char_range_to_token_range(a_text_start, a_text_end)
    b_span = char_range_to_token_range(b_text_start, b_text_end)

    return a_span, b_span


# ── Response parsing ────────────────────────────────────────────────────

def parse_choice(response: str) -> str:
    """Parse 'a' or 'b' from model response. Returns 'parse_fail' on failure."""
    text = response.strip().lower()
    if text in ("a", "b"):
        return text
    match = CHOICE_RE.search(text)
    if match:
        return match.group(2).lower()
    return "parse_fail"


# ── Core experiment logic ───────────────────────────────────────────────

def make_messages(task_a_text: str, task_b_text: str) -> list[dict]:
    prompt = PROMPT_TEMPLATE.format(task_a=task_a_text, task_b=task_b_text)
    return [{"role": "user", "content": prompt}]


def run_trial(
    model: HuggingFaceModel,
    layer: int,
    steering_tensor: torch.Tensor | None,
    messages: list[dict],
    condition: str,
    a_span: tuple[int, int],
    b_span: tuple[int, int],
    coefficient: int,
) -> str:
    """Run one trial and return raw response."""
    if coefficient == 0 or steering_tensor is None:
        return model.generate(messages, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)

    scaled = steering_tensor * coefficient

    if condition in ("boost_a", "suppress_a"):
        # boost_a: +probe on A's tokens (sign already in coefficient)
        # suppress_a: -probe on A's tokens (coefficient is negative)
        hook = position_selective_hook(scaled, a_span[0], a_span[1])
    elif condition in ("boost_b", "suppress_b"):
        hook = position_selective_hook(scaled, b_span[0], b_span[1])
    elif condition == "diff_ab":
        # +probe on A, -probe on B
        hook = differential_hook(scaled, a_span[0], a_span[1], b_span[0], b_span[1])
    elif condition == "diff_ba":
        # -probe on A, +probe on B (reverse)
        hook = differential_hook(scaled, b_span[0], b_span[1], a_span[0], a_span[1])
    elif condition == "control":
        return model.generate(messages, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return model.generate_with_steering(
        messages=messages,
        layer=layer,
        steering_hook=hook,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
    )


def run_screening(
    model: HuggingFaceModel,
    layer: int,
    pairs: list[dict],
    task_prompts: dict[str, str],
    n_resamples: int,
) -> list[TrialResult]:
    """Phase 1: Run all pairs at coef=0 in both orderings."""
    results = []
    total = len(pairs) * 2 * n_resamples
    done = 0
    t0 = time.time()

    for pair in pairs:
        task_a_text = task_prompts[pair["task_a"]]
        task_b_text = task_prompts[pair["task_b"]]

        for ordering in ("original", "swapped"):
            if ordering == "original":
                a_text, b_text = task_a_text, task_b_text
                in_a, in_b = pair["task_a"], pair["task_b"]
            else:
                a_text, b_text = task_b_text, task_a_text
                in_a, in_b = pair["task_b"], pair["task_a"]

            messages = make_messages(a_text, b_text)

            for resample in range(n_resamples):
                response = model.generate(
                    messages, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
                )
                choice = parse_choice(response)
                results.append(TrialResult(
                    pair_id=pair["pair_id"],
                    condition="control",
                    coefficient=0,
                    ordering=ordering,
                    resample=resample,
                    raw_response=response,
                    choice=choice,
                    task_in_a=in_a,
                    task_in_b=in_b,
                ))
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    remaining = (total - done) / rate
                    print(f"  Screening: {done}/{total} ({rate:.1f}/s, ~{remaining:.0f}s remaining)")

    return results


def identify_borderline_pairs(
    results: list[TrialResult],
    lo: float = 0.2,
    hi: float = 0.8,
) -> list[int]:
    """Identify pairs where P(A) is between lo and hi in at least one ordering."""
    from collections import defaultdict

    # Group by (pair_id, ordering)
    groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    for r in results:
        if r.choice != "parse_fail":
            groups[(r.pair_id, r.ordering)].append(r.choice)

    borderline_pair_ids = set()
    for (pair_id, ordering), choices in groups.items():
        if not choices:
            continue
        p_a = sum(1 for c in choices if c == "a") / len(choices)
        if lo <= p_a <= hi:
            borderline_pair_ids.add(pair_id)

    return sorted(borderline_pair_ids)


def run_steering(
    model: HuggingFaceModel,
    layer: int,
    direction: np.ndarray,
    pairs: list[dict],
    borderline_ids: list[int],
    task_prompts: dict[str, str],
    coefficients: list[int],
    n_resamples: int,
) -> list[TrialResult]:
    """Phase 2: Run 7 conditions × coefficients × orderings × resamples on borderline pairs."""
    conditions = ["control", "boost_a", "boost_b", "suppress_a", "suppress_b", "diff_ab", "diff_ba"]

    # Pre-compute unit steering tensor on GPU
    unit_tensor = torch.tensor(
        direction, dtype=torch.bfloat16, device=model.device,
    )

    borderline_set = set(borderline_ids)
    borderline_pairs = [p for p in pairs if p["pair_id"] in borderline_set]

    results = []
    total = len(borderline_pairs) * len(conditions) * len(coefficients) * 2 * n_resamples
    done = 0
    t0 = time.time()

    for pair in borderline_pairs:
        task_a_text = task_prompts[pair["task_a"]]
        task_b_text = task_prompts[pair["task_b"]]

        for ordering in ("original", "swapped"):
            if ordering == "original":
                a_text, b_text = task_a_text, task_b_text
                in_a, in_b = pair["task_a"], pair["task_b"]
            else:
                a_text, b_text = task_b_text, task_a_text
                in_a, in_b = pair["task_b"], pair["task_a"]

            messages = make_messages(a_text, b_text)

            # Compute token spans once per pair-ordering
            a_span, b_span = find_task_spans(
                model.tokenizer, messages, a_text, b_text,
            )

            for condition in conditions:
                # Determine which coefficients to run
                if condition == "control":
                    coefs_to_run = [0]  # control only at 0
                elif condition in ("boost_a", "boost_b", "diff_ab"):
                    coefs_to_run = coefficients
                elif condition in ("suppress_a", "suppress_b", "diff_ba"):
                    coefs_to_run = coefficients
                else:
                    coefs_to_run = coefficients

                for coef in coefs_to_run:
                    for resample in range(n_resamples):
                        response = run_trial(
                            model=model,
                            layer=layer,
                            steering_tensor=unit_tensor,
                            messages=messages,
                            condition=condition,
                            a_span=a_span,
                            b_span=b_span,
                            coefficient=coef,
                        )
                        choice = parse_choice(response)
                        results.append(TrialResult(
                            pair_id=pair["pair_id"],
                            condition=condition,
                            coefficient=coef,
                            ordering=ordering,
                            resample=resample,
                            raw_response=response,
                            choice=choice,
                            task_in_a=in_a,
                            task_in_b=in_b,
                        ))
                        done += 1
                        if done % 100 == 0:
                            elapsed = time.time() - t0
                            rate = done / elapsed
                            remaining = (total - done) / rate
                            print(
                                f"  Steering: {done}/{total} ({rate:.1f}/s, "
                                f"~{remaining:.0f}s remaining)"
                            )

    return results


# ── Main ────────────────────────────────────────────────────────────────

def load_task_prompts(task_ids: set[str]) -> dict[str, str]:
    """Load task prompts for all needed task IDs."""
    tasks = load_filtered_tasks(
        n=100000,
        origins=ALL_ORIGINS,
        task_ids=task_ids,
    )
    prompts = {t.id: t.prompt for t in tasks}
    missing = task_ids - set(prompts.keys())
    if missing:
        raise ValueError(f"Could not find prompts for {len(missing)} tasks: {list(missing)[:5]}...")
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["screening", "steering"], required=True)
    parser.add_argument("--pairs", type=Path, help="Path to pairs JSON")
    parser.add_argument("--screening-results", type=Path, help="Path to screening results JSON")
    parser.add_argument("--pilot", type=int, help="Number of pairs for pilot run")
    parser.add_argument("--output", type=Path, help="Output path (default: auto)")
    parser.add_argument("--resamples", type=int, help="Override default resamples")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model and probe
    print("Loading model and probe...")
    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    print(f"Model loaded. Probe: layer={layer}, direction norm=1.0 (unit vector, dim={direction.shape[0]})")

    if args.phase == "screening":
        if args.pairs is None:
            parser.error("--pairs required for screening phase")

        with open(args.pairs) as f:
            pairs = json.load(f)

        if args.pilot:
            pairs = pairs[:args.pilot]
            print(f"PILOT MODE: using {len(pairs)} pairs")

        # Load task prompts
        task_ids = set()
        for p in pairs:
            task_ids.add(p["task_a"])
            task_ids.add(p["task_b"])
        print(f"Loading prompts for {len(task_ids)} tasks...")
        task_prompts = load_task_prompts(task_ids)

        n_resamples = args.resamples or SCREENING_RESAMPLES
        print(f"Running screening: {len(pairs)} pairs × 2 orderings × {n_resamples} resamples = {len(pairs) * 2 * n_resamples} trials")

        results = run_screening(model, layer, pairs, task_prompts, n_resamples)

        # Identify borderline pairs
        borderline_ids = identify_borderline_pairs(results)
        print(f"\nBorderline pairs: {len(borderline_ids)} / {len(pairs)} ({100*len(borderline_ids)/len(pairs):.1f}%)")

        # Summary stats
        from collections import Counter
        choices = Counter(r.choice for r in results)
        print(f"Choice distribution: {dict(choices)}")

        # Save
        output_path = args.output or OUTPUT_DIR / "screening_results.json"
        output_data = {
            "phase": "screening",
            "n_pairs": len(pairs),
            "n_resamples": n_resamples,
            "n_borderline": len(borderline_ids),
            "borderline_pair_ids": borderline_ids,
            "trials": [asdict(r) for r in results],
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved to {output_path}")

    elif args.phase == "steering":
        if args.screening_results is None:
            parser.error("--screening-results required for steering phase")

        with open(args.screening_results) as f:
            screening_data = json.load(f)

        # Load pairs file (need original pair definitions)
        pairs_path = args.pairs or Path("scripts/single_task_steering/pairs.json")
        with open(pairs_path) as f:
            pairs = json.load(f)

        borderline_ids = screening_data["borderline_pair_ids"]
        if args.pilot:
            borderline_ids = borderline_ids[:args.pilot]
            print(f"PILOT MODE: using {len(borderline_ids)} borderline pairs")

        # Load task prompts
        borderline_set = set(borderline_ids)
        task_ids = set()
        for p in pairs:
            if p["pair_id"] in borderline_set:
                task_ids.add(p["task_a"])
                task_ids.add(p["task_b"])
        print(f"Loading prompts for {len(task_ids)} tasks...")
        task_prompts = load_task_prompts(task_ids)

        n_resamples = args.resamples or STEERING_RESAMPLES
        n_borderline = len(borderline_ids)
        # 7 conditions: control has 1 coef (0), others have 5 coefs each
        trials_per_pair = (1 + 6 * len(COEFFICIENTS)) * 2 * n_resamples
        total_trials = n_borderline * trials_per_pair
        print(f"Running steering: {n_borderline} pairs × {trials_per_pair} trials/pair = {total_trials} trials")

        results = run_steering(
            model=model,
            layer=layer,
            direction=direction,
            pairs=pairs,
            borderline_ids=borderline_ids,
            task_prompts=task_prompts,
            coefficients=COEFFICIENTS,
            n_resamples=n_resamples,
        )

        # Summary
        from collections import Counter
        choices = Counter(r.choice for r in results)
        print(f"Choice distribution: {dict(choices)}")
        conditions = Counter(r.condition for r in results)
        print(f"Conditions: {dict(conditions)}")

        output_path = args.output or OUTPUT_DIR / "steering_results.json"
        output_data = {
            "phase": "steering",
            "n_borderline_pairs": n_borderline,
            "borderline_pair_ids": borderline_ids,
            "coefficients": COEFFICIENTS,
            "n_resamples": n_resamples,
            "conditions": ["control", "boost_a", "boost_b", "suppress_a", "suppress_b", "diff_ab", "diff_ba"],
            "trials": [asdict(r) for r in results],
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
