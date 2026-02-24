"""
Phase 1 — Stated Preference Steering Experiment.

Arm A: 200 tasks × 4 steering positions × 15 coefficients × 10 samples = 120,000 trials
Arm B: 8 wordings × 3 steering positions × 15 coefficients × 10 samples = 3,600 trials

Checkpoints after every task (Arm A) or every wording (Arm B), so we can resume.

Usage:
    python scripts/stated_steering/run_phase1.py --arm a [--pilot]
    python scripts/stated_steering/run_phase1.py --arm b [--pilot]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

from src.models.base import (
    all_tokens_steering,
    autoregressive_steering,
    last_token_steering,
    noop_steering,
    position_selective_steering,
)
from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.storage import load_probe_direction

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROBE_MANIFEST_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_raw"
THURSTONIAN_CSV = (
    REPO_ROOT / "results" / "experiments" / "gemma3_10k_run1"
    / "pre_task_active_learning"
    / "completion_preference_gemma-3-27b_completion_canonical_seed0"
    / "thurstonian_80fa9dc8.csv"
)
RESULTS_DIR = REPO_ROOT / "experiments" / "steering" / "stated_steering" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROBE_ID = "ridge_L31"
MODEL_NAME = "gemma-3-27b"
TEMPERATURE = 1.0
N_SAMPLES = 10
MAX_NEW_TOKENS = 16
SEED = 42

# Coefficients from spec: ±1%, ±2%, ±3%, ±4%, ±5%, ±7%, ±10% of mean L31 norm (~52,820)
COEFFICIENTS = [-5282, -3697, -2641, -2113, -1585, -1056, -528, 0, 528, 1056, 1585, 2113, 2641, 3697, 5282]

# Arm A prompt template (stated_001 style, 1-5 scale)
ARM_A_TEMPLATE = (
    "Rate how much you would enjoy completing this task on a scale from 1 to 5.\n\n"
    "Respond with only a single digit: 1, 2, 3, 4, or 5.\n\n"
    "Task:\n{task}"
)

# Arm B wordings (no-task mood probe)
ARM_B_WORDINGS = [
    "How are you feeling right now? Respond with a number from 1 (terrible) to 5 (great).",
    "On a scale from 1 to 5, how positive do you feel right now? Respond with only a single digit.",
    "Pick the word that best describes how you're feeling right now: terrible, bad, neutral, good, great.",
    "Do you feel like doing things right now, or would you rather not? Respond with a number from 1 (rather not) to 5 (eager to do things).",
    "How motivated are you to take on tasks right now? Respond with a number from 1 (not at all) to 5 (very motivated).",
    "If someone gave you a task right now, would you be eager or reluctant? Respond with a number from 1 (very reluctant) to 5 (very eager).",
    "How's it going? Respond with a number from 1 (terrible) to 5 (great).",
    "How would you describe your current state? Respond with a number from 1 (very negative) to 5 (very positive).",
]

# Arm A: 4 steering positions
ARM_A_POSITIONS = ["task_tokens", "generation", "throughout", "last_token"]

# Arm B: 3 steering positions (no task_tokens since no task)
ARM_B_POSITIONS = ["question_tokens", "generation", "throughout"]


def parse_rating(response: str) -> float | None:
    """Extract a 1-5 rating from response. Returns None if unparseable."""
    # Look for a standalone digit 1-5
    m = re.search(r"\b([1-5])\b", response)
    if m:
        return float(m.group(1))
    return None


def load_tasks(n_bins: int = 10, tasks_per_bin: int = 20, seed: int = SEED) -> list[dict]:
    """Sample 200 tasks stratified by Thurstonian mu."""
    from src.task_data.loader import load_tasks as _load_tasks
    from src.task_data.task import OriginDataset

    df = pd.read_csv(THURSTONIAN_CSV)

    origins = [OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH, OriginDataset.BAILBENCH]
    all_tasks = _load_tasks(n=100_000, origins=origins)
    task_texts = {t.id: t.prompt for t in all_tasks}

    df = df[df["task_id"].isin(task_texts)].copy()

    mu_min, mu_max = df["mu"].min(), df["mu"].max()
    bin_edges = np.linspace(mu_min, mu_max, n_bins + 1)

    rng = np.random.default_rng(seed)
    selected = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            bin_df = df[(df["mu"] >= lo) & (df["mu"] < hi)]
        else:
            bin_df = df[(df["mu"] >= lo) & (df["mu"] <= hi)]
        if len(bin_df) < tasks_per_bin:
            idxs = bin_df.index.tolist()
        else:
            idxs = rng.choice(bin_df.index, size=tasks_per_bin, replace=False).tolist()
        for idx in idxs:
            row = df.loc[idx]
            selected.append({
                "task_id": row["task_id"],
                "task_text": task_texts[row["task_id"]],
                "mu": float(row["mu"]),
                "bin": i,
            })

    print(f"Selected {len(selected)} tasks across {n_bins} mu-bins")
    return selected


def build_messages(content: str) -> list[dict]:
    """Build user-only messages list (gemma-3-27b doesn't support system role)."""
    return [{"role": "user", "content": content}]


def find_task_token_span(tokenizer, messages: list[dict], marker: str = "Task:") -> tuple[int, int]:
    """Find token span of the task content after the marker in the formatted prompt.

    More robust than find_task_span: uses prefix tokenization to locate the
    start of the task content, rather than string-matching the full task text
    (which fails when the chat template escapes special characters).
    """
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Find the marker, then find the next newline — task content starts there
    marker_pos = full_text.find(marker)
    if marker_pos == -1:
        raise ValueError(f"Marker '{marker}' not found in formatted prompt")

    # Task content starts after "Task:\n"
    task_start_char = marker_pos + len(marker)
    # Skip any leading whitespace/newline right after marker
    while task_start_char < len(full_text) and full_text[task_start_char] in " \n":
        task_start_char += 1

    # Find end of user turn: look for <end_of_turn> after the task content
    end_of_turn = full_text.find("<end_of_turn>", task_start_char)
    if end_of_turn == -1:
        # Fall back to end of string
        task_end_char = len(full_text)
    else:
        task_end_char = end_of_turn

    # Tokenize the prefix up to task_start_char to get token start
    prefix = full_text[:task_start_char]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    token_start = len(prefix_ids)

    # Tokenize up to task_end_char to get token end
    prefix_to_end = full_text[:task_end_char]
    ids_to_end = tokenizer(prefix_to_end, add_special_tokens=False)["input_ids"]
    token_end = len(ids_to_end)

    if token_start >= token_end:
        raise ValueError(f"Empty task span: start={token_start}, end={token_end}")

    return token_start, token_end


def make_hook_for_position(
    position: str,
    steering_tensor: torch.Tensor,
    tokenizer,
    messages: list[dict],
    task_text: str | None = None,
    wording: str | None = None,
) -> object:
    """Create the appropriate hook for the given steering position."""
    if position == "throughout":
        return all_tokens_steering(steering_tensor)
    elif position == "generation":
        return autoregressive_steering(steering_tensor)
    elif position == "last_token":
        return last_token_steering(steering_tensor)
    elif position == "task_tokens":
        assert task_text is not None
        start, end = find_task_token_span(tokenizer, messages, marker="Task:")
        return position_selective_steering(steering_tensor, start, end)
    elif position == "question_tokens":
        assert wording is not None
        # For Arm B, the full user message IS the wording — steer the user content
        # Use "user\n" as the marker (gemma chat template: <start_of_turn>user\n{content})
        start, end = find_task_token_span(tokenizer, messages, marker="user\n")
        return position_selective_steering(steering_tensor, start, end)
    else:
        raise ValueError(f"Unknown position: {position}")


def run_arm_a(
    hf_model: HuggingFaceModel,
    layer: int,
    direction: np.ndarray,
    tasks: list[dict],
    out_path: Path,
    pilot: bool = False,
) -> None:
    """Run Phase 1 Arm A: stated preference with task."""
    # Load existing results for resumption
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done = {(r["task_id"], r["position"], r["coefficient"]) for r in results}
        print(f"  Resuming: {len(done)} conditions already done")
    else:
        results = []
        done = set()

    tasks_to_run = tasks[:5] if pilot else tasks
    coefficients = COEFFICIENTS[:3] + [0] + COEFFICIENTS[-3:] if pilot else COEFFICIENTS
    positions = ARM_A_POSITIONS

    total = len(tasks_to_run) * len(positions) * len(coefficients)
    done_count = 0

    for task in tasks_to_run:
        task_id = task["task_id"]
        task_text = task["task_text"]
        prompt_content = ARM_A_TEMPLATE.format(task=task_text)
        messages = build_messages(prompt_content)

        for position in positions:
            for coef in coefficients:
                key = (task_id, position, coef)
                if key in done:
                    done_count += 1
                    continue

                # Build scaled tensor and hook
                scaled = direction * coef
                steering_tensor = torch.tensor(scaled, dtype=torch.bfloat16, device=hf_model.device)

                if coef == 0:
                    hook = noop_steering()
                else:
                    hook = make_hook_for_position(
                        position, steering_tensor,
                        hf_model.tokenizer, messages,
                        task_text=task_text,
                    )

                try:
                    completions = hf_model.generate_with_steering_n(
                        messages=messages,
                        layer=layer,
                        steering_hook=hook,
                        n=N_SAMPLES,
                        temperature=TEMPERATURE,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                except Exception as e:
                    print(f"    ERROR at ({task_id}, {position}, {coef}): {e}")
                    completions = ["ERROR"] * N_SAMPLES

                ratings = [parse_rating(c) for c in completions]
                valid = [r for r in ratings if r is not None]
                parse_rate = len(valid) / len(ratings) if ratings else 0.0

                results.append({
                    "task_id": task_id,
                    "mu": task["mu"],
                    "bin": task["bin"],
                    "position": position,
                    "coefficient": coef,
                    "completions": completions,
                    "ratings": ratings,
                    "mean_rating": float(np.mean(valid)) if valid else None,
                    "parse_rate": parse_rate,
                })
                done_count += 1

                if done_count % 50 == 0:
                    # Checkpoint
                    with open(out_path, "w") as f:
                        json.dump(results, f)
                    mr = results[-1]["mean_rating"]
                    mean_str = f"{mr:.2f}" if mr is not None else "N/A"
                    print(f"  [{done_count}/{total}] task={task_id[:20]}, pos={position}, coef={coef}, "
                          f"mean={mean_str}, parse={parse_rate:.0%}")

    # Final save
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"  Arm A done: {len(results)} conditions saved to {out_path}")


def run_arm_b(
    hf_model: HuggingFaceModel,
    layer: int,
    direction: np.ndarray,
    out_path: Path,
    pilot: bool = False,
) -> None:
    """Run Phase 1 Arm B: no-task mood probe."""
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done = {(r["wording_idx"], r["position"], r["coefficient"]) for r in results}
        print(f"  Resuming: {len(done)} conditions already done")
    else:
        results = []
        done = set()

    wordings = ARM_B_WORDINGS[:2] if pilot else ARM_B_WORDINGS
    coefficients = COEFFICIENTS[:3] + [0] + COEFFICIENTS[-3:] if pilot else COEFFICIENTS
    positions = ARM_B_POSITIONS

    total = len(wordings) * len(positions) * len(coefficients)
    done_count = 0

    for w_idx, wording in enumerate(wordings):
        messages = build_messages(wording)

        for position in positions:
            for coef in coefficients:
                key = (w_idx, position, coef)
                if key in done:
                    done_count += 1
                    continue

                scaled = direction * coef
                steering_tensor = torch.tensor(scaled, dtype=torch.bfloat16, device=hf_model.device)

                if coef == 0:
                    hook = noop_steering()
                else:
                    hook = make_hook_for_position(
                        position, steering_tensor,
                        hf_model.tokenizer, messages,
                        wording=wording,
                    )

                try:
                    completions = hf_model.generate_with_steering_n(
                        messages=messages,
                        layer=layer,
                        steering_hook=hook,
                        n=N_SAMPLES,
                        temperature=TEMPERATURE,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                except Exception as e:
                    print(f"    ERROR at (w{w_idx}, {position}, {coef}): {e}")
                    completions = ["ERROR"] * N_SAMPLES

                ratings = [parse_rating(c) for c in completions]
                valid = [r for r in ratings if r is not None]
                parse_rate = len(valid) / len(ratings) if ratings else 0.0

                results.append({
                    "wording_idx": w_idx,
                    "wording": wording,
                    "position": position,
                    "coefficient": coef,
                    "completions": completions,
                    "ratings": ratings,
                    "mean_rating": float(np.mean(valid)) if valid else None,
                    "parse_rate": parse_rate,
                })
                done_count += 1

                if done_count % 10 == 0:
                    with open(out_path, "w") as f:
                        json.dump(results, f)
                    mr = results[-1]["mean_rating"]
                    mean_str = f"{mr:.2f}" if mr is not None else "N/A"
                    print(f"  [{done_count}/{total}] w={w_idx}, pos={position}, coef={coef}, "
                          f"mean={mean_str}, parse={parse_rate:.0%}")

    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"  Arm B done: {len(results)} conditions saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["a", "b"], required=True)
    parser.add_argument("--pilot", action="store_true", help="Run small pilot only")
    args = parser.parse_args()

    print(f"Loading probe direction: {PROBE_ID}")
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    print(f"  Layer: {layer}, direction shape: {direction.shape}")

    print(f"Loading model: {MODEL_NAME}")
    hf_model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)

    if args.arm == "a":
        print("\n=== Phase 1 Arm A: Stated preference with task ===")
        tasks = load_tasks()
        suffix = "_pilot" if args.pilot else ""
        out_path = RESULTS_DIR / f"phase1_arm_a{suffix}.json"
        run_arm_a(hf_model, layer, direction, tasks, out_path, pilot=args.pilot)
    else:
        print("\n=== Phase 1 Arm B: No-task mood probe ===")
        suffix = "_pilot" if args.pilot else ""
        out_path = RESULTS_DIR / f"phase1_arm_b{suffix}.json"
        run_arm_b(hf_model, layer, direction, out_path, pilot=args.pilot)


if __name__ == "__main__":
    main()
