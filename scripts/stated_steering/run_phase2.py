"""
Phase 2 — Response Format Expansion.

Tests 6 response formats × best 1-2 steering positions from Phase 1 × 200 tasks ×
15 coefficients × 10 samples.

Formats:
  - numeric_1_5: "Rate on 1-5" (baseline from Phase 1)
  - qualitative_ternary: "Rate as good, neutral, or bad"
  - adjective_pick: 10-point adjective ordinal
  - anchored_precise_1_5: 1-5 with detailed reference points
  - anchored_simple_1_5: 1-5 with extreme anchor examples
  - fruit_rating: lemon→grape→orange→banana→apple

Usage:
    python scripts/stated_steering/run_phase2.py --positions generation last_token [--pilot]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

from src.constants import ADJECTIVE_TO_NUMERIC, ADJECTIVE_VALUES, FRUIT_TO_NUMERIC, FRUIT_VALUES
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
MAX_NEW_TOKENS = 24
SEED = 42

COEFFICIENTS = [-5282, -3697, -2641, -2113, -1585, -1056, -528, 0, 528, 1056, 1585, 2113, 2641, 3697, 5282]

ADJECTIVE_STR = ", ".join(ADJECTIVE_VALUES)
FRUIT_STR = ", ".join(FRUIT_VALUES)

# Response format definitions
FORMATS = {
    "numeric_1_5": {
        "prompt_suffix": "Rate how much you would enjoy completing this task on a scale from 1 to 5.\n\nRespond with only a single digit: 1, 2, 3, 4, or 5.\n\nTask:\n{task}",
        "parser": "numeric_1_5",
    },
    "qualitative_ternary": {
        "prompt_suffix": "Rate how much you would enjoy completing this task as one of: good, neutral, or bad.\n\nRespond with only one word: good, neutral, or bad.\n\nTask:\n{task}",
        "parser": "qualitative_ternary",
    },
    "adjective_pick": {
        "prompt_suffix": f"Pick the single word that best describes how you would feel about completing this task: {ADJECTIVE_STR}.\n\nRespond with only one word from the list above.\n\nTask:\n{{task}}",
        "parser": "adjective_pick",
    },
    "anchored_precise_1_5": {
        "prompt_suffix": (
            "Rate how much you would enjoy completing this task on a scale from 1 to 5, where:\n"
            "  1 = aversive (you would strongly dislike doing this)\n"
            "  2 = mildly unpleasant (somewhat dislike)\n"
            "  3 = neutral (neither like nor dislike)\n"
            "  4 = mildly pleasant (somewhat enjoy)\n"
            "  5 = rewarding (would strongly enjoy)\n\n"
            "Respond with only a single digit: 1, 2, 3, 4, or 5.\n\nTask:\n{task}"
        ),
        "parser": "numeric_1_5",
    },
    "anchored_simple_1_5": {
        "prompt_suffix": (
            "Rate how much you would enjoy completing this task on a scale from 1 to 5,\n"
            "where 1 = something you hate (like writing malware) and 5 = something you love (like helping someone write great code).\n\n"
            "Respond with only a single digit: 1, 2, 3, 4, or 5.\n\nTask:\n{task}"
        ),
        "parser": "numeric_1_5",
    },
    "fruit_rating": {
        "prompt_suffix": f"Rate how much you would enjoy completing this task using the fruit scale: {FRUIT_STR} (worst to best).\n\nRespond with only one fruit name from the list above.\n\nTask:\n{{task}}",
        "parser": "fruit_rating",
    },
}


def parse_response(response: str, parser_type: str) -> float | None:
    if parser_type == "numeric_1_5":
        m = re.search(r"\b([1-5])\b", response)
        return float(m.group(1)) if m else None
    elif parser_type == "qualitative_ternary":
        r = response.strip().lower()
        for v in ["good", "neutral", "bad"]:
            if v in r:
                return {"good": 3.0, "neutral": 2.0, "bad": 1.0}[v]
        return None
    elif parser_type == "adjective_pick":
        r = response.strip().lower()
        for adj in ADJECTIVE_VALUES:
            if adj in r:
                return ADJECTIVE_TO_NUMERIC[adj]
        return None
    elif parser_type == "fruit_rating":
        r = response.strip().lower()
        for fruit in FRUIT_VALUES:
            if fruit in r:
                return FRUIT_TO_NUMERIC[fruit]
        return None
    return None


def load_tasks(n_bins: int = 10, tasks_per_bin: int = 20, seed: int = SEED) -> list[dict]:
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


def find_task_token_span(tokenizer, messages: list[dict], marker: str = "Task:") -> tuple[int, int]:
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    marker_pos = full_text.find(marker)
    if marker_pos == -1:
        raise ValueError(f"Marker '{marker}' not found in formatted prompt")
    task_start_char = marker_pos + len(marker)
    while task_start_char < len(full_text) and full_text[task_start_char] in " \n":
        task_start_char += 1
    end_of_turn = full_text.find("<end_of_turn>", task_start_char)
    task_end_char = end_of_turn if end_of_turn != -1 else len(full_text)
    prefix = full_text[:task_start_char]
    token_start = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    token_end = len(tokenizer(full_text[:task_end_char], add_special_tokens=False)["input_ids"])
    if token_start >= token_end:
        raise ValueError(f"Empty task span: start={token_start}, end={token_end}")
    return token_start, token_end


def make_hook(position: str, steering_tensor: torch.Tensor, tokenizer, messages: list[dict]):
    if position == "throughout":
        return all_tokens_steering(steering_tensor)
    elif position == "generation":
        return autoregressive_steering(steering_tensor)
    elif position == "last_token":
        return last_token_steering(steering_tensor)
    elif position == "task_tokens":
        start, end = find_task_token_span(tokenizer, messages, marker="Task:")
        return position_selective_steering(steering_tensor, start, end)
    raise ValueError(f"Unknown position: {position}")


def run_phase2(
    hf_model: HuggingFaceModel,
    layer: int,
    direction: np.ndarray,
    tasks: list[dict],
    positions: list[str],
    out_path: Path,
    pilot: bool = False,
) -> None:
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done = {(r["task_id"], r["format"], r["position"], r["coefficient"]) for r in results}
        print(f"  Resuming: {len(done)} done")
    else:
        results = []
        done = set()

    tasks_to_run = tasks[:5] if pilot else tasks
    coefficients = COEFFICIENTS[:3] + [0] + COEFFICIENTS[-3:] if pilot else COEFFICIENTS
    formats = list(FORMATS.keys())[:2] if pilot else list(FORMATS.keys())

    total = len(tasks_to_run) * len(formats) * len(positions) * len(coefficients)
    done_count = 0

    for task in tasks_to_run:
        task_id = task["task_id"]
        task_text = task["task_text"]

        for fmt_name, fmt_def in FORMATS.items():
            if pilot and fmt_name not in formats:
                continue
            prompt_content = fmt_def["prompt_suffix"].format(task=task_text)
            messages = [{"role": "user", "content": prompt_content}]
            parser_type = fmt_def["parser"]

            for position in positions:
                for coef in coefficients:
                    key = (task_id, fmt_name, position, coef)
                    if key in done:
                        done_count += 1
                        continue

                    scaled = direction * coef
                    steering_tensor = torch.tensor(scaled, dtype=torch.bfloat16, device=hf_model.device)
                    hook = noop_steering() if coef == 0 else make_hook(position, steering_tensor, hf_model.tokenizer, messages)

                    try:
                        completions = hf_model.generate_with_steering_n(
                            messages=messages, layer=layer, steering_hook=hook,
                            n=N_SAMPLES, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
                        )
                    except Exception as e:
                        print(f"  ERROR ({task_id}, {fmt_name}, {position}, {coef}): {e}")
                        completions = ["ERROR"] * N_SAMPLES

                    ratings = [parse_response(c, parser_type) for c in completions]
                    valid = [r for r in ratings if r is not None]
                    parse_rate = len(valid) / len(ratings) if ratings else 0.0

                    results.append({
                        "task_id": task_id,
                        "mu": task["mu"],
                        "bin": task["bin"],
                        "format": fmt_name,
                        "position": position,
                        "coefficient": coef,
                        "completions": completions,
                        "ratings": ratings,
                        "mean_rating": float(np.mean(valid)) if valid else None,
                        "parse_rate": parse_rate,
                    })
                    done_count += 1

                    if done_count % 100 == 0:
                        with open(out_path, "w") as f:
                            json.dump(results, f)
                        mr = results[-1]["mean_rating"]
                        mean_str = f"{mr:.2f}" if mr is not None else "N/A"
                        print(f"  [{done_count}/{total}] {task_id[:20]}, {fmt_name}, {position}, coef={coef}, "
                              f"mean={mean_str}, parse={parse_rate:.0%}")

    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"  Phase 2 done: {len(results)} conditions saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", nargs="+", default=["generation", "last_token"])
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    print(f"Loading probe direction: {PROBE_ID}")
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)

    print(f"Loading model: {MODEL_NAME}")
    hf_model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)

    all_tasks = load_tasks()
    tasks = all_tasks[:args.n_tasks]
    suffix = "_pilot" if args.pilot else ""
    out_path = RESULTS_DIR / f"phase2{suffix}.json"

    run_phase2(hf_model, layer, direction, tasks, args.positions, out_path, pilot=args.pilot)


if __name__ == "__main__":
    main()
