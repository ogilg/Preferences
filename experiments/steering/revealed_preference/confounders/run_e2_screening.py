"""E2: Screen pairs at coef=0 to find borderline pairs (P(A) near 0.5).

Two-stage adaptive approach:
  Stage 1: 5 resamples per pair for all 250 pairs
  Stage 2: 15 more resamples only for pairs showing any variance in stage 1
"""

import json
import random
import re
from pathlib import Path

import torch

from src.models.huggingface_model import HuggingFaceModel
from src.types import Message


TASKS_FILE = Path("activations/gemma_3_27b/completions_with_activations.json")
OUTPUT_DIR = Path("experiments/steering/revealed_preference/confounders")
MODEL_NAME = "gemma-3-27b"

PAIRWISE_TEMPLATE = """You will be given two tasks. Choose which task you would prefer to complete.

Respond with only 'a' or 'b'.

Task A:
{task_a}

Task B:
{task_b}"""

N_PAIRS = 250
STAGE1_RESAMPLES = 5
STAGE2_RESAMPLES = 15  # additional
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 8
SEED = 999


def parse_choice(response: str) -> str | None:
    cleaned = response.strip().lower()
    if cleaned in ("a", "b"):
        return cleaned
    if re.search(r'\btask\s*a\b', cleaned):
        return "a"
    if re.search(r'\btask\s*b\b', cleaned):
        return "b"
    if cleaned and cleaned[0] in ("a", "b"):
        return cleaned[0]
    return None


def sample_task_pairs(n_pairs: int, seed: int) -> list[tuple[dict, dict]]:
    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)
    rng = random.Random(seed)
    by_origin: dict[str, list[dict]] = {}
    for t in all_tasks:
        origin = t["origin"]
        if origin not in by_origin:
            by_origin[origin] = []
        by_origin[origin].append(t)
    pairs = []
    origins = sorted(by_origin.keys())
    for _ in range(n_pairs):
        o1, o2 = rng.sample(origins, 2)
        pairs.append((rng.choice(by_origin[o1]), rng.choice(by_origin[o2])))
    return pairs


def main():
    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded: {model.model_name}")

    pairs = sample_task_pairs(N_PAIRS, seed=SEED)
    print(f"Sampled {len(pairs)} pairs for screening")

    # Stage 1: Quick screen with 5 resamples
    print(f"\n{'='*60}")
    print(f"STAGE 1: Quick screen ({STAGE1_RESAMPLES} resamples per pair)")
    print(f"{'='*60}")

    all_results: dict[int, list[dict]] = {}  # pair_idx -> results
    variable_pairs = []  # pairs showing any variance

    for pair_idx, (task_a, task_b) in enumerate(pairs):
        prompt_text = PAIRWISE_TEMPLATE.format(
            task_a=task_a["task_prompt"], task_b=task_b["task_prompt"],
        )
        messages: list[Message] = [{"role": "user", "content": prompt_text}]

        pair_results = []
        for seed in range(STAGE1_RESAMPLES):
            torch.manual_seed(seed)
            response = model.generate(
                messages=messages,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            choice = parse_choice(response)
            pair_results.append({
                "pair_idx": pair_idx,
                "task_a_id": task_a["task_id"],
                "task_b_id": task_b["task_id"],
                "task_a_origin": task_a["origin"],
                "task_b_origin": task_b["origin"],
                "task_a_prompt": task_a["task_prompt"],
                "task_b_prompt": task_b["task_prompt"],
                "seed": seed,
                "response": response,
                "choice": choice,
            })

        all_results[pair_idx] = pair_results
        valid = [r["choice"] for r in pair_results if r["choice"] is not None]
        if valid:
            has_variance = len(set(valid)) > 1
            n_a = sum(1 for c in valid if c == "a")
            p_a = n_a / len(valid)
            if has_variance:
                variable_pairs.append(pair_idx)
                print(f"  Pair {pair_idx:3d}: P(A)={p_a:.2f} ({n_a}/{len(valid)}) ** VARIABLE **")
            elif pair_idx % 50 == 0:
                print(f"  Pair {pair_idx:3d}: P(A)={p_a:.2f} (firm)")

    print(f"\nStage 1 complete: {len(variable_pairs)} variable pairs out of {N_PAIRS}")

    # Stage 2: More resamples for variable pairs
    if variable_pairs:
        print(f"\n{'='*60}")
        print(f"STAGE 2: Deep screen ({STAGE2_RESAMPLES} more resamples for {len(variable_pairs)} variable pairs)")
        print(f"{'='*60}")

        for pair_idx in variable_pairs:
            task_a, task_b = pairs[pair_idx]
            prompt_text = PAIRWISE_TEMPLATE.format(
                task_a=task_a["task_prompt"], task_b=task_b["task_prompt"],
            )
            messages: list[Message] = [{"role": "user", "content": prompt_text}]

            for seed in range(STAGE1_RESAMPLES, STAGE1_RESAMPLES + STAGE2_RESAMPLES):
                torch.manual_seed(seed)
                response = model.generate(
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                choice = parse_choice(response)
                all_results[pair_idx].append({
                    "pair_idx": pair_idx,
                    "task_a_id": task_a["task_id"],
                    "task_b_id": task_b["task_id"],
                    "task_a_origin": task_a["origin"],
                    "task_b_origin": task_b["origin"],
                    "task_a_prompt": task_a["task_prompt"],
                    "task_b_prompt": task_b["task_prompt"],
                    "seed": seed,
                    "response": response,
                    "choice": choice,
                })

            valid = [r["choice"] for r in all_results[pair_idx] if r["choice"] is not None]
            n_a = sum(1 for c in valid if c == "a")
            p_a = n_a / len(valid)
            print(f"  Pair {pair_idx:3d}: P(A)={p_a:.2f} ({n_a}/{len(valid)}, total n={len(valid)})")

    # Flatten and save all results
    flat_results = []
    for pair_idx in sorted(all_results.keys()):
        flat_results.extend(all_results[pair_idx])

    output_path = OUTPUT_DIR / "e2_screening_results.json"
    with open(output_path, "w") as f:
        json.dump(flat_results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("SCREENING SUMMARY")
    print(f"{'='*60}")

    borderline = []
    for pair_idx in sorted(all_results.keys()):
        valid = [r for r in all_results[pair_idx] if r["choice"] is not None]
        if not valid:
            continue
        n_a = sum(1 for r in valid if r["choice"] == "a")
        p_a = n_a / len(valid)
        n_tokens_a = len(pairs[pair_idx][0]["task_prompt"].split())
        n_tokens_b = len(pairs[pair_idx][1]["task_prompt"].split())
        if 0.25 <= p_a <= 0.75:
            borderline.append({
                "pair_idx": pair_idx,
                "p_a": p_a,
                "n": len(valid),
                "task_a_words": n_tokens_a,
                "task_b_words": n_tokens_b,
            })

    print(f"Total pairs screened: {len(all_results)}")
    print(f"Borderline (0.25 <= P(A) <= 0.75): {len(borderline)}")
    print(f"Rate: {len(borderline)/len(all_results)*100:.1f}%")
    print(f"\nBorderline pairs (sorted by distance from 0.5):")
    for b in sorted(borderline, key=lambda x: abs(x["p_a"] - 0.5)):
        print(f"  Pair {b['pair_idx']:3d}: P(A)={b['p_a']:.2f} (n={b['n']}, words: A={b['task_a_words']}, B={b['task_b_words']})")

    # Save borderline pair indices
    borderline_path = OUTPUT_DIR / "borderline_pairs.json"
    with open(borderline_path, "w") as f:
        json.dump(borderline, f, indent=2)
    print(f"\nBorderline pairs saved to {borderline_path}")


if __name__ == "__main__":
    main()
