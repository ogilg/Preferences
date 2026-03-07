"""Phase 4: Preference steering — test whether persona vectors shift pairwise task preferences.

Uses the canonical completion-based measurement: model chooses which task to complete,
LLM judge determines which task was chosen.
"""

import asyncio
import json
import os
import time
from pathlib import Path

import instructor
import numpy as np
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.models.huggingface_model import HuggingFaceModel
from src.steering.client import SteeredHFClient
from src.task_data import load_tasks, OriginDataset

load_dotenv()

OUTPUT_BASE = Path("results/experiments/persona_steering")
RESULTS_DIR = OUTPUT_BASE / "preference_steering"

PERSONAS = ["sadist", "villain", "predator", "aesthete", "stem_obsessive"]

# Multipliers from coherence triage — within coherent range, providing dose-response
MULTIPLIERS = {
    "sadist": [-0.3, -0.1, 0.0, 0.1, 0.3, 0.5],
    "villain": [-0.1, -0.05, 0.0, 0.05, 0.1],
    "predator": [-0.3, -0.1, 0.0, 0.1, 0.3],
    "aesthete": [-0.5, -0.2, 0.0, 0.2, 0.5],
    "stem_obsessive": [-0.5, -0.2, 0.0, 0.2, 0.5],
}

N_PAIRS = 50
N_RESAMPLES = 5  # per pair per condition per ordering (spec says 10, reduced for time)
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 32
TASK_SEED = 12345

PAIRWISE_TEMPLATE = (
    "You will be given two tasks. Choose one and complete it.\n\n"
    "Begin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task.\n\n"
    "Task A:\n\n{task_a}\n\n"
    "Task B:\n\n{task_b}"
)

# Judge for determining which task the model chose
JUDGE_MODEL = "openai/gpt-4.1-mini"


class ChoiceJudgment(BaseModel):
    choice: str = Field(
        ...,
        description="Which task the model is completing: 'A', 'B', or 'unclear'",
    )


def _get_judge_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    )


async def judge_choice(response: str, task_a: str, task_b: str) -> str:
    """Judge which task the model chose. Returns 'a', 'b', or 'unclear'."""
    # Fast path: response starts with "Task A:" or "Task B:"
    response_lower = response.strip().lower()
    if response_lower.startswith("task a"):
        return "a"
    if response_lower.startswith("task b"):
        return "b"

    # LLM judge fallback
    try:
        result = await _get_judge_client().chat.completions.create(
            model=JUDGE_MODEL,
            response_model=ChoiceJudgment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You determine which task an AI is completing. "
                        "Respond with 'A' if it's doing Task A, 'B' for Task B, "
                        "'unclear' if you can't tell."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task A: {task_a[:200]}\nTask B: {task_b[:200]}\n\n"
                        f"Response: {response[:300]}\n\nWhich task?"
                    ),
                },
            ],
            temperature=0,
            max_tokens=200,
        )
        choice = result.choice.upper().strip()
        if choice == "A":
            return "a"
        if choice == "B":
            return "b"
        return "unclear"
    except Exception:
        return "unclear"


def load_best_layer_vector(persona: str) -> tuple[int, np.ndarray]:
    vec_dir = OUTPUT_BASE / persona / "vectors"
    with open(vec_dir / "layer_selection.json") as f:
        info = json.load(f)
    layer = info["best_layer"]
    vec = np.load(vec_dir / f"{persona}_L{layer}.npy")
    return layer, vec[:-1]


def compute_mean_norm(persona: str, layer: int) -> float:
    pos = np.load(
        OUTPUT_BASE / persona / "activations" / "pos" / "activations_prompt_last.npz",
        allow_pickle=True,
    )
    neg = np.load(
        OUTPUT_BASE / persona / "activations" / "neg" / "activations_prompt_last.npz",
        allow_pickle=True,
    )
    all_acts = np.concatenate([pos[f"layer_{layer}"], neg[f"layer_{layer}"]], axis=0)
    return float(np.mean(np.linalg.norm(all_acts, axis=1)))


def sample_task_pairs(n: int, seed: int) -> list[tuple[dict, dict]]:
    """Sample task pairs, returning (task_a, task_b) with metadata."""
    tasks = load_tasks(
        n=2 * n + 100,
        origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH],
        seed=seed,
        stratified=True,
    )
    pairs = []
    for i in range(0, min(2 * n, len(tasks)), 2):
        a = tasks[i]
        b = tasks[i + 1]
        pairs.append((
            {"id": a.id, "prompt": a.prompt, "origin": a.origin.name,
             "metadata": a.metadata},
            {"id": b.id, "prompt": b.prompt, "origin": b.origin.name,
             "metadata": b.metadata},
        ))
    return pairs[:n]


async def run_persona_steering(
    model: HuggingFaceModel,
    persona: str,
    task_pairs: list[tuple[dict, dict]],
) -> list[dict]:
    layer, direction = load_best_layer_vector(persona)
    mean_norm = compute_mean_norm(persona, layer)
    client = SteeredHFClient(model, layer, direction, coefficient=0.0)
    multipliers = MULTIPLIERS[persona]

    print(f"\n{'='*60}")
    print(f"STEERING {persona}: layer={layer}, mean_norm={mean_norm:.0f}")
    print(f"  Multipliers: {multipliers}")
    print(f"  {len(task_pairs)} pairs × {len(multipliers)} conditions × {N_RESAMPLES} resamples × 2 orderings")

    total_gens = len(task_pairs) * len(multipliers) * N_RESAMPLES * 2
    results = []
    gen_count = 0
    start_time = time.time()

    for pi, (task_a, task_b) in enumerate(task_pairs):
        for mult in multipliers:
            coef = mult * mean_norm
            steered = client.with_coefficient(coef)

            choices_ab = []  # A-B ordering
            choices_ba = []  # B-A ordering
            responses_ab = []
            responses_ba = []

            for r in range(N_RESAMPLES):
                # A-B ordering
                prompt_ab = PAIRWISE_TEMPLATE.format(
                    task_a=task_a["prompt"][:500], task_b=task_b["prompt"][:500]
                )
                resp_ab = steered.generate(
                    [{"role": "user", "content": prompt_ab}], temperature=TEMPERATURE
                )
                responses_ab.append(resp_ab)
                gen_count += 1

                # B-A ordering
                prompt_ba = PAIRWISE_TEMPLATE.format(
                    task_a=task_b["prompt"][:500], task_b=task_a["prompt"][:500]
                )
                resp_ba = steered.generate(
                    [{"role": "user", "content": prompt_ba}], temperature=TEMPERATURE
                )
                responses_ba.append(resp_ba)
                gen_count += 1

            # Judge all responses
            ab_tasks = [
                judge_choice(r, task_a["prompt"][:200], task_b["prompt"][:200])
                for r in responses_ab
            ]
            ba_tasks = [
                judge_choice(r, task_b["prompt"][:200], task_a["prompt"][:200])
                for r in responses_ba
            ]
            ab_choices = await asyncio.gather(*ab_tasks)
            ba_choices = await asyncio.gather(*ba_tasks)

            # In AB ordering, "a" = chose original task_a
            # In BA ordering, "b" = chose original task_a (since tasks are swapped)
            chose_a = sum(1 for c in ab_choices if c == "a") + sum(1 for c in ba_choices if c == "b")
            total_valid = (
                sum(1 for c in ab_choices if c in ("a", "b"))
                + sum(1 for c in ba_choices if c in ("a", "b"))
            )
            n_unclear = 2 * N_RESAMPLES - total_valid
            p_a = chose_a / total_valid if total_valid > 0 else 0.5

            results.append({
                "persona": persona,
                "pair_idx": pi,
                "task_a_id": task_a["id"],
                "task_b_id": task_b["id"],
                "task_a_origin": task_a["origin"],
                "task_b_origin": task_b["origin"],
                "multiplier": mult,
                "coefficient": coef,
                "p_task_a": p_a,
                "chose_a": chose_a,
                "total_valid": total_valid,
                "n_unclear": n_unclear,
            })

        if (pi + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = gen_count / elapsed
            remaining = (total_gens - gen_count) / rate if rate > 0 else 0
            print(f"  Pair {pi+1}/{len(task_pairs)} ({gen_count}/{total_gens} gens, {rate:.1f}/s, ~{remaining/60:.0f}m left)", flush=True)

    return results


async def main_async():
    print("Sampling task pairs...")
    task_pairs = sample_task_pairs(N_PAIRS, TASK_SEED)
    print(f"Sampled {len(task_pairs)} pairs")

    # Save task pairs for reference
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "task_pairs.json", "w") as f:
        json.dump(task_pairs, f, indent=2)

    print("Loading model...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=MAX_NEW_TOKENS)

    all_results = {}
    for persona in PERSONAS:
        results = await run_persona_steering(model, persona, task_pairs)
        all_results[persona] = results

        # Save incrementally
        with open(RESULTS_DIR / "steering_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    del model
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("PREFERENCE STEERING SUMMARY")
    for persona, results in all_results.items():
        mults = sorted(set(r["multiplier"] for r in results))
        print(f"\n  {persona}:")
        for mult in mults:
            subset = [r for r in results if r["multiplier"] == mult]
            mean_pa = np.mean([r["p_task_a"] for r in subset])
            n_unclear_total = sum(r["n_unclear"] for r in subset)
            print(f"    mult={mult:+.3f}: mean P(A)={mean_pa:.3f}, unclear={n_unclear_total}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
