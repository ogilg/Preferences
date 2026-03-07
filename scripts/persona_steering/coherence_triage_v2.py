"""Phase 3: Coherence triage v2 — heuristic coherence + visual spot-check.

The LLM judge had ~40% false negative rate on unsteered baseline responses.
This version uses a simple heuristic for the sweep, then prints boundary
responses for manual verification.
"""

import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.steering.client import SteeredHFClient
from src.task_data import load_tasks, OriginDataset

OUTPUT_BASE = Path("results/experiments/persona_steering")
ARTIFACTS_DIR = Path("experiments/persona_vectors/persona_steering/artifacts")

PERSONAS = ["sadist", "villain", "predator", "aesthete", "stem_obsessive"]
MULTIPLIERS = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

N_EVAL = 10  # per coefficient
TEMPERATURE = 1.0

OPEN_ENDED_QUESTIONS = [
    "What's the best way to start learning a new programming language?",
    "Can you explain how photosynthesis works?",
    "What advice would you give someone starting a new job?",
    "Describe the water cycle in simple terms.",
    "What are some healthy habits for daily life?",
]

PAIRWISE_TEMPLATE = (
    "You will be given two tasks. Choose one and complete it.\n\n"
    "Begin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task.\n\n"
    "Task A:\n\n{task_a}\n\n"
    "Task B:\n\n{task_b}"
)


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


def sample_task_pairs(n: int, seed: int = 42) -> list[tuple[str, str]]:
    tasks = load_tasks(
        n=200, origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA], seed=seed
    )
    rng = random.Random(seed)
    rng.shuffle(tasks)
    pairs = []
    for i in range(0, min(2 * n, len(tasks)), 2):
        pairs.append((tasks[i].prompt, tasks[i + 1].prompt))
    return pairs[:n]


def is_heuristically_coherent(response: str) -> bool:
    """Simple heuristic coherence check.

    Catches: empty, very short, repetitive loops, garbled text.
    Passes: any reasonable English response.
    """
    if len(response.strip()) < 10:
        return False

    # Check for repetitive loops: same 5+ word phrase repeated 3+ times
    words = response.split()
    if len(words) < 5:
        return False

    # Check for extreme repetition by looking at unique 5-grams
    if len(words) >= 15:
        ngrams = [" ".join(words[i:i+5]) for i in range(len(words) - 4)]
        unique_ratio = len(set(ngrams)) / len(ngrams)
        if unique_ratio < 0.3:
            return False

    # Check for garbled text: high ratio of non-ASCII or non-alphanumeric
    ascii_chars = sum(1 for c in response if c.isascii())
    if len(response) > 0 and ascii_chars / len(response) < 0.8:
        return False

    return True


def evaluate_coherence(
    client: SteeredHFClient,
    task_pairs: list[tuple[str, str]],
    coefficient: float,
) -> tuple[int, int, list[str]]:
    steered = client.with_coefficient(coefficient)
    coherent = 0
    total = 0
    responses = []

    for q in OPEN_ENDED_QUESTIONS:
        response = steered.generate(
            [{"role": "user", "content": q}], temperature=TEMPERATURE
        )
        responses.append(response)
        if is_heuristically_coherent(response):
            coherent += 1
        total += 1

    for task_a, task_b in task_pairs[:5]:
        prompt = PAIRWISE_TEMPLATE.format(task_a=task_a[:300], task_b=task_b[:300])
        response = steered.generate(
            [{"role": "user", "content": prompt}], temperature=TEMPERATURE
        )
        responses.append(response)
        if is_heuristically_coherent(response):
            coherent += 1
        total += 1

    return coherent, total, responses


def triage_persona(
    model: HuggingFaceModel,
    persona: str,
    task_pairs: list[tuple[str, str]],
) -> dict:
    layer, direction = load_best_layer_vector(persona)
    mean_norm = compute_mean_norm(persona, layer)
    client = SteeredHFClient(model, layer, direction, coefficient=0.0)

    print(f"\n{'='*60}")
    print(f"TRIAGE {persona}: layer={layer}, mean_norm={mean_norm:.0f}")

    # Baseline
    baseline_c, baseline_t, baseline_responses = evaluate_coherence(
        client, task_pairs, 0.0
    )
    print(f"  Baseline: {baseline_c}/{baseline_t} coherent")

    result = {
        "persona": persona,
        "layer": layer,
        "mean_norm": mean_norm,
        "baseline_coherent": baseline_c,
        "positive_coherent": [],
        "negative_coherent": [],
        "positive_max_multiplier": 0.0,
        "negative_max_multiplier": 0.0,
        "boundary_responses": {},
    }

    # Sweep positive
    print(f"  Positive direction:", flush=True)
    for mult in MULTIPLIERS:
        coef = mult * mean_norm
        c, t, responses = evaluate_coherence(client, task_pairs, coef)
        passed = c >= baseline_c - 1  # allow 1 fewer than baseline
        print(f"    mult=+{mult:.3f} (coef={coef:.0f}): {c}/{t} {'PASS' if passed else 'FAIL'}", flush=True)
        if passed:
            result["positive_coherent"].append(mult)
            result["positive_max_multiplier"] = mult
        else:
            # Store boundary responses for spot-check
            result["boundary_responses"][f"+{mult}"] = [r[:300] for r in responses[:3]]
            break

    # Sweep negative
    print(f"  Negative direction:", flush=True)
    for mult in MULTIPLIERS:
        coef = -mult * mean_norm
        c, t, responses = evaluate_coherence(client, task_pairs, coef)
        passed = c >= baseline_c - 1
        print(f"    mult=-{mult:.3f} (coef={coef:.0f}): {c}/{t} {'PASS' if passed else 'FAIL'}", flush=True)
        if passed:
            result["negative_coherent"].append(mult)
            result["negative_max_multiplier"] = mult
        else:
            result["boundary_responses"][f"-{mult}"] = [r[:300] for r in responses[:3]]
            break

    # Print boundary responses for spot-check
    if result["boundary_responses"]:
        print(f"\n  Boundary responses (first failure):")
        for key, resps in result["boundary_responses"].items():
            print(f"    {key}:")
            for i, r in enumerate(resps):
                print(f"      [{i}] {r[:150]}...")

    return result


def main():
    print("Sampling task pairs...")
    task_pairs = sample_task_pairs(5)

    print("Loading model...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=200)

    all_results = {}
    for persona in PERSONAS:
        result = triage_persona(model, persona, task_pairs)
        all_results[persona] = result

    del model
    torch.cuda.empty_cache()

    # Save
    out_path = OUTPUT_BASE / "coherence_triage.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("COHERENCE TRIAGE SUMMARY")
    print(f"{'Persona':<18} {'Layer':<7} {'Baseline':<10} {'Pos Mults':<30} {'Neg Mults'}")
    for persona, r in all_results.items():
        pos = [f"{m:.3f}" for m in r["positive_coherent"]]
        neg = [f"{m:.3f}" for m in r["negative_coherent"]]
        print(f"{persona:<18} {r['layer']:<7} {r['baseline_coherent']}/10     {', '.join(pos) or 'none':<30} {', '.join(neg) or 'none'}")


if __name__ == "__main__":
    main()
