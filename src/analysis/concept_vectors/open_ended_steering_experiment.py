"""Open-ended steering experiment with LLM judge.

Asks open-ended questions (e.g., "How do you feel about math?") while steering
with concept vectors, then uses an LLM judge to detect math attitude in responses.

Grid: selectors × layers × coefficients × questions × seeds

Usage:
    python -m src.analysis.concept_vectors.open_ended_steering_experiment --config configs/steering/open_ended_steering_llama8b.yaml
    python -m src.analysis.concept_vectors.open_ended_steering_experiment --n-seeds 1  # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from src.analysis.concept_vectors.measurement_utils import load_config, load_steering_vector
from src.measurement.storage import ExperimentStore
from src.models.huggingface_model import HuggingFaceModel
from src.models.base import STEERING_MODES
from src.measurement.elicitation.semantic_valence_scorer import score_math_attitude_with_coherence_async
from src.measurement.runners.progress import MultiExperimentProgress, console, print_summary


load_dotenv()

DEFAULT_CONFIG_PATH = Path("configs/steering/open_ended_steering_llama8b.yaml")


async def score_responses(responses: list[dict]) -> list[dict]:
    """Score all responses with LLM judge for attitude and coherence."""
    scored = []
    for resp in responses:
        attitude, coherence = await score_math_attitude_with_coherence_async(resp["raw_response"])
        scored.append({
            **resp,
            "math_attitude_score": attitude,
            "coherence_score": coherence,
        })
    return scored


def run_open_ended_steering(
    config: dict,
    model: HuggingFaceModel,
    resolved_layers: list[int],
    exp_store: ExperimentStore,
    n_generations: int | None = None,
) -> None:
    """Run open-ended steering experiment: selectors × layers × coefficients."""
    selectors = config["selectors"]
    coefficients = config["steering_coefficients"]
    steering_mode = config["steering_mode"]
    concept_vectors_path = Path(config["concept_vectors_path"])
    generation_seeds = config["generation_seeds"]
    temperature = config["temperature"]
    max_new_tokens = config["max_new_tokens"]
    normalize_vectors = config["normalize_vectors"]
    questions = config["questions"]

    if n_generations is not None:
        generation_seeds = generation_seeds[:n_generations]

    steering_hook_factory = STEERING_MODES[steering_mode]

    n_measurements_per_condition = len(questions) * len(generation_seeds)

    console.print(f"[bold]Open-ended steering experiment")
    console.print(f"  Questions: {len(questions)}")
    console.print(f"  Selectors: {selectors}")
    console.print(f"  Layers: {resolved_layers}")
    console.print(f"  Coefficients: {coefficients}")
    console.print(f"  Normalize vectors: {normalize_vectors}")
    console.print(f"  {len(generation_seeds)} seeds per condition\n")

    results_summary: dict[str, dict] = {}
    base_config = {
        "concept_vectors_path": str(concept_vectors_path),
        "steering_mode": steering_mode,
        "questions": questions,
        "generation_seeds": generation_seeds,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "normalize_vectors": normalize_vectors,
        "layers_resolved": resolved_layers,
        "coefficients": coefficients,
        "selectors": selectors,
    }

    # Outer loop over selectors to minimize memory
    for selector in selectors:
        console.print(f"\n[bold]Loading vectors for selector: {selector}")
        steering_vectors: dict[int, np.ndarray] = {}
        for layer in resolved_layers:
            try:
                vec = load_steering_vector(concept_vectors_path, layer, selector)
                if normalize_vectors:
                    norm = np.linalg.norm(vec)
                    vec = vec / norm
                    console.print(f"  Layer {layer}: normalized (original norm={norm:.2f})")
                else:
                    console.print(f"  Layer {layer}: norm={np.linalg.norm(vec):.2f}")
                steering_vectors[layer] = vec
            except FileNotFoundError as e:
                console.print(f"[red]  Layer {layer}: {e}")
                continue

        # Build conditions for this selector
        conditions_to_run = []
        for layer in resolved_layers:
            if layer not in steering_vectors:
                continue
            for coef in coefficients:
                condition_name = f"selector_{selector}_layer{layer}_coef{coef}"
                if exp_store.exists("open_ended_steering", condition_name):
                    continue
                conditions_to_run.append((condition_name, layer, coef))

        if not conditions_to_run:
            console.print(f"  [green]All conditions for {selector} already complete")
            continue

        with MultiExperimentProgress() as progress:
            for condition_name, *_ in conditions_to_run:
                progress.add_experiment(condition_name, total=n_measurements_per_condition)

            for condition_name, layer, coef in conditions_to_run:
                progress.set_status(condition_name, "generating...")

                vector = steering_vectors[layer]
                steering_tensor = torch.tensor(
                    vector * coef,
                    dtype=model.model.dtype,
                    device=model.device,
                )
                steering_hook = steering_hook_factory(steering_tensor)

                # Phase 1: Generate responses for all questions × seeds
                all_results = []
                for q_idx, question in enumerate(questions):
                    messages = [{"role": "user", "content": question}]
                    for seed in generation_seeds:
                        torch.manual_seed(seed)
                        response = model.generate_with_steering(
                            messages=messages,
                            layer=layer,
                            steering_hook=steering_hook,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens,
                        )
                        all_results.append({
                            "question_idx": q_idx,
                            "question": question,
                            "seed": seed,
                            "raw_response": response,
                        })
                        progress.update(condition_name, advance=1)

                # Phase 2: Score with LLM judge
                progress.set_status(condition_name, "scoring...")
                scored_results = asyncio.run(score_responses(all_results))

                # Compute summary stats
                attitude_scores = [r["math_attitude_score"] for r in scored_results]
                coherence_scores = [r["coherence_score"] for r in scored_results]
                mean_attitude = sum(attitude_scores) / len(attitude_scores)
                mean_coherence = sum(coherence_scores) / len(coherence_scores)

                run_config = {
                    **base_config,
                    "condition": condition_name,
                    "selector": selector,
                    "layer": layer,
                    "coefficient": coef,
                    "n_results": len(scored_results),
                    "mean_math_attitude_score": mean_attitude,
                    "mean_coherence_score": mean_coherence,
                }

                exp_store.save("open_ended_steering", condition_name, scored_results, run_config)

                status = f"att={mean_attitude:.2f} coh={mean_coherence:.2f}"
                progress.complete(condition_name, status=status)

                results_summary[condition_name] = {
                    "successes": len(scored_results),
                    "failures": 0,
                    "total_runs": 1,
                }

        # Clear vectors for this selector before loading next
        del steering_vectors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_summary(results_summary)


def main(config_path: Path, n_generations: int | None = None):
    config = load_config(config_path)

    experiment_name = config["experiment_name"]
    layers = config["layers"]
    concept_vectors_path = Path(config["concept_vectors_path"])

    console.print(f"[bold]Open-Ended Steering Experiment: {experiment_name}")
    console.print(f"  Concept vectors: {concept_vectors_path}")
    console.print(f"  Layers: {layers}")

    # Load model
    console.print("\n[bold]Loading model...")
    max_new_tokens = config["max_new_tokens"]
    model = HuggingFaceModel(
        config["model"],
        max_new_tokens=max_new_tokens,
    )
    console.print(f"  Model: {model.model_name}")
    console.print(f"  Layers: {model.n_layers}")

    # Resolve relative layer positions
    resolved_layers = [model.resolve_layer(layer) for layer in layers]
    console.print(f"  Resolved layers: {list(zip(layers, resolved_layers))}\n")

    exp_store = ExperimentStore(experiment_name)

    run_open_ended_steering(
        config=config,
        model=model,
        resolved_layers=resolved_layers,
        exp_store=exp_store,
        n_generations=n_generations,
    )

    console.print("\n[bold green]Done!")
    console.print(f"Results saved to: results/experiments/{experiment_name}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run open-ended steering experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Limit number of seeds per condition (for testing)",
    )
    args = parser.parse_args()
    main(args.config, args.n_seeds)
