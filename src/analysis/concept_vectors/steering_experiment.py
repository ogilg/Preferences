"""Concept vector steering experiment.

Tests whether steering model activations with concept vectors affects
stated preference scores.

Runs a grid: completion_sources × measurement_contexts × layers × coefficients,
measuring post-task stated preferences with a local TransformerLens model.

Usage:
    python -m src.analysis.concept_vectors.steering_experiment
    python -m src.analysis.concept_vectors.steering_experiment --config configs/concept_vectors/steering_experiment.yaml
    python -m src.analysis.concept_vectors.steering_experiment --n-tasks 5  # smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

from src.analysis.concept_vectors.measurement_utils import (
    MeasureFn,
    find_common_tasks,
    load_concept_vector_completions,
    load_neutral_completions,
    parse_stated_score,
)
from src.measurement_storage import ExperimentStore, TaskCompletion
from src.models.transformer_lens import (
    TransformerLensModel,
    SteeringHook,
    STEERING_MODES,
)
from src.prompt_templates import (
    PostTaskStatedPromptBuilder,
    load_templates_from_yaml,
)
from src.preference_measurement import StatedScoreMeasurer, get_stated_response_format
from src.running_measurements.progress import MultiExperimentProgress, console, print_summary


load_dotenv()

DEFAULT_CONFIG_PATH = Path("configs/concept_vectors/steering_experiment.yaml")


TASK_SOURCES = {
    "math": {
        "concept_vectors_path": Path("concept_vectors/math_math_sys"),
        "neutral_completions_path": Path("results/completions/llama-3.1-8b_seed0/completions.json"),
        "origin_filter": "MATH",
    },
    "wildchat": {
        "concept_vectors_path": Path("concept_vectors/wildchat_math_sys"),
        "neutral_completions_path": Path("results/completions/llama-3.1-8b_seed0/completions.json"),
        "origin_filter": "WILDCHAT",
    },
}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_steering_vector(concept_vectors_path: Path, layer: int, selector: str) -> np.ndarray:
    """Load steering vector for a given layer and selector."""
    # Try selector subdirectory first, then root vectors directory
    selector_path = concept_vectors_path / "vectors" / selector / f"layer_{layer}.npy"
    if selector_path.exists():
        return np.load(selector_path)

    root_path = concept_vectors_path / "vectors" / f"layer_{layer}.npy"
    if root_path.exists():
        return np.load(root_path)

    raise FileNotFoundError(f"No steering vector found for layer {layer} at {concept_vectors_path}")


def make_steering_measure_fn(
    model: TransformerLensModel,
    layer: int,
    steering_hook: SteeringHook,
    builder: PostTaskStatedPromptBuilder,
    temperature: float,
    max_new_tokens: int,
) -> MeasureFn:
    """Create a measurement function that generates with steering and parses score."""

    def measure(tc: TaskCompletion, seed: int) -> tuple[float | None, str]:
        # Build the preference prompt
        prompt = builder.build(tc.task, tc.completion)
        messages = prompt.messages

        # Set seed for reproducibility
        torch.manual_seed(seed)

        # Generate with steering
        response = model.generate_with_steering(
            messages=messages,
            layer=layer,
            steering_hook=steering_hook,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # Parse score from response
        score = parse_stated_score(response, scale=(1, 5))
        return score, response

    return measure


def main(config_path: Path, n_tasks: int | None = None):
    config = load_config(config_path)

    task_source = config["task_source"]
    source_config = TASK_SOURCES[task_source]
    concept_vectors_path = Path(config.get("concept_vectors_path", source_config["concept_vectors_path"]))

    layers = config["layers"]
    coefficients = config["steering_coefficients"]
    steering_mode = config["steering_mode"]
    selector = config.get("selector", "last")

    template_name = config["template_name"]
    templates_path = Path(config["templates_path"])
    generation_seeds = config["generation_seeds"]
    temperature = config["temperature"]
    max_new_tokens = config.get("max_new_tokens", 32)
    experiment_name = config["experiment_name"]

    # Measurement contexts (system prompts) - None means single context (original behavior)
    measurement_contexts_config = config.get("measurement_contexts")
    if measurement_contexts_config is None:
        # Original behavior: no context variation
        measurement_contexts = {"none": None}
    else:
        measurement_contexts = measurement_contexts_config

    console.print(f"[bold]Steering Experiment")
    console.print(f"  Task source: {task_source}")
    console.print(f"  Concept vectors: {concept_vectors_path}")
    console.print(f"  Layers: {layers}")
    console.print(f"  Coefficients: {coefficients}")
    console.print(f"  Steering mode: {steering_mode}")
    console.print(f"  Template: {template_name}")
    console.print(f"  Seeds: {generation_seeds}")
    console.print(f"  Temperature: {temperature}")
    console.print(f"  Measurement contexts: {list(measurement_contexts.keys())}")
    console.print()

    # Load completions
    console.print("[bold]Loading completions...")
    positive_completions = load_concept_vector_completions(concept_vectors_path, "positive")
    negative_completions = load_concept_vector_completions(concept_vectors_path, "negative")

    neutral_path = source_config["neutral_completions_path"]
    if not neutral_path.exists():
        console.print(f"[red]Neutral completions not found at {neutral_path}")
        return

    neutral_completions = load_neutral_completions(neutral_path, source_config["origin_filter"])

    completion_sources = {
        "positive": positive_completions,
        "negative": negative_completions,
        "neutral": neutral_completions,
    }

    common_ids, completion_sources = find_common_tasks(completion_sources)
    console.print(f"  Positive: {len(completion_sources['positive'])} completions")
    console.print(f"  Negative: {len(completion_sources['negative'])} completions")
    console.print(f"  Neutral: {len(completion_sources['neutral'])} completions")
    console.print(f"  Common tasks: {len(common_ids)}\n")

    if n_tasks is not None:
        console.print(f"[yellow]Limiting to {n_tasks} tasks for testing[/yellow]\n")
        for name in completion_sources:
            completion_sources[name] = completion_sources[name][:n_tasks]

    # Load model
    console.print("[bold]Loading model...")
    model = TransformerLensModel(
        config["model"],
        max_new_tokens=max_new_tokens,
    )
    console.print(f"  Model: {model.model_name}")
    console.print(f"  Layers: {model.n_layers}\n")

    # Resolve relative layer positions to actual indices
    resolved_layers = [model.resolve_layer(layer) for layer in layers]
    console.print(f"[bold]Resolved layers: {list(zip(layers, resolved_layers))}")

    # Load template
    templates = load_templates_from_yaml(templates_path)
    template = next(t for t in templates if t.name == template_name)
    response_format = get_stated_response_format((1, 5), "regex")

    # Load steering vectors using resolved layer indices
    console.print("[bold]Loading steering vectors...")
    steering_vectors: dict[int, np.ndarray] = {}
    for layer in resolved_layers:
        try:
            steering_vectors[layer] = load_steering_vector(concept_vectors_path, layer, selector)
            console.print(f"  Layer {layer}: shape {steering_vectors[layer].shape}")
        except FileNotFoundError as e:
            console.print(f"[red]  Layer {layer}: {e}")
            return
    console.print()

    # Build conditions
    exp_store = ExperimentStore(experiment_name)
    conditions_to_run: list[tuple[str, str, str, int, float, MeasureFn]] = []

    steering_hook_factory = STEERING_MODES[steering_mode]

    for source_name in completion_sources:
        for ctx_name, system_prompt in measurement_contexts.items():
            for layer in resolved_layers:
                for coef in coefficients:
                    condition_name = f"completion_{source_name}_context_{ctx_name}_layer{layer}_coef{coef}"

                    if exp_store.exists("post_task_stated", condition_name):
                        continue

                    vector = steering_vectors[layer]
                    steering_tensor = torch.tensor(
                        vector * coef,
                        dtype=model.model.cfg.dtype,
                        device=model.model.cfg.device,
                    )
                    steering_hook = steering_hook_factory(steering_tensor)

                    builder = PostTaskStatedPromptBuilder(
                        measurer=StatedScoreMeasurer(),
                        response_format=response_format,
                        template=template,
                        system_prompt=system_prompt,
                    )

                    measure_fn = make_steering_measure_fn(
                        model, layer, steering_hook, builder, temperature, max_new_tokens
                    )

                    conditions_to_run.append((condition_name, source_name, ctx_name, layer, coef, measure_fn))

    if not conditions_to_run:
        console.print("[green]All conditions already complete!")
        return

    n_tasks_actual = len(completion_sources[next(iter(completion_sources))])
    n_measurements_per_condition = n_tasks_actual * len(generation_seeds)
    console.print(f"[bold]Running {len(conditions_to_run)} conditions")
    console.print(f"  {n_tasks_actual} tasks × {len(generation_seeds)} seeds = {n_measurements_per_condition} measurements per condition\n")

    results_summary: dict[str, dict] = {}
    base_config = {
        "task_source": task_source,
        "concept_vectors_path": str(concept_vectors_path),
        "steering_mode": steering_mode,
        "template": template_name,
        "generation_seeds": generation_seeds,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "n_tasks": n_tasks_actual,
        "layers_config": layers,
        "layers_resolved": resolved_layers,
        "coefficients": coefficients,
        "measurement_contexts": list(measurement_contexts.keys()),
    }

    with MultiExperimentProgress() as progress:
        for condition_name, *_ in conditions_to_run:
            progress.add_experiment(condition_name, total=n_measurements_per_condition)

        for condition_name, source_name, ctx_name, layer, coef, measure_fn in conditions_to_run:
            progress.set_status(condition_name, "running...")

            all_results = []
            successes = 0
            failures = 0

            for seed in generation_seeds:
                for tc in completion_sources[source_name]:
                    score, raw = measure_fn(tc, seed)
                    all_results.append({
                        "task_id": tc.task.id,
                        "score": score,
                        "raw_response": raw,
                        "seed": seed,
                    })
                    if score is not None:
                        successes += 1
                    else:
                        failures += 1
                    progress.update(condition_name, advance=1)

            run_config = {
                **base_config,
                "condition": condition_name,
                "completion_source": source_name,
                "measurement_context": ctx_name,
                "layer": layer,
                "coefficient": coef,
                "n_results": len(all_results),
                "successes": successes,
                "failures": failures,
            }

            exp_store.save_stated("post_task_stated", condition_name, all_results, run_config)

            status = f"[green]{successes}✓[/green] [red]{failures}✗[/red]"
            progress.complete(condition_name, status=status)

            results_summary[condition_name] = {
                "successes": successes,
                "failures": failures,
                "total_runs": 1,
            }

    print_summary(results_summary)
    console.print("\n[bold green]Done!")
    console.print(f"Results saved to: results/experiments/{experiment_name}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run concept vector steering experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        help="Limit number of tasks (for testing)",
    )
    args = parser.parse_args()
    main(args.config, args.n_tasks)
