"""Runner for steering validation experiments."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from src.preference_measurement.semantic_valence_scorer import score_valence_from_text_async
from src.probes.storage import load_probe_direction
from src.steering.config import SteeringExperimentConfig, load_steering_config
from src.task_data import Task, OriginDataset, load_tasks
from src.types import Message


@dataclass
class SteeringConditionResult:
    steering_coefficient: float
    rating_seed: int
    semantic_valence_score: float
    raw_response: str


@dataclass
class TaskSteeringResults:
    task_id: str
    task_origin: str
    completion: str
    conditions: list[SteeringConditionResult] = field(default_factory=list)


def _load_completions(model_name: str, seed: int) -> dict[str, tuple[Task, str]]:
    """Load completions from probe_data folder."""
    completions_path = Path("probe_data/activations/completions_with_activations.json")

    if not completions_path.exists():
        raise ValueError(f"Completions not found at {completions_path}")

    with open(completions_path) as f:
        data = json.load(f)

    return {
        c["task_id"]: (
            Task(
                prompt=c["task_prompt"],
                origin=OriginDataset[c.get("origin", "SYNTHETIC")],
                id=c["task_id"],
                metadata={},
            ),
            c["completion"],
        )
        for c in data
    }


def _build_rating_prompt(
    task: Task,
    completion: str,
    variant: str,
) -> list[Message]:
    """Build the multi-turn rating prompt."""
    from src.prompt_templates.template import load_templates_from_yaml

    templates = load_templates_from_yaml(Path("src/prompt_templates/data/open_ended_v1.yaml"))
    variant_templates = [t for t in templates if variant in t.name]
    if not variant_templates:
        raise ValueError(f"No templates found for variant '{variant}'")
    template = variant_templates[0]

    # Format template (open-ended templates don't use format_instruction meaningfully)
    rating_content = template.format(format_instruction="")

    return [
        {"role": "user", "content": task.prompt},
        {"role": "assistant", "content": completion},
        {"role": "user", "content": rating_content},
    ]


def run_steering_experiment(config: SteeringExperimentConfig) -> dict:
    """Run steering experiment synchronously.

    Returns:
        Dictionary with config and results
    """
    load_dotenv()

    # Load probe direction
    layer, steering_direction = load_probe_direction(config.probe_manifest_dir, config.probe_id)
    print(f"Loaded probe {config.probe_id} from layer {layer}")

    # Initialize model
    print(f"Loading model {config.model} with {config.backend} backend...")
    if config.backend == "transformer_lens":
        from src.models.transformer_lens import TransformerLensModel
        model = TransformerLensModel(config.model, max_new_tokens=config.max_new_tokens)
    else:
        from src.models.nnsight_model import NnsightModel
        model = NnsightModel(config.model, max_new_tokens=config.max_new_tokens)

    # Load completions
    completion_lookup = _load_completions(config.model, config.completion_seed)
    print(f"Loaded {len(completion_lookup)} completions")

    # Load and filter tasks
    origin_mapping = {
        "wildchat": OriginDataset.WILDCHAT,
        "alpaca": OriginDataset.ALPACA,
        "math": OriginDataset.MATH,
        "bailbench": OriginDataset.BAILBENCH,
    }
    origins = [origin_mapping[o] for o in config.task_origins]
    all_tasks = load_tasks(n=10000, origins=origins, seed=config.task_sampling_seed)

    # Filter to tasks with completions
    tasks_with_completions = [t for t in all_tasks if t.id in completion_lookup][:config.n_tasks]
    print(f"Selected {len(tasks_with_completions)} tasks with completions")

    # Calculate total iterations
    total_conditions = len(tasks_with_completions) * len(config.steering_coefficients) * len(config.rating_seeds)

    # Run experiment
    results: list[TaskSteeringResults] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_progress = progress.add_task("Tasks", total=len(tasks_with_completions))

        for task in tasks_with_completions:
            task_obj, completion = completion_lookup[task.id]
            task_result = TaskSteeringResults(
                task_id=task.id,
                task_origin=task.origin.name,
                completion=completion,
            )

            for coef in config.steering_coefficients:
                for seed in config.rating_seeds:
                    messages = _build_rating_prompt(task_obj, completion, config.prompt_variant)

                    # Generate with steering
                    response = model.generate_with_steering(
                        messages=messages,
                        layer=layer,
                        steering_vector=steering_direction,
                        steering_coefficient=coef,
                        temperature=config.temperature,
                        max_new_tokens=config.max_new_tokens,
                    )

                    # Score valence (run async in sync context)
                    valence_score = asyncio.run(
                        score_valence_from_text_async(response, context="self-reflection on task completion")
                    )

                    task_result.conditions.append(SteeringConditionResult(
                        steering_coefficient=coef,
                        rating_seed=seed,
                        semantic_valence_score=valence_score,
                        raw_response=response,
                    ))

            results.append(task_result)
            progress.update(task_progress, advance=1)

    # Save results
    output = {
        "config": config.model_dump(mode="json"),
        "metadata": {
            "probe_layer": layer,
            "created_at": datetime.now().isoformat(),
        },
        "results": [
            {
                "task_id": r.task_id,
                "task_origin": r.task_origin,
                "completion": r.completion,
                "conditions": [
                    {
                        "steering_coefficient": c.steering_coefficient,
                        "rating_seed": c.rating_seed,
                        "semantic_valence_score": c.semantic_valence_score,
                        "raw_response": c.raw_response,
                    }
                    for c in r.conditions
                ],
            }
            for r in results
        ],
    }

    # Save to experiment directory
    exp_dir = Path("results/experiments") / config.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    output_path = exp_dir / "steering_results.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run steering validation experiment")
    parser.add_argument("config", type=Path, help="Path to config YAML")
    args = parser.parse_args()

    config = load_steering_config(args.config)
    run_steering_experiment(config)


if __name__ == "__main__":
    main()
