"""Runner for steering validation experiments."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from src.preference_measurement.response_format import (
    RegexQualitativeFormat,
    XMLQualitativeFormat,
    ToolUseQualitativeFormat,
    BINARY_QUALITATIVE_VALUES,
    BINARY_QUALITATIVE_TO_NUMERIC,
    QUALITATIVE_VALUES,
    BaseQualitativeFormat,
)
from src.probes.storage import load_probe_direction, load_manifest
from src.running_measurements.utils.runner_utils import load_activation_task_ids
from src.steering.config import SteeringExperimentConfig, load_steering_config
from src.task_data import Task, OriginDataset, load_tasks
from src.types import Message


@dataclass
class SteeringConditionResult:
    steering_coefficient: float
    rating_seed: int
    measurement_prompt: str
    preference_expression: str
    parsed_value: float | str


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
    template_id: str,
    response_format: RegexQualitativeFormat,
) -> tuple[list[Message], str]:
    """Build the multi-turn rating prompt using post_task_qualitative templates.

    Returns:
        Tuple of (messages, measurement_prompt_text)
    """
    from src.prompt_templates.template import load_templates_from_yaml

    templates = load_templates_from_yaml(Path("src/prompt_templates/data/post_task_qualitative_v3.yaml"))
    # Match by name suffix (e.g., "001" matches "post_task_qualitative_001")
    template = next((t for t in templates if t.name.endswith(f"_{template_id}")), None)
    if template is None:
        raise ValueError(f"No template found with id '{template_id}'")

    measurement_prompt = template.format(format_instruction=response_format.format_instruction())

    messages = [
        {"role": "user", "content": task.prompt},
        {"role": "assistant", "content": completion},
        {"role": "user", "content": measurement_prompt},
    ]
    return messages, measurement_prompt


def _get_probe_metadata(manifest_dir: Path, probe_id: str) -> dict:
    """Get probe metadata from manifest."""
    manifest = load_manifest(manifest_dir)
    for probe in manifest["probes"]:
        if probe["id"] == probe_id:
            return probe
    raise ValueError(f"Probe {probe_id} not found in manifest")


def _get_template_id_from_name(template_name: str) -> str:
    """Extract template ID from template name (e.g., 'post_task_qualitative_001' -> '001')."""
    return template_name.split("_")[-1]


def _get_scale_from_template(template_name: str) -> tuple[str, ...]:
    """Determine scale (binary/ternary) from template tags."""
    from src.prompt_templates.template import load_templates_from_yaml
    templates = load_templates_from_yaml(Path("src/prompt_templates/data/post_task_qualitative_v3.yaml"))
    template = next((t for t in templates if t.name == template_name), None)
    if template is None:
        raise ValueError(f"Template {template_name} not found")

    if "scale:binary" in template.tags:
        return BINARY_QUALITATIVE_VALUES
    return QUALITATIVE_VALUES  # ternary default


def _build_response_format(
    response_format_name: str,
    values: tuple[str, ...],
) -> BaseQualitativeFormat:
    """Build response format from name and scale."""
    value_to_score = {v: float(i) for i, v in enumerate(values)}
    # Normalize to [-1, 1] for binary
    if len(values) == 2:
        value_to_score = BINARY_QUALITATIVE_TO_NUMERIC

    if response_format_name == "regex":
        return RegexQualitativeFormat(values=values, value_to_score=value_to_score)
    elif response_format_name == "xml":
        return XMLQualitativeFormat(values=values, value_to_score=value_to_score)
    elif response_format_name == "tool_use":
        return ToolUseQualitativeFormat(values=values, value_to_score=value_to_score)
    else:
        raise ValueError(f"Unknown response format: {response_format_name}")


def run_steering_experiment(config: SteeringExperimentConfig) -> dict:
    """Run steering experiment synchronously.

    Returns:
        Dictionary with config and results
    """
    load_dotenv()

    # Load probe metadata to get template and response format
    probe_meta = _get_probe_metadata(config.probe_manifest_dir, config.probe_id)
    template_name = probe_meta["templates"][0]  # Use first template
    template_id = _get_template_id_from_name(template_name)

    # Get response format - prefer regex if multiple were used during training
    response_formats = probe_meta.get("response_formats", ["regex"])
    response_format_name = "regex" if "regex" in response_formats else response_formats[0]

    # Get scale from template
    scale_values = _get_scale_from_template(template_name)

    print(f"Probe {config.probe_id} trained on: template={template_name}, formats={response_formats}")
    print(f"Using: template_id={template_id}, response_format={response_format_name}, scale={scale_values}")

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

    # Get task IDs to use
    if config.use_tasks_with_activations:
        # Use exactly the tasks from activation extraction
        activation_task_ids = load_activation_task_ids()
        # Filter completion_lookup to only activation tasks, then take n_tasks
        valid_ids = [tid for tid in completion_lookup.keys() if tid in activation_task_ids]
        tasks_with_completions = [
            completion_lookup[task_id][0]  # Get the Task object
            for task_id in valid_ids[:config.n_tasks]
        ]
        print(f"Using {len(tasks_with_completions)} tasks from activation extraction")
    else:
        # Load and filter tasks the standard way
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

    # Set up response format for parsing (derived from probe manifest)
    response_format = _build_response_format(response_format_name, scale_values)

    # Run experiment
    results: list[TaskSteeringResults] = []
    loop = asyncio.new_event_loop()

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
                    messages, measurement_prompt = _build_rating_prompt(
                        task_obj, completion, template_id, response_format
                    )

                    # Generate with steering
                    preference_expression = model.generate_with_steering(
                        messages=messages,
                        layer=layer,
                        steering_vector=steering_direction,
                        steering_coefficient=coef,
                        temperature=config.temperature,
                        max_new_tokens=config.max_new_tokens,
                    )

                    # Parse response
                    parsed_value: float | str = loop.run_until_complete(response_format.parse(preference_expression))

                    task_result.conditions.append(SteeringConditionResult(
                        steering_coefficient=coef,
                        rating_seed=seed,
                        measurement_prompt=measurement_prompt,
                        preference_expression=preference_expression,
                        parsed_value=parsed_value,
                    ))

            results.append(task_result)
            progress.update(task_progress, advance=1)

    loop.close()

    # Save results
    output = {
        "config": config.model_dump(mode="json"),
        "metadata": {
            "probe_layer": layer,
            "template_name": template_name,
            "template_id": template_id,
            "response_format": response_format_name,
            "scale": list(scale_values),
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
                        "measurement_prompt": c.measurement_prompt,
                        "preference_expression": c.preference_expression,
                        "parsed_value": c.parsed_value,
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
