"""Collect probe training data: activations and self-reported enjoyment ratings."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from src.models import NnsightModel
from src.preferences.measurement.measurer import StatedScoreMeasurer
from src.preferences.measurement.response_format import RegexQualitativeFormat
from src.preferences.templates import TEMPLATES_DATA_DIR, PostTaskStatedPromptBuilder, load_templates_from_yaml
from src.probes.data import ProbeDataPoint, save_probe_dataset
from src.task_data import load_tasks, OriginDataset

DEFAULT_TEMPLATE_PATH = TEMPLATES_DATA_DIR / "post_task_qualitative_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect probe training data")
    parser.add_argument("config", type=Path, help="Path to probe experiment config YAML")
    return parser.parse_args()


def load_probe_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_probe_config(args.config)

    model_name = config["model"]
    n_tasks = config["n_tasks"]
    task_origins = [OriginDataset[o.upper()] for o in config["task_origins"]]
    layers = config["layers_to_extract"]
    temperature = config.get("temperature", 1.0)
    max_new_tokens = config.get("max_new_tokens", 2048)
    rating_max_tokens = config.get("rating_max_tokens", 32)
    seed = config.get("seed")
    output_dir = Path(config["output_dir"])
    template_path = Path(config.get("template", DEFAULT_TEMPLATE_PATH))

    print(f"Loading {n_tasks} tasks from {[o.value for o in task_origins]}...")
    tasks = load_tasks(n=n_tasks, origins=task_origins, seed=seed)

    print(f"Loading template from {template_path}...")
    templates = load_templates_from_yaml(template_path)
    template = templates[0]
    print(f"Using template: {template.name}")

    response_format = RegexQualitativeFormat()
    builder = PostTaskStatedPromptBuilder(
        measurer=StatedScoreMeasurer(),
        response_format=response_format,
        template=template,
    )

    print(f"Loading model: {model_name}...")
    model = NnsightModel(model_name, max_new_tokens=max_new_tokens)

    resolved_layers = [model.resolve_layer(layer) for layer in layers]
    print(f"Extracting from layers: {layers} -> resolved to {resolved_layers} (model has {model.n_layers} layers)")

    data_points: list[ProbeDataPoint] = []
    failures: list[tuple[str, str]] = []

    n_truncated = 0
    print(f"Collecting data for {len(tasks)} tasks...")
    for task in tqdm(tasks, desc="Tasks"):
        try:
            messages = [{"role": "user", "content": task.prompt}]
            completion, activations = model.generate_with_activations(
                messages, layers=resolved_layers, temperature=temperature
            )
            completion_tokens = len(model.tokenizer.encode(completion))
            truncated = completion_tokens >= max_new_tokens
            if truncated:
                n_truncated += 1

            prompt_tokens = len(model.tokenizer.encode(task.prompt))

            rating_prompt = builder.build(task, completion)
            rating_response = model.generate(
                rating_prompt.messages, temperature=0.0, max_new_tokens=rating_max_tokens
            )

            try:
                parsed = rating_prompt.measurer.parse(rating_response, rating_prompt)
                score = parsed.result.score
            except ValueError as e:
                failures.append((task.id, f"Parse error: {e}"))
                continue

            data_points.append(ProbeDataPoint(
                task_id=task.id,
                activations=activations,
                score=score,
                completion=completion,
                raw_rating_response=rating_response,
                truncated=truncated,
                origin=task.origin.name,
                task_metadata=task.metadata,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ))

        except Exception as e:
            failures.append((task.id, str(e)))
            continue

    print(f"\nCollected {len(data_points)} data points, {len(failures)} failures, {n_truncated} truncated")

    if data_points:
        metadata = {
            "model": model_name,
            "n_tasks": n_tasks,
            "task_origins": [o.value for o in task_origins],
            "layers_config": layers,
            "layers_resolved": resolved_layers,
            "n_model_layers": model.n_layers,
            "template": template.name,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "rating_max_tokens": rating_max_tokens,
            "seed": seed,
            "collected_at": datetime.now().isoformat(),
            "n_successes": len(data_points),
            "n_failures": len(failures),
            "n_truncated": n_truncated,
        }

        print(f"Saving to {output_dir}...")
        save_probe_dataset(data_points, output_dir, metadata=metadata)
        print("Done!")
    else:
        print("No data points collected, nothing to save.")


if __name__ == "__main__":
    main()
