from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, field_validator

from src.models import HyperbolicModel
from src.preferences.ranking import ThurstonianResult, save_thurstonian, load_thurstonian
from src.preferences.storage.base import (
    RESULTS_DIR,
    BaseRunConfig,
    extract_template_id,
    model_short_name,
    run_exists,
    save_yaml,
    load_yaml,
)
from src.preferences.templates.template import PromptTemplate
from src.task_data import Task
from src.types import BinaryPreferenceMeasurement


class BinaryRunConfig(BaseRunConfig):
    pass


@dataclass
class BinaryMeasurementRun:
    config: BinaryRunConfig
    measurements: list[dict]
    thurstonian: ThurstonianResult | None = None
    path: Path | None = None


class ThurstonianData(BaseModel):
    """Lightweight version without Task objects, for visualization."""

    model_config = {"arbitrary_types_allowed": True}

    task_ids: list[str]
    mu: np.ndarray
    sigma: np.ndarray
    converged: bool
    neg_log_likelihood: float

    @field_validator("mu", "sigma", mode="before")
    @classmethod
    def convert_to_array(cls, v: Any) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        return np.array(v)

    def ranking_order(self) -> list[int]:
        return list(np.argsort(-self.mu))


def binary_run_exists(
    template: PromptTemplate,
    model: HyperbolicModel,
    n_tasks: int,
    results_dir: Path | str = RESULTS_DIR,
) -> bool:
    return run_exists(template, model, n_tasks, Path(results_dir))


def save_measurements(
    measurements: list[BinaryPreferenceMeasurement],
    path: Path | str,
) -> None:
    data = [
        {"task_a": m.task_a.id, "task_b": m.task_b.id, "choice": m.choice}
        for m in measurements
    ]
    save_yaml(data, Path(path))


def save_run(
    template: PromptTemplate,
    template_file: str,
    model: HyperbolicModel,
    temperature: float,
    tasks: list[Task],
    measurements: list[BinaryPreferenceMeasurement],
    thurstonian: ThurstonianResult,
    results_dir: Path | str = RESULTS_DIR,
) -> Path:
    """Returns path to created run directory."""
    results_dir = Path(results_dir)
    template_id = extract_template_id(template.name)
    short_name = model_short_name(model.model_name)

    run_dir = results_dir / f"{template_id}_{short_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = BinaryRunConfig(
        template_id=template_id,
        template_name=template.name,
        template_file=template_file,
        template_tags=template.tags_dict,
        model=model.model_name,
        model_short=short_name,
        temperature=temperature,
        task_origin=tasks[0].origin.name.lower(),
        n_tasks=len(tasks),
        task_ids=[t.id for t in tasks],
        task_prompts={t.id: t.prompt for t in tasks},
    )

    save_yaml(config.model_dump(), run_dir / "config.yaml")
    save_measurements(measurements, run_dir / "measurements.yaml")
    save_thurstonian(thurstonian, run_dir / "thurstonian.yaml")

    update_index(results_dir)
    return run_dir


def load_run(
    run_dir: Path | str,
    tasks: list[Task] | None = None,
) -> BinaryMeasurementRun:
    """Pass tasks to reconstruct ThurstonianResult."""
    run_dir = Path(run_dir)

    config = BinaryRunConfig.model_validate(load_yaml(run_dir / "config.yaml"))
    measurements = load_yaml(run_dir / "measurements.yaml")

    thurstonian = None
    if tasks is not None:
        thurstonian = load_thurstonian(run_dir / "thurstonian.yaml", tasks)

    return BinaryMeasurementRun(
        config=config,
        measurements=measurements,
        thurstonian=thurstonian,
        path=run_dir,
    )


def list_runs(
    results_dir: Path | str = RESULTS_DIR,
    **filters: str,
) -> list[BinaryRunConfig]:
    """Filter by config fields or template tags, e.g., model_short="llama-3.1-8b"."""
    results_dir = Path(results_dir)
    index_path = results_dir / "index.yaml"

    if not index_path.exists():
        return []

    index_data = load_yaml(index_path)

    runs = []
    for entry in index_data.get("runs", []):
        if all(entry.get(k) == v for k, v in filters.items()):
            runs.append(BinaryRunConfig.model_validate(load_yaml(results_dir / entry["dir"] / "config.yaml")))

    return runs


def update_index(results_dir: Path | str = RESULTS_DIR) -> None:
    results_dir = Path(results_dir)

    runs = []
    for run_dir in sorted(results_dir.iterdir()):
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            continue

        config = load_yaml(config_path)

        # Skip directories that don't have the expected run config format
        if "template_id" not in config or "model_short" not in config:
            continue

        entry = {
            "dir": run_dir.name,
            "template_id": config["template_id"],
            "model_short": config["model_short"],
        }
        entry.update(config.get("template_tags", {}))
        runs.append(entry)

    save_yaml({"runs": runs}, results_dir / "index.yaml")


def load_thurstonian_data(run_dir: Path | str) -> ThurstonianData:
    run_dir = Path(run_dir)
    data = load_yaml(run_dir / "thurstonian.yaml")

    return ThurstonianData(
        task_ids=data["task_ids"],
        mu=np.array(data["mu"]),
        sigma=np.array(data["sigma"]),
        converged=data["converged"],
        neg_log_likelihood=data["neg_log_likelihood"],
    )
