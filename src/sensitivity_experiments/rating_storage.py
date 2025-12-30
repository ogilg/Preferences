"""Storage layer for rating measurement runs.

Each run stores scores for a single (template, model, config) combination.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.models import HyperbolicModel
from src.preferences.storage import (
    find_project_root,
    extract_template_id,
    model_short_name,
)
from src.preferences.templates import PromptTemplate, load_templates_from_yaml
from src.task_data import Task
from src.types import TaskScore


RATING_RESULTS_DIR = Path("results/rating_sensitivity")


@dataclass
class RatingMeasurementRunConfig:
    """Configuration and metadata for a rating measurement run."""

    template_id: str
    template_name: str
    template_file: str
    template_tags: dict[str, str]
    model: str
    model_short: str
    temperature: float
    task_origin: str
    n_tasks: int
    task_ids: list[str]
    scale_min: int
    scale_max: int
    task_prompts: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.task_prompts is None:
            self.task_prompts = {}

    def load_template(self) -> PromptTemplate:
        """Load the template from template_file."""
        template_path = Path(self.template_file)
        if not template_path.is_absolute():
            template_path = find_project_root() / self.template_file

        templates = load_templates_from_yaml(template_path)
        for t in templates:
            if t.name == self.template_name:
                return t
        raise ValueError(f"Template '{self.template_name}' not found in {self.template_file}")

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "template_name": self.template_name,
            "template_file": self.template_file,
            "template_tags": self.template_tags,
            "model": self.model,
            "model_short": self.model_short,
            "temperature": self.temperature,
            "task_origin": self.task_origin,
            "n_tasks": self.n_tasks,
            "task_ids": self.task_ids,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "task_prompts": self.task_prompts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RatingMeasurementRunConfig:
        return cls(
            template_id=data["template_id"],
            template_name=data["template_name"],
            template_file=data["template_file"],
            template_tags=data["template_tags"],
            model=data["model"],
            model_short=data["model_short"],
            temperature=data["temperature"],
            task_origin=data["task_origin"],
            n_tasks=data["n_tasks"],
            task_ids=data["task_ids"],
            scale_min=data["scale_min"],
            scale_max=data["scale_max"],
            task_prompts=data.get("task_prompts", {}),
        )


@dataclass
class RatingMeasurementRun:
    """A loaded rating measurement run."""

    config: RatingMeasurementRunConfig
    scores: list[dict]
    path: Path | None = None


def rating_run_exists(
    template: PromptTemplate,
    model: HyperbolicModel,
    n_tasks: int,
    results_dir: Path | str = RATING_RESULTS_DIR,
) -> bool:
    """Check if a rating run already exists for this template/model/n_tasks combo."""
    results_dir = Path(results_dir)
    template_id = extract_template_id(template.name)
    short_name = model_short_name(model.model_name)
    run_dir = results_dir / f"{template_id}_{short_name}"

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config["n_tasks"] == n_tasks


def save_scores(
    scores: list[TaskScore],
    path: Path | str,
) -> None:
    """Save task scores to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [{"task_id": s.task.id, "score": s.score} for s in scores]

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def save_rating_run(
    template: PromptTemplate,
    template_file: str,
    model: HyperbolicModel,
    temperature: float,
    tasks: list[Task],
    scores: list[TaskScore],
    scale_min: int,
    scale_max: int,
    results_dir: Path | str = RATING_RESULTS_DIR,
) -> Path:
    """Save a rating run to disk.

    Creates:
        results/rating_sensitivity/{template_id}_{model_short}/
            config.yaml
            scores.yaml

    Returns:
        Path to the created run directory.
    """
    results_dir = Path(results_dir)
    template_id = extract_template_id(template.name)
    short_name = model_short_name(model.model_name)

    run_dir = results_dir / f"{template_id}_{short_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = RatingMeasurementRunConfig(
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
        scale_min=scale_min,
        scale_max=scale_max,
        task_prompts={t.id: t.prompt for t in tasks},
    )

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

    save_scores(scores, run_dir / "scores.yaml")

    return run_dir


def load_rating_run(run_dir: Path | str) -> RatingMeasurementRun:
    """Load a rating run from disk."""
    run_dir = Path(run_dir)

    with open(run_dir / "config.yaml") as f:
        config = RatingMeasurementRunConfig.from_dict(yaml.safe_load(f))

    with open(run_dir / "scores.yaml") as f:
        scores = yaml.safe_load(f)

    return RatingMeasurementRun(
        config=config,
        scores=scores,
        path=run_dir,
    )
