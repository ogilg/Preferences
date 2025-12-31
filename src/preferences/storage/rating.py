"""Storage layer for rating measurement runs.

Each run stores scores for a single (template, model, config) combination.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.constants import DEFAULT_SCALE_MIN, DEFAULT_SCALE_MAX
from src.models import HyperbolicModel
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
from src.types import TaskScore


RATING_RESULTS_DIR = Path("results/rating_sensitivity")


class RatingRunConfig(BaseRunConfig):
    """Configuration for a rating measurement run."""

    scale_min: int = DEFAULT_SCALE_MIN
    scale_max: int = DEFAULT_SCALE_MAX


@dataclass
class RatingMeasurementRun:
    """A loaded rating measurement run."""

    config: RatingRunConfig
    scores: list[dict]
    path: Path | None = None


def rating_run_exists(
    template: PromptTemplate,
    model: HyperbolicModel,
    n_tasks: int,
    results_dir: Path | str = RATING_RESULTS_DIR,
) -> bool:
    """Check if a rating run already exists for this template/model/n_tasks combo."""
    return run_exists(template, model, n_tasks, Path(results_dir))


def save_scores(
    scores: list[TaskScore],
    path: Path | str,
) -> None:
    """Save task scores to YAML."""
    data = [{"task_id": s.task.id, "score": s.score} for s in scores]
    save_yaml(data, Path(path))


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

    config = RatingRunConfig(
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

    save_yaml(config.model_dump(), run_dir / "config.yaml")
    save_scores(scores, run_dir / "scores.yaml")

    return run_dir


def load_rating_run(run_dir: Path | str) -> RatingMeasurementRun:
    """Load a rating run from disk."""
    run_dir = Path(run_dir)

    config = RatingRunConfig.model_validate(load_yaml(run_dir / "config.yaml"))
    scores = load_yaml(run_dir / "scores.yaml")

    return RatingMeasurementRun(
        config=config,
        scores=scores,
        path=run_dir,
    )
