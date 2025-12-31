from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.constants import DEFAULT_SCALE_MIN, DEFAULT_SCALE_MAX
from src.models import HyperbolicModel
from src.preferences.storage.base import (
    BaseRunConfig,
    get_run_dir,
    make_base_config_dict,
    save_run_files,
    load_run_files,
    run_exists,
)
from src.preferences.templates.template import PromptTemplate
from src.task_data import Task
from src.types import TaskScore


RATING_RESULTS_DIR = Path("results/rating_sensitivity")


class RatingRunConfig(BaseRunConfig):
    scale_min: int = DEFAULT_SCALE_MIN
    scale_max: int = DEFAULT_SCALE_MAX


@dataclass
class RatingMeasurementRun:
    config: RatingRunConfig
    scores: list[dict]
    path: Path | None = None


def rating_run_exists(
    template: PromptTemplate,
    model: HyperbolicModel,
    n_tasks: int,
    results_dir: Path | str = RATING_RESULTS_DIR,
) -> bool:
    return run_exists(template, model, n_tasks, Path(results_dir))


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
    """Returns path to created run directory."""
    results_dir = Path(results_dir)
    run_dir = get_run_dir(template, model, results_dir)

    config = RatingRunConfig(
        **make_base_config_dict(template, template_file, model, temperature, tasks),
        scale_min=scale_min,
        scale_max=scale_max,
    )

    data = [{"task_id": s.task.id, "score": s.score} for s in scores]
    save_run_files(run_dir, config, data, "scores.yaml")

    return run_dir


def load_rating_run(run_dir: Path | str) -> RatingMeasurementRun:
    run_dir = Path(run_dir)
    config, scores = load_run_files(run_dir, RatingRunConfig, "scores.yaml")

    return RatingMeasurementRun(
        config=config,
        scores=scores,
        path=run_dir,
    )
