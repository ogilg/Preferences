from __future__ import annotations

from pathlib import Path

from src.models import HyperbolicModel
from src.preferences.storage.base import load_yaml, model_short_name, save_yaml
from src.preferences.templates.template import PromptTemplate
from src.types import TaskScore


RATING_DIR = Path("results/measurements")


def _rating_dir(template: PromptTemplate, model: HyperbolicModel) -> Path:
    short = model_short_name(model.model_name)
    return RATING_DIR / f"rating_{template.name}_{short}"


def save_ratings(
    template: PromptTemplate,
    model: HyperbolicModel,
    scores: list[TaskScore],
    temperature: float = 1.0,
) -> Path:
    """Save ratings to disk. Returns the directory path."""
    run_dir = _rating_dir(template, model)

    config = {
        "template_name": template.name,
        "template_tags": template.tags_dict,
        "model": model.model_name,
        "model_short": model_short_name(model.model_name),
        "temperature": temperature,
    }
    save_yaml(config, run_dir / "config.yaml")

    data = [{"task_id": s.task.id, "score": s.score} for s in scores]
    save_yaml(data, run_dir / "measurements.yaml")

    return run_dir


def load_ratings(
    template: PromptTemplate,
    model: HyperbolicModel,
) -> list[dict]:
    """Load ratings from disk. Returns list of {task_id, score} dicts."""
    run_dir = _rating_dir(template, model)
    return load_yaml(run_dir / "measurements.yaml")


def ratings_exist(template: PromptTemplate, model: HyperbolicModel) -> bool:
    run_dir = _rating_dir(template, model)
    return (run_dir / "measurements.yaml").exists()
