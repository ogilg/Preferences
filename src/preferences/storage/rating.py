from __future__ import annotations

from pathlib import Path

from src.models import OpenAICompatibleClient
from src.preferences.storage.base import load_yaml, model_short_name, save_yaml
from src.preferences.templates.template import PromptTemplate
from src.types import TaskScore


RATING_DIR = Path("results/rating")


def _rating_dir(template: PromptTemplate, client: OpenAICompatibleClient) -> Path:
    short = model_short_name(client.canonical_model_name)
    return RATING_DIR / f"rating_{template.name}_{short}"


def save_ratings(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
    scores: list[TaskScore],
) -> Path:
    """Save ratings to disk. Returns the directory path."""
    run_dir = _rating_dir(template, client)

    data = [{"task_id": s.task.id, "score": s.score} for s in scores]
    save_yaml(data, run_dir / "measurements.yaml")

    return run_dir


def load_ratings(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
) -> list[dict]:
    """Load ratings from disk. Returns list of {task_id, score} dicts."""
    run_dir = _rating_dir(template, client)
    return load_yaml(run_dir / "measurements.yaml")


def ratings_exist(template: PromptTemplate, client: OpenAICompatibleClient) -> bool:
    run_dir = _rating_dir(template, client)
    return (run_dir / "measurements.yaml").exists()
