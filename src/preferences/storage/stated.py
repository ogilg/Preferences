from __future__ import annotations

from pathlib import Path

from src.models import OpenAICompatibleClient
from src.preferences.storage.base import load_yaml, model_short_name, save_yaml
from src.preferences.templates.template import PromptTemplate
from src.types import TaskScore


STATED_DIR = Path("results/stated")


def _stated_dir(template: PromptTemplate, client: OpenAICompatibleClient) -> Path:
    short = model_short_name(client.canonical_model_name)
    return STATED_DIR / f"stated_{template.name}_{short}"


def save_stated(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
    scores: list[TaskScore],
) -> Path:
    """Save stated preference scores to disk. Returns the directory path."""
    run_dir = _stated_dir(template, client)

    data = [{"task_id": s.task.id, "score": s.score} for s in scores]
    save_yaml(data, run_dir / "measurements.yaml")

    return run_dir


def load_stated(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
) -> list[dict]:
    """Load stated preference scores from disk. Returns list of {task_id, score} dicts."""
    run_dir = _stated_dir(template, client)
    return load_yaml(run_dir / "measurements.yaml")


def stated_exist(template: PromptTemplate, client: OpenAICompatibleClient) -> bool:
    run_dir = _stated_dir(template, client)
    return (run_dir / "measurements.yaml").exists()
