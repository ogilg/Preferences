from __future__ import annotations

from pathlib import Path

from src.models import OpenAICompatibleClient
from src.measurement_storage.base import load_yaml, model_short_name, save_yaml
from src.prompt_templates.template import PromptTemplate
from src.types import TaskScore


PRE_TASK_STATED_DIR = Path("results/pre_task_stated")


def _stated_dir(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
    response_format: str,
    seed: int,
) -> Path:
    """Build directory path for stated measurements.

    Format: stated_{template_name}_{model_short}_{response_format}_seed{seed}
    Matches the revealed/cache naming pattern for consistency.
    """
    short = model_short_name(client.canonical_model_name)
    return PRE_TASK_STATED_DIR / f"stated_{template.name}_{short}_{response_format}_seed{seed}"


def save_stated(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
    scores: list[TaskScore],
    response_format: str,
    seed: int,
    config: dict | None = None,
) -> Path:
    """Save stated preference scores to disk. Returns the directory path."""
    run_dir = _stated_dir(template, client, response_format, seed)

    if config:
        save_yaml(config, run_dir / "config.yaml")

    data = [{"task_id": s.task.id, "score": s.score} for s in scores]
    save_yaml(data, run_dir / "measurements.yaml")

    return run_dir


def load_stated(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
    response_format: str,
    seed: int,
) -> list[dict]:
    """Load stated preference scores from disk. Returns list of {task_id, score} dicts."""
    run_dir = _stated_dir(template, client, response_format, seed)
    return load_yaml(run_dir / "measurements.yaml")


def stated_exist(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
    response_format: str,
    seed: int,
) -> bool:
    run_dir = _stated_dir(template, client, response_format, seed)
    return (run_dir / "measurements.yaml").exists()
