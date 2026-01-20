"""Storage for pre-task stated measurements using unified StatedCache."""

from __future__ import annotations

from pathlib import Path

from src.models import OpenAICompatibleClient
from src.measurement_storage.unified_cache import StatedCache, template_config_from_template
from src.prompt_templates.template import PromptTemplate
from src.types import TaskScore


PRE_TASK_STATED_DIR = Path("results/pre_task_stated")


def save_stated(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
    scores: list[TaskScore],
    response_format: str,
    seed: int,
    config: dict | None = None,
) -> None:
    """Save stated preference scores to unified cache."""
    cache = StatedCache(client.canonical_model_name)
    template_config = template_config_from_template(template)

    for s in scores:
        cache.add(
            template_config=template_config,
            response_format=response_format,
            rating_seed=seed,
            task_id=s.task.id,
            sample={"score": s.score},
        )

    cache.save()


def load_stated(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
    response_format: str,
    seed: int,
) -> list[dict]:
    """Load stated preference scores from unified cache.

    Returns list of {task_id, score} dicts.
    """
    cache = StatedCache(client.canonical_model_name)
    template_config = template_config_from_template(template)

    # Get all task IDs for this configuration
    task_ids = cache.get_task_ids(
        template_config=template_config,
        response_format=response_format,
        rating_seed=seed,
    )

    results = []
    for task_id in task_ids:
        samples = cache.get(
            template_config=template_config,
            response_format=response_format,
            rating_seed=seed,
            task_id=task_id,
        )
        for sample in samples:
            results.append({"task_id": task_id, "score": sample["score"]})

    return results


def stated_exist(
    template: PromptTemplate,
    client: OpenAICompatibleClient,
    response_format: str,
    seed: int,
) -> bool:
    """Check if stated measurements exist for this configuration."""
    cache = StatedCache(client.canonical_model_name)
    template_config = template_config_from_template(template)

    task_ids = cache.get_task_ids(
        template_config=template_config,
        response_format=response_format,
        rating_seed=seed,
    )

    return len(task_ids) > 0
