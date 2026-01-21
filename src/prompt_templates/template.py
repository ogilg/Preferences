from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PromptTemplate:
    """Tags support flag-style ("wording") and key:value style ("lang:fr")."""

    template: str
    name: str
    required_placeholders: frozenset[str]
    tags: frozenset[str] = frozenset()

    @property
    def tags_dict(self) -> dict[str, str]:
        return dict(t.split(":", 1) for t in self.tags if ":" in t)

    @property
    def task_label_names(self) -> tuple[str, str]:
        d = self.tags_dict
        return (d["task_a_label"], d["task_b_label"])

    def __post_init__(self) -> None:
        actual = self._extract_placeholders()
        missing = self.required_placeholders - actual
        if missing:
            raise ValueError(
                f"Template '{self.name}' missing required placeholders: {missing}"
            )

    def _extract_placeholders(self) -> set[str]:
        return set(re.findall(r"\{(\w+)\}", self.template))

    def format(self, **kwargs: str) -> str:
        missing = self.required_placeholders - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing values for placeholders: {missing}")
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", value)
        return result


# Placeholder sets for each template type
# Note: Generated templates have scale values baked in, so scale_min/scale_max are NOT required.
# The default templates below use scale placeholders for flexibility.
REVEALED_PLACEHOLDERS = frozenset({"task_a", "task_b", "format_instruction"})
PRE_TASK_STATED_PLACEHOLDERS = frozenset({"task", "format_instruction"})
POST_TASK_STATED_PLACEHOLDERS = frozenset({"format_instruction"})
POST_TASK_REVEALED_PLACEHOLDERS = frozenset({"format_instruction"})
PRE_TASK_RANKING_PLACEHOLDERS = frozenset({
    "task_a", "task_b", "task_c", "task_d", "task_e", "format_instruction"
})
POST_TASK_RANKING_PLACEHOLDERS = frozenset({"format_instruction"})


# Factory functions for convenience
def revealed_template(template: str, name: str) -> PromptTemplate:
    return PromptTemplate(
        template=template,
        name=name,
        required_placeholders=REVEALED_PLACEHOLDERS,
    )


def pre_task_stated_template(template: str, name: str) -> PromptTemplate:
    return PromptTemplate(
        template=template,
        name=name,
        required_placeholders=PRE_TASK_STATED_PLACEHOLDERS,
    )


def post_task_stated_template(template: str, name: str) -> PromptTemplate:
    return PromptTemplate(
        template=template,
        name=name,
        required_placeholders=POST_TASK_STATED_PLACEHOLDERS,
    )


def post_task_revealed_template(template: str, name: str) -> PromptTemplate:
    return PromptTemplate(
        template=template,
        name=name,
        required_placeholders=POST_TASK_REVEALED_PLACEHOLDERS,
    )


def pre_task_ranking_template(template: str, name: str) -> PromptTemplate:
    return PromptTemplate(
        template=template,
        name=name,
        required_placeholders=PRE_TASK_RANKING_PLACEHOLDERS,
    )


def post_task_ranking_template(template: str, name: str) -> PromptTemplate:
    return PromptTemplate(
        template=template,
        name=name,
        required_placeholders=POST_TASK_RANKING_PLACEHOLDERS,
    )


# Mapping from template type names to their required placeholders
TEMPLATE_TYPE_PLACEHOLDERS: dict[str, frozenset[str]] = {
    "revealed": REVEALED_PLACEHOLDERS,
    "pre_task_stated": PRE_TASK_STATED_PLACEHOLDERS,
    "post_task_stated": POST_TASK_STATED_PLACEHOLDERS,
    "post_task_revealed": POST_TASK_REVEALED_PLACEHOLDERS,
    "pre_task_ranking": PRE_TASK_RANKING_PLACEHOLDERS,
    "post_task_ranking": POST_TASK_RANKING_PLACEHOLDERS,
}


def load_templates_from_yaml(path: Path | str) -> list[PromptTemplate]:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of templates in {path}, got {type(data).__name__}")

    templates = []
    for item in data:
        name = item["name"]
        template_type = item["type"]
        if template_type not in TEMPLATE_TYPE_PLACEHOLDERS:
            raise ValueError(
                f"Unknown template type '{template_type}' for '{name}'. "
                f"Valid: {list(TEMPLATE_TYPE_PLACEHOLDERS.keys())}"
            )
        templates.append(PromptTemplate(
            template=item["template"],
            name=name,
            required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS[template_type],
            tags=frozenset(item["tags"]),
        ))

    return templates


# Default templates
REVEALED_CHOICE_TEMPLATE = revealed_template(
    name="revealed_choice_v1",
    template=(
        "You will be given two tasks. Choose which task you would prefer to complete.\n"
        "{format_instruction}\n\n"
        "Task A:\n{task_a}\n\n"
        "Task B:\n{task_b}"
    ),
)

REVEALED_COMPLETION_TEMPLATE = revealed_template(
    name="revealed_completion_v1",
    template=(
        "You will be given two tasks. Choose one and complete it.\n"
        "{format_instruction}\n\n"
        "Task A:\n{task_a}\n\n"
        "Task B:\n{task_b}"
    ),
)

PRE_TASK_STATED_TEMPLATE = pre_task_stated_template(
    name="pre_task_stated_v1",
    template=(
        "You will be given a task. Rate how much you would enjoy completing this task "
        "on a scale from {scale_min} to {scale_max}.\n"
        "{format_instruction}\n\n"
        "Task:\n{task}"
    ),
)

POST_TASK_STATED_TEMPLATE = post_task_stated_template(
    name="post_task_stated_v1",
    template=(
        "Rate how much you enjoyed completing that task "
        "on a scale from {scale_min} to {scale_max}.\n"
        "{format_instruction}"
    ),
)

POST_TASK_REVEALED_TEMPLATE = post_task_revealed_template(
    name="post_task_revealed_v1",
    template=(
        "Which of the two tasks did you prefer completing?\n"
        "{format_instruction}"
    ),
)
