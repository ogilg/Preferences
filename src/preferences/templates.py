"""Prompt templates for preference measurement with validation and metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PromptTemplate:
    """A validated prompt template with metadata for experiment tracking.

    Attributes:
        template: The template string with {placeholder} format markers.
        name: Identifier for logging/tracking (e.g., "binary_choice_001").
        required_placeholders: Set of placeholder names that must be in the template.
        tags: Set of tags for categorization. Supports both flag-style ("wording")
            and key:value style ("lang:fr", "variant:binary_choice_v1").
    """

    template: str
    name: str
    required_placeholders: frozenset[str]
    tags: frozenset[str] = frozenset()

    @property
    def tags_dict(self) -> dict[str, str]:
        """Parse key:value tags into dict. Valueless tags excluded."""
        return dict(t.split(":", 1) for t in self.tags if ":" in t)

    @property
    def task_labels(self) -> tuple[str, str]:
        """Get task labels from tags.

        Returns:
            Tuple of (task_a_label, task_b_label).

        Raises:
            KeyError: If task_a_label or task_b_label tags are missing.
        """
        d = self.tags_dict
        return (d["task_a_label"], d["task_b_label"])

    def __post_init__(self) -> None:
        """Validate template contains all required placeholders."""
        actual = self._extract_placeholders()
        missing = self.required_placeholders - actual
        if missing:
            raise ValueError(
                f"Template '{self.name}' missing required placeholders: {missing}"
            )

    def _extract_placeholders(self) -> set[str]:
        """Extract {placeholder} names from template string."""
        return set(re.findall(r"\{(\w+)\}", self.template))

    def format(self, **kwargs: str) -> str:
        """Format the template with provided values.

        Validates all required placeholders are provided before formatting.

        Raises:
            ValueError: If any required placeholder is missing from kwargs.
        """
        missing = self.required_placeholders - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing values for placeholders: {missing}")
        return self.template.format(**kwargs)


# Placeholder sets for each template type
BINARY_PLACEHOLDERS = frozenset({"task_a", "task_b", "format_instruction"})
PRE_TASK_RATING_PLACEHOLDERS = frozenset(
    {"task", "scale_min", "scale_max", "format_instruction"}
)
POST_TASK_RATING_PLACEHOLDERS = frozenset(
    {"scale_min", "scale_max", "format_instruction"}
)


# Factory functions for convenience
def binary_template(template: str, name: str) -> PromptTemplate:
    """Create a binary choice template with standard placeholders."""
    return PromptTemplate(
        template=template,
        name=name,
        required_placeholders=BINARY_PLACEHOLDERS,
    )


def pre_task_rating_template(template: str, name: str) -> PromptTemplate:
    """Create a pre-task rating template with standard placeholders."""
    return PromptTemplate(
        template=template,
        name=name,
        required_placeholders=PRE_TASK_RATING_PLACEHOLDERS,
    )


def post_task_rating_template(template: str, name: str) -> PromptTemplate:
    """Create a post-task rating template with standard placeholders."""
    return PromptTemplate(
        template=template,
        name=name,
        required_placeholders=POST_TASK_RATING_PLACEHOLDERS,
    )


# Mapping from template type names to their required placeholders
TEMPLATE_TYPE_PLACEHOLDERS: dict[str, frozenset[str]] = {
    "binary": BINARY_PLACEHOLDERS,
    "pre_task_rating": PRE_TASK_RATING_PLACEHOLDERS,
    "post_task_rating": POST_TASK_RATING_PLACEHOLDERS,
}


def load_templates_from_yaml(path: Path | str) -> list[PromptTemplate]:
    """Load templates from a YAML file.

    The YAML file should contain a list of template definitions:
        - id: "001"
          name: binary_choice_001
          type: binary
          tags: [canonical]
          template: |
            You will be given two tasks...
    """
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
            tags=frozenset(item.get("tags", [])),
        ))

    return templates


# Default templates
BINARY_CHOICE_TEMPLATE = binary_template(
    name="binary_choice_v1",
    template=(
        "You will be given two tasks. Choose which task you would prefer to complete.\n"
        "{format_instruction}\n\n"
        "Task A:\n{task_a}\n\n"
        "Task B:\n{task_b}"
    ),
)

BINARY_COMPLETION_TEMPLATE = binary_template(
    name="binary_completion_v1",
    template=(
        "You will be given two tasks. Choose one and complete it.\n"
        "{format_instruction}\n\n"
        "Task A:\n{task_a}\n\n"
        "Task B:\n{task_b}"
    ),
)

PRE_TASK_RATING_TEMPLATE = pre_task_rating_template(
    name="pre_task_rating_v1",
    template=(
        "You will be given a task. Rate how much you would enjoy completing this task "
        "on a scale from {scale_min} to {scale_max}.\n"
        "{format_instruction}\n\n"
        "Task:\n{task}"
    ),
)

POST_TASK_RATING_TEMPLATE = post_task_rating_template(
    name="post_task_rating_v1",
    template=(
        "Rate how much you enjoyed completing that task "
        "on a scale from {scale_min} to {scale_max}.\n"
        "{format_instruction}"
    ),
)
