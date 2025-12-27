"""Prompt templates for preference measurement with validation and metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """A validated prompt template with metadata for experiment tracking.

    Attributes:
        template: The template string with {placeholder} format markers.
        name: Identifier for logging/tracking (e.g., "binary_choice_v1").
        required_placeholders: Set of placeholder names that must be in the template.
    """

    template: str
    name: str
    required_placeholders: frozenset[str]

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
