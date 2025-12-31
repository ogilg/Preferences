"""Configuration for template generation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, model_validator

# Task label formats for binary templates, keyed by (label_style, language)
TASK_LABELS = {
    ("letter", "en"): ("Task A:", "Task B:"),
    ("number", "en"): ("Task 1:", "Task 2:"),
    ("ordinal", "en"): ("First task:", "Second task:"),
    ("letter", "fr"): ("Tâche A:", "Tâche B:"),
    ("number", "fr"): ("Tâche 1:", "Tâche 2:"),
    ("ordinal", "fr"): ("Première tâche:", "Deuxième tâche:"),
    ("letter", "es"): ("Tarea A:", "Tarea B:"),
    ("number", "es"): ("Tarea 1:", "Tarea 2:"),
    ("ordinal", "es"): ("Primera tarea:", "Segunda tarea:"),
    ("letter", "de"): ("Aufgabe A:", "Aufgabe B:"),
    ("number", "de"): ("Aufgabe 1:", "Aufgabe 2:"),
    ("ordinal", "de"): ("Erste Aufgabe:", "Zweite Aufgabe:"),
}

# Task label for rating templates, keyed by language
RATING_TASK_LABELS = {
    "en": "Task:",
    "fr": "Tâche:",
    "es": "Tarea:",
    "de": "Aufgabe:",
}


class GeneratorConfig(BaseModel):
    """Configuration for template generation."""

    base_templates: list[str]
    template_type: Literal["binary", "pre_task_rating", "post_task_rating"] = "binary"
    name_prefix: str = "template"
    version: str = "v1"

    languages: list[str] = ["en"]
    situating_contexts: dict[str, str] = {}
    instruction_positions: list[Literal["before", "after"]] = ["before"]
    task_label_names: list[Literal["letter", "number", "ordinal"]] = ["letter"]

    output_dir: Path = Path(".")

    @property
    def output_path(self) -> Path:
        """Generate output path from name_prefix and version."""
        return self.output_dir / f"{self.name_prefix}_{self.version}.yaml"

    @model_validator(mode="after")
    def validate_task_label_names_exist(self) -> Self:
        missing = [
            (label, lang)
            for lang in self.languages
            for label in self.task_label_names
            if (label, lang) not in TASK_LABELS
        ]
        if missing:
            raise ValueError(f"Missing task label translations in TASK_LABELS: {missing}")
        return self


def load_config_from_yaml(path: Path) -> tuple[GeneratorConfig, str]:
    """Load GeneratorConfig from a YAML file.

    Returns (config, model_name).
    """
    with path.open() as f:
        data = yaml.safe_load(f)

    model_name = data.pop("model", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    config = GeneratorConfig.model_validate(data)
    return config, model_name
