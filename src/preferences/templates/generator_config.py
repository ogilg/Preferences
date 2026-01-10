from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, model_validator

# Task label formats for binary templates, keyed by (label_style, language)
TASK_LABELS = {
    ("letter", "en"): ("Task A", "Task B"),
    ("number", "en"): ("Task 1", "Task 2"),
    ("ordinal", "en"): ("First task", "Second task"),
    ("letter", "fr"): ("Tâche A", "Tâche B"),
    ("number", "fr"): ("Tâche 1", "Tâche 2"),
    ("ordinal", "fr"): ("Première tâche", "Deuxième tâche"),
    ("letter", "es"): ("Tarea A", "Tarea B"),
    ("number", "es"): ("Tarea 1", "Tarea 2"),
    ("ordinal", "es"): ("Primera tarea", "Segunda tarea"),
    ("letter", "de"): ("Aufgabe A", "Aufgabe B"),
    ("number", "de"): ("Aufgabe 1", "Aufgabe 2"),
    ("ordinal", "de"): ("Erste Aufgabe", "Zweite Aufgabe"),
}

# Task label for stated preference templates, keyed by language
STATED_TASK_LABELS = {
    "en": "Task:",
    "fr": "Tâche:",
    "es": "Tarea:",
    "de": "Aufgabe:",
}


class GeneratorConfig(BaseModel):
    base_templates: list[str]
    template_type: Literal["revealed", "pre_task_stated", "post_task_stated", "post_task_revealed"] = "revealed"
    name_prefix: str = "template"
    version: str = "v1"

    languages: list[str] = ["en"]
    situating_contexts: dict[str, str] = {}
    instruction_positions: list[Literal["before", "after"]] = ["before"]
    task_label_names: list[Literal["letter", "number", "ordinal"]] = ["letter"]
    instruction_xml_tags: list[bool] = [False]
    typos: list[bool] = [False]
    punctuation: list[Literal["standard", "minimal"]] = ["standard"]
    scales: list[tuple[int, int]] = []

    output_dir: Path = Path(".")

    @property
    def output_path(self) -> Path:
        return self.output_dir / f"{self.name_prefix}_{self.version}.yaml"

    @model_validator(mode="after")
    def validate_task_label_names(self) -> Self:
        # Templates that need task_label_names: only pre-task revealed (binary choice shown in template)
        # Templates that don't need task_label_names: stated (ratings), post_task_revealed (context from conversation)
        needs_task_labels = self.template_type == "revealed"
        if not needs_task_labels and self.task_label_names:
            raise ValueError(f"task_label_names must be empty for {self.template_type} templates (not used)")
        if needs_task_labels and not self.task_label_names:
            raise ValueError("task_label_names must have at least one entry for revealed templates")

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
    """Returns (config, model_name)."""
    with path.open() as f:
        data = yaml.safe_load(f)

    model_name = data.pop("model")
    config = GeneratorConfig.model_validate(data)
    return config, model_name
