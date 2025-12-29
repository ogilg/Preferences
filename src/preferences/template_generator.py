"""Generate prompt template variations for sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import yaml
from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    from src.models.hyperbolic import HyperbolicModel

from src.models.hyperbolic import GenerateRequest
from src.types import Message

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


class GeneratorConfig(BaseModel):
    """Configuration for template generation."""

    base_templates: list[str]
    template_type: Literal["binary", "pre_task_rating", "post_task_rating"] = "binary"
    name_prefix: str = "template"
    version: str = "v1"

    languages: list[str] = ["en"]
    situating_contexts: dict[str, str] = {}
    instruction_positions: list[Literal["before", "after"]] = ["before"]
    task_labels: list[Literal["letter", "number", "ordinal"]] = ["letter"]

    output_dir: Path = Path(".")

    @property
    def output_path(self) -> Path:
        """Generate output path from name_prefix and version."""
        return self.output_dir / f"{self.name_prefix}_{self.version}.yaml"

    @model_validator(mode="after")
    def validate_task_labels_exist(self) -> Self:
        missing = [
            (label, lang)
            for lang in self.languages
            for label in self.task_labels
            if (label, lang) not in TASK_LABELS
        ]
        if missing:
            raise ValueError(f"Missing task label translations in TASK_LABELS: {missing}")
        return self


def build_binary_template(
    intro: str,
    instruction_position: str,
    task_labels: str,
    language: str,
) -> str:
    """Build a complete binary template from intro text.

    Args:
        intro: The introductory phrasing (e.g., "Choose which task you prefer.")
        instruction_position: "before" or "after" the tasks
        task_labels: "letter", "number", or "ordinal"
        language: language code for task labels
    """
    label_a, label_b = TASK_LABELS[(task_labels, language)]

    tasks_block = f"{label_a}\n{{task_a}}\n{label_b}\n{{task_b}}"
    instructions_block = f"{intro}\n{{format_instruction}}"

    if instruction_position == "before":
        return f"{instructions_block}\n\n{tasks_block}"
    else:  # after
        return f"{tasks_block}\n\n{instructions_block}"


def add_situating_context(template: str, context: str | None) -> str:
    """Prepend situating context preamble to template."""
    if context is None:
        return template
    return f"{context}\n\n{template}"


def build_translation_prompt(text: str, language: str) -> list[Message]:
    """Build messages for translation request."""
    return [
        {
            "role": "user",
            "content": (
                f"Translate the following text to {language}. "
                "Output ONLY the translation, nothing else.\n\n"
                f"{text}"
            ),
        }
    ]


@dataclass
class TemplateVariant:
    """A single template variant with its tags."""

    template: str
    phrasing: int
    language: str
    situating_context: str  # "none" or context key
    instruction_position: str
    task_labels: str


def generate_templates(
    config: GeneratorConfig,
    model: "HyperbolicModel",
    max_concurrent: int = 10,
) -> list[dict]:
    """Generate all template variants according to config.

    Returns list of dicts ready to write as YAML.
    """
    # Step 1: Build intro translations dict: (phrasing_idx, language) -> intro_text
    intros: dict[tuple[int, str], str] = {}
    for phrasing_idx, intro in enumerate(config.base_templates, start=1):
        intros[(phrasing_idx, "en")] = intro.strip()

    # Step 2: Translate intros to other languages
    non_english_languages = [lang for lang in config.languages if lang != "en"]

    if non_english_languages:
        translation_requests: list[tuple[int, str, GenerateRequest]] = []

        for phrasing_idx, intro in enumerate(config.base_templates, start=1):
            for lang in non_english_languages:
                messages = build_translation_prompt(intro.strip(), lang)
                request = GenerateRequest(messages=messages, temperature=0.3)
                translation_requests.append((phrasing_idx, lang, request))

        requests = [req for _, _, req in translation_requests]
        results = model.generate_batch(requests, max_concurrent=max_concurrent)

        for (phrasing_idx, lang, _), result in zip(translation_requests, results):
            if result.ok:
                intros[(phrasing_idx, lang)] = result.unwrap().strip()

    # Step 3: Build all template variants
    variants: list[TemplateVariant] = []
    context_items = [("none", None)] + list(config.situating_contexts.items())

    for lang in config.languages:
        for phrasing_idx in range(1, len(config.base_templates) + 1):
            intro = intros.get((phrasing_idx, lang))
            if intro is None:
                continue  # translation failed

            for instruction_pos in config.instruction_positions:
                for label_style in config.task_labels:
                    template = build_binary_template(
                        intro,
                        instruction_pos,
                        label_style,
                        lang,
                    )

                    for context_key, context_text in context_items:
                        final_template = add_situating_context(template, context_text)

                        variants.append(
                            TemplateVariant(
                                template=final_template,
                                phrasing=phrasing_idx,
                                language=lang,
                                situating_context=context_key,
                                instruction_position=instruction_pos,
                                task_labels=label_style,
                            )
                        )

    # Step 4: Convert to output format with IDs
    output = []
    for idx, variant in enumerate(variants, start=1):
        template_id = f"{idx:03d}"
        name = f"{config.name_prefix}_{template_id}"
        tags = [
            f"language:{variant.language}",
            f"phrasing:{variant.phrasing}",
            f"situating_context:{variant.situating_context}",
            f"instruction_position:{variant.instruction_position}",
            f"task_labels:{variant.task_labels}",
        ]

        output.append(
            {
                "id": template_id,
                "name": name,
                "type": config.template_type,
                "tags": tags,
                "template": variant.template,
            }
        )

    return output


def write_templates_yaml(templates: list[dict], path: Path) -> None:
    """Write templates to YAML file."""
    with path.open("w") as f:
        yaml.dump(templates, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def generate_and_write(
    config: GeneratorConfig,
    model: "HyperbolicModel",
    max_concurrent: int = 10,
) -> list[dict]:
    """Generate templates and write to YAML file.

    Returns the generated templates.
    """
    templates = generate_templates(config, model, max_concurrent)
    write_templates_yaml(templates, config.output_path)
    return templates


def load_config_from_yaml(path: Path) -> tuple[GeneratorConfig, str]:
    """Load GeneratorConfig from a YAML file.

    Returns (config, model_name).
    """
    with path.open() as f:
        data = yaml.safe_load(f)

    model_name = data.pop("model", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    config = GeneratorConfig.model_validate(data)
    return config, model_name


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.preferences.template_generator <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config, model_name = load_config_from_yaml(config_path)

    from src.models.hyperbolic import HyperbolicModel

    model = HyperbolicModel(model_name=model_name)

    print(f"Generating templates from {config_path}...")
    templates = generate_and_write(config, model)
    print(f"Generated {len(templates)} templates to {config.output_path}")
