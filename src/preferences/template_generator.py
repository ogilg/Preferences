"""Generate prompt template variations for sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from src.models.hyperbolic import HyperbolicModel

from src.models.hyperbolic import GenerateRequest
from src.types import Message


@dataclass
class GeneratorConfig:
    """Configuration for template generation."""

    base_templates: list[str]
    template_type: str  # "binary" | "pre_task_rating" | "post_task_rating"
    name_prefix: str

    languages: list[str] = field(default_factory=lambda: ["en"])
    situating_contexts: dict[str, str] = field(default_factory=dict)
    instruction_positions: list[str] = field(default_factory=lambda: ["before"])
    task_labels: list[str] = field(default_factory=lambda: ["letter"])

    output_path: Path = field(default_factory=lambda: Path("templates.yaml"))


# Task label formats for binary templates
TASK_LABELS = {
    "letter": ("Task A:", "Task B:"),
    "number": ("Task 1:", "Task 2:"),
    "ordinal": ("First task:", "Second task:"),
}


def build_binary_template(
    intro: str,
    instruction_position: str,
    task_labels: str,
) -> str:
    """Build a complete binary template from intro text.

    Args:
        intro: The introductory phrasing (e.g., "Choose which task you prefer.")
        instruction_position: "before" or "after" the tasks
        task_labels: "letter", "number", or "ordinal"
    """
    label_a, label_b = TASK_LABELS[task_labels]

    tasks_block = f"{label_a}\n{{task_a}}\n\n{label_b}\n{{task_b}}"

    if instruction_position == "before":
        return f"{intro}\n{{format_instruction}}\n\n{tasks_block}"
    else:  # after
        return f"{intro}\n\n{tasks_block}\n\n{{format_instruction}}"


def add_situating_context(template: str, context: str | None) -> str:
    """Prepend situating context preamble to template."""
    if context is None:
        return template
    return f"{context}\n\n{template}"


def build_translation_prompt(template: str, language: str) -> list[Message]:
    """Build messages for translation request."""
    return [
        {
            "role": "user",
            "content": (
                f"Translate the following prompt template to {language}.\n"
                "Preserve all placeholders exactly as written: {task_a}, {task_b}, "
                "{format_instruction}, {task}, {scale_min}, {scale_max}, etc.\n"
                "Only translate the natural language text.\n\n"
                f"Template:\n{template}"
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
    variants: list[TemplateVariant] = []

    # Step 1: Build structural variants for each base template (intro text)
    for phrasing_idx, intro in enumerate(config.base_templates, start=1):
        for instruction_pos in config.instruction_positions:
            for label_style in config.task_labels:
                # Build complete template from intro
                template = build_binary_template(
                    intro.strip(),
                    instruction_pos,
                    label_style,
                )

                # Step 2: Add situating contexts (including "none")
                context_items = [("none", None)] + list(config.situating_contexts.items())

                for context_key, context_text in context_items:
                    final_template = add_situating_context(template, context_text)

                    variants.append(
                        TemplateVariant(
                            template=final_template,
                            phrasing=phrasing_idx,
                            language="en",
                            situating_context=context_key,
                            instruction_position=instruction_pos,
                            task_labels=label_style,
                        )
                    )

    # Step 3: Translate to other languages
    non_english_languages = [lang for lang in config.languages if lang != "en"]

    if non_english_languages:
        # Build translation requests for all variants x languages
        translation_requests: list[tuple[TemplateVariant, str, GenerateRequest]] = []

        for variant in variants:
            for lang in non_english_languages:
                messages = build_translation_prompt(variant.template, lang)
                request = GenerateRequest(messages=messages, temperature=0.3)
                translation_requests.append((variant, lang, request))

        # Batch translate
        requests = [req for _, _, req in translation_requests]
        results = model.generate_batch(requests, max_concurrent=max_concurrent)

        # Create translated variants
        for (original_variant, lang, _), result in zip(translation_requests, results):
            if result.ok:
                translated_template = result.unwrap()
                variants.append(
                    TemplateVariant(
                        template=translated_template,
                        phrasing=original_variant.phrasing,
                        language=lang,
                        situating_context=original_variant.situating_context,
                        instruction_position=original_variant.instruction_position,
                        task_labels=original_variant.task_labels,
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

    config = GeneratorConfig(
        base_templates=data["base_templates"],
        template_type=data.get("template_type", "binary"),
        name_prefix=data.get("name_prefix", "template"),
        languages=data.get("languages", ["en"]),
        situating_contexts=data.get("situating_contexts", {}),
        instruction_positions=data.get("instruction_positions", ["before"]),
        task_labels=data.get("task_labels", ["letter"]),
        output_path=Path(data.get("output_path", "generated_templates.yaml")),
    )

    model_name = data.get("model", "meta-llama/Meta-Llama-3.1-8B-Instruct")
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
