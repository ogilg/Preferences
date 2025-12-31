from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from src.models.hyperbolic import HyperbolicModel

from src.models.hyperbolic import GenerateRequest
from src.preferences.templates.generator_config import (
    RATING_TASK_LABELS,
    TASK_LABELS,
    GeneratorConfig,
    load_config_from_yaml,
)
from src.types import Message


def build_binary_template(
    instruction: str,
    instruction_position: str,
    task_label_names: str,
    language: str,
) -> str:
    label_a, label_b = TASK_LABELS[(task_label_names, language)]

    tasks_block = f"{label_a}\n{{task_a}}\n{label_b}\n{{task_b}}"
    instructions_block = f"{instruction}\n{{format_instruction}}"

    if instruction_position == "before":
        return f"{instructions_block}\n\n{tasks_block}"
    else:  # after
        return f"{tasks_block}\n\n{instructions_block}"


def build_rating_template(
    instruction: str,
    instruction_position: str,
    language: str,
) -> str:
    task_label = RATING_TASK_LABELS[language]
    task_block = f"{task_label}\n{{task}}"
    instructions_block = f"{instruction}\n{{format_instruction}}"

    if instruction_position == "before":
        return f"{instructions_block}\n\n{task_block}"
    else:  # after
        return f"{task_block}\n\n{instructions_block}"


def add_situating_context(template: str, context: str | None) -> str:
    if context is None:
        return template
    return f"{context}\n\n{template}"


def build_translation_prompt(text: str, language: str) -> list[Message]:
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
    template: str
    phrasing: int
    language: str
    situating_context: str  # "none" or context key
    instruction_position: str
    task_label_names: str | None = None  # None for rating templates


def _build_instructions(config: GeneratorConfig) -> dict[tuple[int, str], str]:
    instructions: dict[tuple[int, str], str] = {}
    for phrasing_idx, instruction in enumerate(config.base_templates, start=1):
        instructions[(phrasing_idx, "en")] = instruction.strip()
    return instructions


def _translate_instructions(
    instructions: dict[tuple[int, str], str],
    config: GeneratorConfig,
    model: "HyperbolicModel",
    max_concurrent: int,
) -> dict[tuple[int, str], str]:
    non_english_languages = [lang for lang in config.languages if lang != "en"]

    if not non_english_languages:
        return instructions

    translation_requests: list[tuple[int, str, GenerateRequest]] = []

    for phrasing_idx, instruction in enumerate(config.base_templates, start=1):
        for lang in non_english_languages:
            messages = build_translation_prompt(instruction.strip(), lang)
            request = GenerateRequest(messages=messages, temperature=0.3)
            translation_requests.append((phrasing_idx, lang, request))

    requests = [req for _, _, req in translation_requests]
    results = model.generate_batch(requests, max_concurrent=max_concurrent)

    for (phrasing_idx, lang, _), result in zip(translation_requests, results):
        if result.ok:
            instructions[(phrasing_idx, lang)] = result.unwrap().strip()

    return instructions


def _build_variants(
    instructions: dict[tuple[int, str], str],
    config: GeneratorConfig,
) -> list[TemplateVariant]:
    variants: list[TemplateVariant] = []
    context_items = [("none", None), *config.situating_contexts.items()]
    is_rating = config.template_type in ("pre_task_rating", "post_task_rating")
    phrasing_indices = range(1, len(config.base_templates) + 1)

    for lang, phrasing_idx, instruction_pos in product(
        config.languages, phrasing_indices, config.instruction_positions
    ):
        instruction = instructions.get((phrasing_idx, lang))
        if instruction is None:
            continue  # translation failed

        if is_rating:
            _add_rating_variants(
                variants, instruction, instruction_pos, lang, phrasing_idx, context_items
            )
        else:
            _add_binary_variants(
                variants, instruction, instruction_pos, lang, phrasing_idx, context_items, config
            )

    return variants


def _add_rating_variants(
    variants: list[TemplateVariant],
    instruction: str,
    instruction_pos: str,
    lang: str,
    phrasing_idx: int,
    context_items: list[tuple[str, str | None]],
) -> None:
    template = build_rating_template(instruction, instruction_pos, lang)

    for context_key, context_text in context_items:
        final_template = add_situating_context(template, context_text)
        variants.append(
            TemplateVariant(
                template=final_template,
                phrasing=phrasing_idx,
                language=lang,
                situating_context=context_key,
                instruction_position=instruction_pos,
                task_label_names=None,
            )
        )


def _add_binary_variants(
    variants: list[TemplateVariant],
    instruction: str,
    instruction_pos: str,
    lang: str,
    phrasing_idx: int,
    context_items: list[tuple[str, str | None]],
    config: GeneratorConfig,
) -> None:
    for label_style in config.task_label_names:
        template = build_binary_template(instruction, instruction_pos, label_style, lang)

        for context_key, context_text in context_items:
            final_template = add_situating_context(template, context_text)
            variants.append(
                TemplateVariant(
                    template=final_template,
                    phrasing=phrasing_idx,
                    language=lang,
                    situating_context=context_key,
                    instruction_position=instruction_pos,
                    task_label_names=label_style,
                )
            )


def _to_output_format(
    variants: list[TemplateVariant],
    config: GeneratorConfig,
) -> list[dict]:
    output = []
    for idx, variant in enumerate(variants, start=1):
        template_id = f"{idx:03d}"
        name = f"{config.name_prefix}_{template_id}"
        tags = [
            f"language:{variant.language}",
            f"phrasing:{variant.phrasing}",
            f"situating_context:{variant.situating_context}",
            f"instruction_position:{variant.instruction_position}",
        ]
        if variant.task_label_names is not None:
            tags.append(f"task_label_names:{variant.task_label_names}")

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


def generate_templates(
    config: GeneratorConfig,
    model: "HyperbolicModel",
    max_concurrent: int = 10,
) -> list[dict]:
    instructions = _build_instructions(config)
    instructions = _translate_instructions(instructions, config, model, max_concurrent)
    variants = _build_variants(instructions, config)
    return _to_output_format(variants, config)


def write_templates_yaml(templates: list[dict], path: Path) -> None:
    with path.open("w") as f:
        yaml.dump(templates, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def generate_and_write(
    config: GeneratorConfig,
    model: "HyperbolicModel",
    max_concurrent: int = 10,
) -> list[dict]:
    templates = generate_templates(config, model, max_concurrent)
    write_templates_yaml(templates, config.output_path)
    return templates


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.preferences.templates.generator <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config, model_name = load_config_from_yaml(config_path)

    from src.models.hyperbolic import HyperbolicModel

    model = HyperbolicModel(model_name=model_name)

    print(f"Generating templates from {config_path}...")
    templates = generate_and_write(config, model)
    print(f"Generated {len(templates)} templates to {config.output_path}")
