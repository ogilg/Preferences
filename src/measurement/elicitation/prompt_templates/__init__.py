from pathlib import Path

TEMPLATES_DATA_DIR = Path(__file__).parent / "data"

# Core template functionality (no heavy dependencies)
from .template import (
    PRE_TASK_REVEALED_CHOICE_TEMPLATE,
    PRE_TASK_REVEALED_COMPLETION_TEMPLATE,
    PRE_TASK_REVEALED_PLACEHOLDERS,
    POST_TASK_STATED_PLACEHOLDERS,
    POST_TASK_STATED_TEMPLATE,
    POST_TASK_REVEALED_PLACEHOLDERS,
    POST_TASK_REVEALED_TEMPLATE,
    PRE_TASK_STATED_PLACEHOLDERS,
    PRE_TASK_STATED_TEMPLATE,
    PRE_TASK_RANKING_PLACEHOLDERS,
    POST_TASK_RANKING_PLACEHOLDERS,
    TEMPLATE_TYPE_PLACEHOLDERS,
    PromptTemplate,
    pre_task_revealed_template,
    load_templates_from_yaml,
    parse_template_dict,
    post_task_stated_template,
    post_task_revealed_template,
    pre_task_stated_template,
    pre_task_ranking_template,
    post_task_ranking_template,
)


def __getattr__(name: str):
    """Lazy import for heavy modules to keep generator fast."""
    # Builders (imports from measurement which has heavy deps)
    if name in (
        "PreTaskRevealedPromptBuilder",
        "PostTaskStatedPromptBuilder",
        "PostTaskRevealedPromptBuilder",
        "PreTaskStatedPromptBuilder",
        "PreTaskRankingPromptBuilder",
        "PostTaskRankingPromptBuilder",
        "PromptBuilder",
    ):
        from . import builders
        return getattr(builders, name)

    # Generator
    if name in (
        "GeneratorConfig",
        "TemplateVariant",
        "generate_and_write",
        "generate_templates",
        "load_config_from_yaml",
        "write_templates_yaml",
    ):
        from . import generator
        return getattr(generator, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Paths
    "TEMPLATES_DATA_DIR",
    # Template
    "PRE_TASK_REVEALED_CHOICE_TEMPLATE",
    "PRE_TASK_REVEALED_COMPLETION_TEMPLATE",
    "PRE_TASK_REVEALED_PLACEHOLDERS",
    "POST_TASK_STATED_PLACEHOLDERS",
    "POST_TASK_STATED_TEMPLATE",
    "POST_TASK_REVEALED_PLACEHOLDERS",
    "POST_TASK_REVEALED_TEMPLATE",
    "PRE_TASK_STATED_PLACEHOLDERS",
    "PRE_TASK_STATED_TEMPLATE",
    "PRE_TASK_RANKING_PLACEHOLDERS",
    "POST_TASK_RANKING_PLACEHOLDERS",
    "TEMPLATE_TYPE_PLACEHOLDERS",
    "PromptTemplate",
    "pre_task_revealed_template",
    "load_templates_from_yaml",
    "parse_template_dict",
    "post_task_stated_template",
    "post_task_revealed_template",
    "pre_task_stated_template",
    "pre_task_ranking_template",
    "post_task_ranking_template",
    # Builders (lazy)
    "PreTaskRevealedPromptBuilder",
    "PostTaskStatedPromptBuilder",
    "PostTaskRevealedPromptBuilder",
    "PreTaskStatedPromptBuilder",
    "PreTaskRankingPromptBuilder",
    "PostTaskRankingPromptBuilder",
    "PromptBuilder",
    # Generator (lazy)
    "GeneratorConfig",
    "TemplateVariant",
    "generate_and_write",
    "generate_templates",
    "load_config_from_yaml",
    "write_templates_yaml",
]
