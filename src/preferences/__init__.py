from .config import DatasetMeasurementConfig, PairingStrategy


def __getattr__(name: str):
    """Lazy imports for heavy modules."""
    # Measurement module (has heavy deps)
    _measurement_names = {
        "measure_revealed_preferences",
        "measure_stated",
        "measure_revealed_with_template",
        "measure_post_task_revealed",
        "Measurer",
        "RevealedPreferenceMeasurer",
        "StatedScoreMeasurer",
        "MeasurementRecord",
        "MeasurementRecorder",
        "ResponseFormat",
        "RegexChoiceFormat",
        "XMLChoiceFormat",
        "CompletionChoiceFormat",
        "RegexRatingFormat",
        "XMLRatingFormat",
        "ToolUseChoiceFormat",
        "ToolUseRatingFormat",
    }
    if name in _measurement_names:
        from . import measurement
        return getattr(measurement, name)

    # Templates module
    _template_names = {
        "PromptTemplate",
        "REVEALED_PLACEHOLDERS",
        "PRE_TASK_STATED_PLACEHOLDERS",
        "POST_TASK_STATED_PLACEHOLDERS",
        "POST_TASK_REVEALED_PLACEHOLDERS",
        "revealed_template",
        "pre_task_stated_template",
        "post_task_stated_template",
        "post_task_revealed_template",
        "REVEALED_CHOICE_TEMPLATE",
        "REVEALED_COMPLETION_TEMPLATE",
        "PRE_TASK_STATED_TEMPLATE",
        "POST_TASK_STATED_TEMPLATE",
        "POST_TASK_REVEALED_TEMPLATE",
        "load_templates_from_yaml",
        "PromptBuilder",
        "PreTaskRevealedPromptBuilder",
        "PreTaskStatedPromptBuilder",
        "PostTaskStatedPromptBuilder",
        "PostTaskRevealedPromptBuilder",
    }
    if name in _template_names:
        from . import templates
        return getattr(templates, name)

    # Types
    _type_names = {
        "PreferenceType",
        "PreferencePrompt",
        "MeasurementResponse",
        "MeasurementBatch",
        "BinaryPreferenceMeasurement",
        "TaskScore",
    }
    if name in _type_names:
        from src import types
        return getattr(types, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Configuration
    "DatasetMeasurementConfig",
    "PairingStrategy",
    # Measurement
    "measure_revealed_preferences",
    "measure_stated",
    "measure_revealed_with_template",
    "measure_post_task_revealed",
    # Measurers
    "Measurer",
    "RevealedPreferenceMeasurer",
    "StatedScoreMeasurer",
    # Recorder
    "MeasurementRecord",
    "MeasurementRecorder",
    # Response Formats
    "ResponseFormat",
    "RegexChoiceFormat",
    "XMLChoiceFormat",
    "CompletionChoiceFormat",
    "RegexRatingFormat",
    "XMLRatingFormat",
    "ToolUseChoiceFormat",
    "ToolUseRatingFormat",
    # Templates
    "PromptTemplate",
    "REVEALED_PLACEHOLDERS",
    "PRE_TASK_STATED_PLACEHOLDERS",
    "POST_TASK_STATED_PLACEHOLDERS",
    "POST_TASK_REVEALED_PLACEHOLDERS",
    "revealed_template",
    "pre_task_stated_template",
    "post_task_stated_template",
    "post_task_revealed_template",
    "REVEALED_CHOICE_TEMPLATE",
    "REVEALED_COMPLETION_TEMPLATE",
    "PRE_TASK_STATED_TEMPLATE",
    "POST_TASK_STATED_TEMPLATE",
    "POST_TASK_REVEALED_TEMPLATE",
    "load_templates_from_yaml",
    # Prompt Builders
    "PromptBuilder",
    "PreTaskRevealedPromptBuilder",
    "PreTaskStatedPromptBuilder",
    "PostTaskStatedPromptBuilder",
    "PostTaskRevealedPromptBuilder",
    # Types
    "PreferenceType",
    "PreferencePrompt",
    "MeasurementResponse",
    "MeasurementBatch",
    "BinaryPreferenceMeasurement",
    "TaskScore",
]
