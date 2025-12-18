from typing import Any

from ..model import Model
from ..types import PreferencePrompt, MeasurementResponse
from .prompt_builders import PromptBuilder


def run_measurement(
    model: Model,
    builder: PromptBuilder,
    *args: Any,
) -> MeasurementResponse:
    """Run a measurement by building a prompt, generating a response, and parsing it.

    Args:
        model: Model to generate responses.
        builder: PromptBuilder to construct the prompt.
        *args: Arguments to pass to the builder's build method.

    Returns:
        Parsed Response from the model's output.
    """
    prompt = builder.build(*args)
    text = model.generate(prompt.messages)
    return prompt.measurer.parse(text, prompt)


def run_with_prompt(
    model: Model,
    prompt: PreferencePrompt,
) -> MeasurementResponse:
    """Run a measurement with a pre-built prompt.

    Args:
        model: Model to generate responses.
        prompt: Pre-built PreferencePrompt.

    Returns:
        Parsed Response from the model's output.
    """
    text = model.generate(prompt.messages)
    return prompt.measurer.parse(text, prompt)
