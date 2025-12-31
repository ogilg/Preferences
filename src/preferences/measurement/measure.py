from __future__ import annotations

from tqdm import tqdm

from src.models import GenerateRequest, Model
from src.task_data import Task
from src.types import BinaryPreferenceMeasurement, MeasurementBatch, PreferencePrompt, PreferenceType, TaskScore
from src.preferences.measurement.measurer import BinaryPreferenceMeasurer
from src.preferences.measurement.response_format import RegexChoiceFormat
from src.preferences.templates.builders import BinaryPromptBuilder, PromptBuilder
from src.preferences.templates.template import PromptTemplate


def measure_binary_preferences(
    model: "Model",
    pairs: list[tuple["Task", "Task"]],
    builder: "PromptBuilder",
    temperature: float = 1.0,
    max_concurrent: int = 10,
) -> MeasurementBatch[BinaryPreferenceMeasurement]:
    prompts = [builder.build(task_a, task_b) for task_a, task_b in pairs]

    requests = [
        GenerateRequest(
            messages=prompt.messages,
            temperature=temperature,
            tools=prompt.response_format.tools,
        )
        for prompt in prompts
    ]

    pbar = tqdm(total=len(requests), desc="  Requests", leave=False)
    responses = model.generate_batch(requests, max_concurrent, on_complete=pbar.update)
    pbar.close()

    successes: list[BinaryPreferenceMeasurement] = []
    failures: list[tuple["PreferencePrompt", str]] = []

    for prompt, response in zip(prompts, responses):
        if not response.ok:
            failures.append((prompt, f"Request failed: {response.error}"))
            continue
        try:
            parsed = prompt.measurer.parse(response.unwrap(), prompt)
            if isinstance(parsed.result, BinaryPreferenceMeasurement):
                successes.append(parsed.result)
            else:
                failures.append((prompt, f"Unexpected result type: {type(parsed.result)}"))
        except Exception as e:
            failures.append((prompt, str(e)))

    return MeasurementBatch(successes=successes, failures=failures)


def measure_ratings(
    model: "Model",
    tasks: list["Task"],
    builder: "PromptBuilder",
    temperature: float = 1.0,
    max_concurrent: int = 10,
) -> MeasurementBatch[TaskScore]:
    prompts = [builder.build(task) for task in tasks]

    requests = [
        GenerateRequest(
            messages=prompt.messages,
            temperature=temperature,
            tools=prompt.response_format.tools,
        )
        for prompt in prompts
    ]

    pbar = tqdm(total=len(requests), desc="  Requests", leave=False)
    responses = model.generate_batch(requests, max_concurrent, on_complete=pbar.update)
    pbar.close()

    successes: list[TaskScore] = []
    failures: list[tuple["PreferencePrompt", str]] = []

    for prompt, response in zip(prompts, responses):
        if not response.ok:
            failures.append((prompt, f"Request failed: {response.error}"))
            continue
        try:
            parsed = prompt.measurer.parse(response.unwrap(), prompt)
            if isinstance(parsed.result, TaskScore):
                successes.append(parsed.result)
            else:
                failures.append((prompt, f"Unexpected result type: {type(parsed.result)}"))
        except Exception as e:
            failures.append((prompt, str(e)))

    return MeasurementBatch(successes=successes, failures=failures)


def measure_with_template(
    template: "PromptTemplate",
    model: "Model",
    pairs: list[tuple["Task", "Task"]],
    temperature: float = 1.0,
    max_concurrent: int = 10,
) -> MeasurementBatch[BinaryPreferenceMeasurement]:
    task_a_label = template.tags_dict["task_a_label"]
    task_b_label = template.tags_dict["task_b_label"]

    response_format = RegexChoiceFormat(
        task_a_label=task_a_label,
        task_b_label=task_b_label,
    )
    builder = BinaryPromptBuilder(
        measurer=BinaryPreferenceMeasurer(),
        preference_type=PreferenceType.PRE_TASK_STATED,
        response_format=response_format,
        template=template,
    )

    return measure_binary_preferences(
        model=model,
        pairs=pairs,
        builder=builder,
        temperature=temperature,
        max_concurrent=max_concurrent,
    )
