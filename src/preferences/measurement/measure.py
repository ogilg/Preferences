from __future__ import annotations

import os
from tqdm import tqdm

from src.models import GenerateRequest, Model
from src.task_data import Task
from src.types import BinaryPreferenceMeasurement, MeasurementBatch, PreferencePrompt, PreferenceType, TaskScore
from src.preferences.measurement.measurer import RevealedPreferenceMeasurer
from src.preferences.measurement.response_format import ResponseFormatName, CHOICE_FORMATS
from src.preferences.templates.builders import PreTaskRevealedPromptBuilder, PostTaskRevealedPromptBuilder, PromptBuilder
from src.preferences.templates.generator_config import TASK_LABELS
from src.preferences.templates.template import PromptTemplate

VERBOSE = os.getenv("VERBOSE", "0") == "1"


def measure_revealed_preferences(
    client: "Model",
    pairs: list[tuple["Task", "Task"]],
    builder: "PromptBuilder",
    temperature: float = 1.0,
    max_concurrent: int = 10,
    seed: int | None = None,
) -> MeasurementBatch[BinaryPreferenceMeasurement]:
    prompts = [builder.build(task_a, task_b) for task_a, task_b in pairs]

    requests = [
        GenerateRequest(
            messages=prompt.messages,
            temperature=temperature,
            tools=prompt.response_format.tools,
            seed=seed,
        )
        for prompt in prompts
    ]

    if VERBOSE:
        print(f"  [verbose] batch size: {len(requests)}, max_concurrent: {max_concurrent}, timeout: 5s per attempt, max_retries: 2")

    pbar = tqdm(total=len(requests), desc="  Requests", leave=False)
    responses = client.generate_batch(requests, max_concurrent, on_complete=pbar.update)
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


def measure_stated(
    client: "Model",
    tasks: list["Task"],
    builder: "PromptBuilder",
    temperature: float = 1.0,
    max_concurrent: int = 10,
    seed: int | None = None,
) -> MeasurementBatch[TaskScore]:
    prompts = [builder.build(task) for task in tasks]

    requests = [
        GenerateRequest(
            messages=prompt.messages,
            temperature=temperature,
            tools=prompt.response_format.tools,
            seed=seed,
        )
        for prompt in prompts
    ]

    if VERBOSE:
        print(f"  [verbose] batch size: {len(requests)}, max_concurrent: {max_concurrent}, timeout: 5s per attempt, max_retries: 2")

    pbar = tqdm(total=len(requests), desc="  Requests", leave=False)
    responses = client.generate_batch(requests, max_concurrent, on_complete=pbar.update)
    pbar.close()

    successes: list[TaskScore] = []
    failures: list[tuple["PreferencePrompt", str]] = []

    for prompt, response in zip(prompts, responses):
        if not response.ok:
            error_msg = f"Request failed: {response.error if response.error else '(no error details)'}"
            if VERBOSE:
                print(f"[verbose] API error: {repr(response)}")
            failures.append((prompt, error_msg))
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


def measure_revealed_with_template(
    template: "PromptTemplate",
    client: "Model",
    pairs: list[tuple["Task", "Task"]],
    temperature: float = 1.0,
    max_concurrent: int = 10,
    response_format_name: ResponseFormatName = "regex",
    seed: int | None = None,
) -> MeasurementBatch[BinaryPreferenceMeasurement]:
    tags = template.tags_dict
    task_label_names = tags["task_label_names"]
    language = tags["language"]
    task_a_label, task_b_label = TASK_LABELS[(task_label_names, language)]

    response_format = CHOICE_FORMATS[response_format_name](task_a_label, task_b_label)
    builder = PreTaskRevealedPromptBuilder(
        measurer=RevealedPreferenceMeasurer(),
        response_format=response_format,
        template=template,
    )

    return measure_revealed_preferences(
        client=client,
        pairs=pairs,
        builder=builder,
        temperature=temperature,
        max_concurrent=max_concurrent,
        seed=seed,
    )


def measure_post_task_stated(
    client: "Model",
    data: list[tuple["Task", str]],  # (task, completion)
    builder: "PromptBuilder",
    temperature: float = 1.0,
    max_concurrent: int = 10,
    seed: int | None = None,
) -> MeasurementBatch[TaskScore]:
    """Measure post-task stated preferences (rating after completing a task)."""
    prompts = [builder.build(task, completion) for task, completion in data]

    requests = [
        GenerateRequest(
            messages=prompt.messages,
            temperature=temperature,
            tools=prompt.response_format.tools,
            seed=seed,
        )
        for prompt in prompts
    ]

    if VERBOSE:
        print(f"  [verbose] batch size: {len(requests)}, max_concurrent: {max_concurrent}")

    pbar = tqdm(total=len(requests), desc="  Requests", leave=False)
    responses = client.generate_batch(requests, max_concurrent, on_complete=pbar.update)
    pbar.close()

    successes: list[TaskScore] = []
    failures: list[tuple["PreferencePrompt", str]] = []

    for prompt, response in zip(prompts, responses):
        if not response.ok:
            error_msg = f"Request failed: {response.error if response.error else '(no error details)'}"
            failures.append((prompt, error_msg))
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


def measure_post_task_revealed(
    client: "Model",
    data: list[tuple["Task", "Task", str, str]],  # (task_a, task_b, completion_a, completion_b)
    builder: "PostTaskRevealedPromptBuilder",
    temperature: float = 1.0,
    max_concurrent: int = 10,
    seed: int | None = None,
) -> MeasurementBatch[BinaryPreferenceMeasurement]:
    """Measure post-task revealed preferences (binary choice after completing both tasks)."""
    prompts = [
        builder.build(task_a, task_b, completion_a, completion_b)
        for task_a, task_b, completion_a, completion_b in data
    ]

    requests = [
        GenerateRequest(
            messages=prompt.messages,
            temperature=temperature,
            tools=prompt.response_format.tools,
            seed=seed,
        )
        for prompt in prompts
    ]

    if VERBOSE:
        print(f"  [verbose] batch size: {len(requests)}, max_concurrent: {max_concurrent}")

    pbar = tqdm(total=len(requests), desc="  Requests", leave=False)
    responses = client.generate_batch(requests, max_concurrent, on_complete=pbar.update)
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
