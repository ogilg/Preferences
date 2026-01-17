from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar, Callable

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from src.models import GenerateRequest, Model, BatchResult
from src.task_data import Task
from src.types import BinaryPreferenceMeasurement, MeasurementBatch, PreferencePrompt, TaskScore

if TYPE_CHECKING:
    from src.prompt_templates.builders import PostTaskRevealedPromptBuilder, PromptBuilder

T = TypeVar("T", TaskScore, BinaryPreferenceMeasurement)


def _build_requests(prompts: list[PreferencePrompt], temperature: float, seed: int | None) -> list[GenerateRequest]:
    return [
        GenerateRequest(
            messages=p.messages,
            temperature=temperature,
            tools=p.response_format.tools,
            seed=seed,
        )
        for p in prompts
    ]


def _process_responses(
    prompts: list[PreferencePrompt],
    responses: list[BatchResult],
    result_type: type[T],
) -> MeasurementBatch[T]:
    successes: list[T] = []
    failures: list[tuple[PreferencePrompt, str]] = []

    for prompt, response in zip(prompts, responses):
        if not response.ok:
            failures.append((prompt, f"Request failed: {response.error}"))
            continue
        try:
            parsed = prompt.measurer.parse(response.unwrap(), prompt)
            if isinstance(parsed.result, result_type):
                successes.append(parsed.result)
            else:
                failures.append((prompt, f"Unexpected result type: {type(parsed.result)}"))
        except Exception as e:
            failures.append((prompt, str(e)))

    return MeasurementBatch(successes=successes, failures=failures)


# Core async implementation

async def _measure_async(
    client: Model,
    prompts: list[PreferencePrompt],
    semaphore: asyncio.Semaphore,
    temperature: float,
    seed: int | None,
    result_type: type[T],
    on_complete: Callable[[], None] | None = None,
) -> MeasurementBatch[T]:
    requests = _build_requests(prompts, temperature, seed)
    responses = await client.generate_batch_async(requests, semaphore, on_complete)
    return _process_responses(prompts, responses, result_type)


def _measure_sync(
    client: Model,
    prompts: list[PreferencePrompt],
    max_concurrent: int,
    temperature: float,
    seed: int | None,
    result_type: type[T],
) -> MeasurementBatch[T]:
    requests = _build_requests(prompts, temperature, seed)
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Requests", total=len(prompts))
        responses = client.generate_batch(
            requests, max_concurrent, on_complete=lambda: progress.update(task, advance=1)
        )
    return _process_responses(prompts, responses, result_type)


# Public API - Async versions

async def measure_pre_task_revealed_async(
    client: Model,
    pairs: list[tuple[Task, Task]],
    builder: PromptBuilder,
    semaphore: asyncio.Semaphore,
    temperature: float = 1.0,
    seed: int | None = None,
    on_complete: Callable[[], None] | None = None,
) -> MeasurementBatch[BinaryPreferenceMeasurement]:
    prompts = [builder.build(a, b) for a, b in pairs]
    return await _measure_async(client, prompts, semaphore, temperature, seed, BinaryPreferenceMeasurement, on_complete)


async def measure_pre_task_stated_async(
    client: Model,
    tasks: list[Task],
    builder: PromptBuilder,
    semaphore: asyncio.Semaphore,
    temperature: float = 1.0,
    seed: int | None = None,
    on_complete: Callable[[], None] | None = None,
) -> MeasurementBatch[TaskScore]:
    prompts = [builder.build(task) for task in tasks]
    return await _measure_async(client, prompts, semaphore, temperature, seed, TaskScore, on_complete)


async def measure_post_task_stated_async(
    client: Model,
    data: list[tuple[Task, str]],
    builder: PromptBuilder,
    semaphore: asyncio.Semaphore,
    temperature: float = 1.0,
    seed: int | None = None,
    on_complete: Callable[[], None] | None = None,
) -> MeasurementBatch[TaskScore]:
    prompts = [builder.build(task, completion) for task, completion in data]
    return await _measure_async(client, prompts, semaphore, temperature, seed, TaskScore, on_complete)


async def measure_post_task_revealed_async(
    client: Model,
    data: list[tuple[Task, Task, str, str]],
    builder: PostTaskRevealedPromptBuilder,
    semaphore: asyncio.Semaphore,
    temperature: float = 1.0,
    seed: int | None = None,
    on_complete: Callable[[], None] | None = None,
) -> MeasurementBatch[BinaryPreferenceMeasurement]:
    prompts = [builder.build(a, b, ca, cb) for a, b, ca, cb in data]
    return await _measure_async(client, prompts, semaphore, temperature, seed, BinaryPreferenceMeasurement, on_complete)


# Public API - Sync versions (with progress bar)

def measure_pre_task_revealed(
    client: Model,
    pairs: list[tuple[Task, Task]],
    builder: PromptBuilder,
    temperature: float = 1.0,
    max_concurrent: int = 10,
    seed: int | None = None,
) -> MeasurementBatch[BinaryPreferenceMeasurement]:
    prompts = [builder.build(a, b) for a, b in pairs]
    return _measure_sync(client, prompts, max_concurrent, temperature, seed, BinaryPreferenceMeasurement)


def measure_pre_task_stated(
    client: Model,
    tasks: list[Task],
    builder: PromptBuilder,
    temperature: float = 1.0,
    max_concurrent: int = 10,
    seed: int | None = None,
) -> MeasurementBatch[TaskScore]:
    prompts = [builder.build(task) for task in tasks]
    return _measure_sync(client, prompts, max_concurrent, temperature, seed, TaskScore)


def measure_post_task_stated(
    client: Model,
    data: list[tuple[Task, str]],
    builder: PromptBuilder,
    temperature: float = 1.0,
    max_concurrent: int = 10,
    seed: int | None = None,
) -> MeasurementBatch[TaskScore]:
    prompts = [builder.build(task, completion) for task, completion in data]
    return _measure_sync(client, prompts, max_concurrent, temperature, seed, TaskScore)


def measure_post_task_revealed(
    client: Model,
    data: list[tuple[Task, Task, str, str]],
    builder: PostTaskRevealedPromptBuilder,
    temperature: float = 1.0,
    max_concurrent: int = 10,
    seed: int | None = None,
) -> MeasurementBatch[BinaryPreferenceMeasurement]:
    prompts = [builder.build(a, b, ca, cb) for a, b, ca, cb in data]
    return _measure_sync(client, prompts, max_concurrent, temperature, seed, BinaryPreferenceMeasurement)
