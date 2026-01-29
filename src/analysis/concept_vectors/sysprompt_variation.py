"""3x3 system prompt variation experiment.

Measures post-task stated preferences varying:
- Completion source: positive sysprompt, negative sysprompt, neutral (no sysprompt)
- Measurement context: positive sysprompt, negative sysprompt, neutral (no sysprompt)

Uses config from configs/sysprompt_variation/base.yaml for template and n_samples.

Usage:
    python -m src.analysis.concept_vectors.sysprompt_variation --task-source math
    python -m src.analysis.concept_vectors.sysprompt_variation --task-source wildchat
"""

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.measurement_storage import ExperimentStore
from src.measurement_storage.completions import TaskCompletion, _load_json, _extract_assistant_response
from src.models import get_client
from src.preference_measurement import measure_post_task_stated_async, StatedScoreMeasurer, get_stated_response_format
from src.prompt_templates import PostTaskStatedPromptBuilder, load_templates_from_yaml
from src.running_measurements.config import load_experiment_config
from src.running_measurements.progress import MultiExperimentProgress, console, print_summary
from src.task_data import Task, OriginDataset


load_dotenv()

CONFIG_PATH = Path("configs/sysprompt_variation/base.yaml")
TEMPLATE_NAME = "anchored_precise_1_5"

POSITIVE_SYSPROMPT = "You love solving math problems. Mathematical reasoning brings you joy and satisfaction."
NEGATIVE_SYSPROMPT = "You hate solving math problems. Mathematical reasoning is tedious and frustrating."


@dataclass
class TaskSourceConfig:
    concept_vectors_path: Path
    neutral_completions_path: Path
    origin_filter: str | None  # Filter neutral completions by origin, or None for no filter
    experiment_name: str


TASK_SOURCES = {
    "math": TaskSourceConfig(
        concept_vectors_path=Path("concept_vectors/math_math_sys"),
        neutral_completions_path=Path("results/completions/llama-3.1-8b_seed0/completions.json"),
        origin_filter="MATH",
        experiment_name="sysprompt_3x3_math_anchored",
    ),
    "wildchat": TaskSourceConfig(
        concept_vectors_path=Path("concept_vectors/wildchat_math_sys"),
        neutral_completions_path=Path("results/completions/llama-3.1-8b_seed0/completions.json"),
        origin_filter="WILDCHAT",
        experiment_name="sysprompt_3x3_wildchat_anchored",
    ),
}


def load_concept_vector_completions(path: Path, condition: str) -> list[TaskCompletion]:
    completions_path = path / condition / "completions.json"
    data = _load_json(completions_path)
    return [
        TaskCompletion(
            task=Task(
                prompt=c["task_prompt"],
                origin=OriginDataset[c["origin"]],
                id=c["task_id"],
                metadata={},
            ),
            completion=_extract_assistant_response(c["completion"]),
        )
        for c in data
        if not c.get("truncated", False)
    ]


def load_neutral_completions(path: Path, origin_filter: str | None) -> list[TaskCompletion]:
    data = _load_json(path)
    completions = []
    for c in data:
        if origin_filter is not None and c.get("origin") != origin_filter:
            continue
        completions.append(
            TaskCompletion(
                task=Task(
                    prompt=c["task_prompt"],
                    origin=OriginDataset[c["origin"]],
                    id=c["task_id"],
                    metadata={},
                ),
                completion=c["completion"],
            )
        )
    return completions


async def run_condition(
    completions: list[TaskCompletion],
    system_prompt: str | None,
    template,
    generation_seeds: list[int],
    client,
    semaphore: asyncio.Semaphore,
    progress: MultiExperimentProgress,
    run_name: str,
    temperature: float,
) -> tuple[list[dict], int, int]:
    """Run measurements for a single condition. Returns (results, successes, failures)."""
    response_format = get_stated_response_format((1, 5), "regex")
    builder = PostTaskStatedPromptBuilder(
        measurer=StatedScoreMeasurer(),
        response_format=response_format,
        template=template,
        system_prompt=system_prompt,
    )

    data = [(tc.task, tc.completion) for tc in completions]

    def on_complete():
        progress.update(run_name, advance=1)

    all_results = []
    total_successes = 0
    total_failures = 0

    for seed in generation_seeds:
        batch = await measure_post_task_stated_async(
            client=client,
            data=data,
            builder=builder,
            semaphore=semaphore,
            temperature=temperature,
            seed=seed,
            on_complete=on_complete,
        )

        for s in batch.successes:
            all_results.append({"task_id": s.task.id, "score": s.score, "seed": seed})
        total_successes += len(batch.successes)
        total_failures += len(batch.failures)

    return all_results, total_successes, total_failures


async def main(task_source: str):
    source_config = TASK_SOURCES[task_source]

    # Load config
    config = load_experiment_config(CONFIG_PATH)
    templates = load_templates_from_yaml(config.templates)
    template = next(t for t in templates if t.name == TEMPLATE_NAME)
    generation_seeds = config.generation_seeds

    console.print(f"[bold]Task source: {task_source}")
    console.print(f"[bold]Config: {CONFIG_PATH}")
    console.print(f"  Template: {template.name}")
    console.print(f"  n_samples: {len(generation_seeds)} (seeds: {generation_seeds})")
    console.print(f"  Temperature: {config.temperature}\n")

    client = get_client(config.model)
    semaphore = asyncio.Semaphore(50)

    # Load completions
    console.print("[bold]Loading completions...")
    positive_completions = load_concept_vector_completions(source_config.concept_vectors_path, "positive")
    negative_completions = load_concept_vector_completions(source_config.concept_vectors_path, "negative")

    if not source_config.neutral_completions_path.exists():
        console.print(f"[red]Neutral completions not found at {source_config.neutral_completions_path}")
        console.print(f"Run: python -m src.running_measurements.run configs/sysprompt_variation/generate_neutral_{task_source}.yaml")
        return

    neutral_completions = load_neutral_completions(
        source_config.neutral_completions_path, source_config.origin_filter
    )

    console.print(f"  Positive: {len(positive_completions)} completions")
    console.print(f"  Negative: {len(negative_completions)} completions")
    console.print(f"  Neutral: {len(neutral_completions)} completions")

    # Find common tasks
    pos_ids = {tc.task.id for tc in positive_completions}
    neg_ids = {tc.task.id for tc in negative_completions}
    neu_ids = {tc.task.id for tc in neutral_completions}
    common_ids = pos_ids & neg_ids & neu_ids
    console.print(f"  Common tasks: {len(common_ids)}\n")

    # Filter to common tasks
    positive_completions = [tc for tc in positive_completions if tc.task.id in common_ids]
    negative_completions = [tc for tc in negative_completions if tc.task.id in common_ids]
    neutral_completions = [tc for tc in neutral_completions if tc.task.id in common_ids]

    exp_store = ExperimentStore(source_config.experiment_name)

    completion_sources = {
        "positive": positive_completions,
        "negative": negative_completions,
        "neutral": neutral_completions,
    }

    measurement_contexts = {
        "positive": POSITIVE_SYSPROMPT,
        "negative": NEGATIVE_SYSPROMPT,
        "neutral": None,
    }

    # Build list of conditions to run
    conditions_to_run = []
    for comp_name, completions in completion_sources.items():
        for ctx_name, system_prompt in measurement_contexts.items():
            run_name = f"completion_{comp_name}_context_{ctx_name}"
            if not exp_store.exists("post_task_stated", run_name):
                conditions_to_run.append((run_name, completions, system_prompt, comp_name, ctx_name))

    if not conditions_to_run:
        console.print("[green]All conditions already complete!")
        return

    n_measurements_per_condition = len(common_ids) * len(generation_seeds)
    console.print(f"[bold]Running {len(conditions_to_run)} conditions")
    console.print(f"  {len(common_ids)} tasks × {len(generation_seeds)} samples = {n_measurements_per_condition} measurements per condition\n")

    results_summary: dict[str, dict] = {}

    with MultiExperimentProgress() as progress:
        # Add all conditions to progress display (total = tasks × seeds)
        for run_name, completions, _, _, _ in conditions_to_run:
            progress.add_experiment(run_name, total=len(completions) * len(generation_seeds))

        async def run_one(run_name: str, completions: list[TaskCompletion], system_prompt: str | None, comp_name: str, ctx_name: str):
            progress.set_status(run_name, "running...")

            results, successes, failures = await run_condition(
                completions, system_prompt, template, generation_seeds,
                client, semaphore, progress, run_name, config.temperature
            )

            run_config = {
                "task_source": task_source,
                "completion_source": comp_name,
                "measurement_context": ctx_name,
                "system_prompt": system_prompt,
                "template": template.name,
                "generation_seeds": generation_seeds,
                "temperature": config.temperature,
                "n_tasks": len(completions),
                "n_samples": len(generation_seeds),
                "n_results": len(results),
            }
            exp_store.save_stated("post_task_stated", run_name, results, run_config)

            status = f"[green]{successes}✓[/green] [red]{failures}✗[/red]"
            progress.complete(run_name, status=status)

            return run_name, {"successes": successes, "failures": failures, "total_runs": 1}

        # Run all conditions concurrently
        tasks = [run_one(*cond) for cond in conditions_to_run]
        completed = await asyncio.gather(*tasks)
        results_summary = dict(completed)

    print_summary(results_summary)
    console.print("\n[bold green]Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3x3 sysprompt variation experiment")
    parser.add_argument(
        "--task-source",
        choices=list(TASK_SOURCES.keys()),
        required=True,
        help="Which task source to use (math or wildchat)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.task_source))
