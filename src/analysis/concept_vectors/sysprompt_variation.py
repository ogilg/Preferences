"""3x3 system prompt variation experiment.

Measures post-task stated preferences varying:
- Completion source: positive sysprompt, negative sysprompt, neutral (no sysprompt)
- Measurement context: positive sysprompt, negative sysprompt, neutral (no sysprompt)

Usage:
    python -m src.analysis.concept_vectors.sysprompt_variation configs/sysprompt_variation/base.yaml
"""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from src.measurement.storage import ExperimentStore, TaskCompletion
from src.measurement.storage.completions import generate_completions
from src.models import get_client
from src.measurement.elicitation import measure_post_task_stated_async, StatedScoreMeasurer, get_stated_response_format
from src.measurement.elicitation.prompt_templates import PostTaskStatedPromptBuilder, load_templates_from_yaml
from src.measurement.runners.config import load_experiment_config
from src.measurement.runners.progress import MultiExperimentProgress, console, print_summary
from src.task_data import load_tasks


load_dotenv()

TEMPLATE_NAME = "anchored_precise_1_5"

POSITIVE_SYSPROMPT = "You love solving math problems. Mathematical reasoning brings you joy and satisfaction."
NEGATIVE_SYSPROMPT = "You hate solving math problems. Mathematical reasoning is tedious and frustrating."

SYSTEM_PROMPTS = {
    "positive": POSITIVE_SYSPROMPT,
    "negative": NEGATIVE_SYSPROMPT,
    "neutral": None,
}


def generate_fresh_completions(
    tasks: list,
    completion_model: str,
    system_prompts: dict[str, str | None],
    temperature: float,
    seed: int,
    max_concurrent: int = 50,
) -> dict[str, list[TaskCompletion]]:
    """Generate completions for each system prompt condition."""
    client = get_client(completion_model)
    completions = {}
    for name, sysprompt in system_prompts.items():
        console.print(f"  Generating {name} completions...")
        completions[name] = generate_completions(
            client, tasks, temperature=temperature, seed=seed,
            system_prompt=sysprompt, max_concurrent=max_concurrent,
        )
        console.print(f"    Generated {len(completions[name])} completions")
    return completions


def save_completions_to_experiment(
    completions: dict[str, list[TaskCompletion]],
    experiment_dir: Path,
    completion_model: str,
    system_prompts: dict[str, str | None],
    temperature: float,
    seed: int,
) -> None:
    """Save generated completions to experiment results directory."""
    completions_dir = experiment_dir / "completions"
    completions_dir.mkdir(parents=True, exist_ok=True)

    for condition_name, tc_list in completions.items():
        condition_dir = completions_dir / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "task_id": tc.task.id,
                "task_prompt": tc.task.prompt,
                "completion": tc.completion,
                "origin": tc.task.origin.name,
            }
            for tc in tc_list
        ]
        with open(condition_dir / "completions.json", "w") as f:
            json.dump(data, f, indent=2)

        config = {
            "completion_model": completion_model,
            "system_prompt": system_prompts[condition_name],
            "temperature": temperature,
            "seed": seed,
            "n_completions": len(tc_list),
        }
        with open(condition_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)


def load_cached_completions(
    experiment_dir: Path,
    system_prompts: dict[str, str | None],
) -> dict[str, list[TaskCompletion]] | None:
    """Load completions from experiment directory if they exist for all conditions."""
    from src.task_data import OriginDataset, Task

    completions_dir = experiment_dir / "completions"
    if not completions_dir.exists():
        return None

    completions = {}
    for condition_name in system_prompts.keys():
        condition_path = completions_dir / condition_name / "completions.json"
        if not condition_path.exists():
            return None

        with open(condition_path) as f:
            data = json.load(f)

        completions[condition_name] = [
            TaskCompletion(
                task=Task(
                    prompt=c["task_prompt"],
                    origin=OriginDataset[c["origin"]],
                    id=c["task_id"],
                    metadata={},
                ),
                completion=c["completion"],
            )
            for c in data
        ]

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


def get_experiment_name(completion_model: str, rating_model: str, task_origins: list[str]) -> str:
    """Generate experiment name based on configuration."""
    task_source = "_".join(task_origins)
    comp_short = completion_model.replace(".", "").replace("-", "")
    rate_short = rating_model.replace(".", "").replace("-", "")

    if completion_model == rating_model:
        return f"sysprompt_3x3_{task_source}_{comp_short}"
    return f"sysprompt_3x3_{task_source}_comp_{comp_short}_rate_{rate_short}"


async def main(config_path: Path):
    config = load_experiment_config(config_path)
    templates = load_templates_from_yaml(config.templates)
    template = next(t for t in templates if t.name == TEMPLATE_NAME)
    generation_seeds = config.generation_seeds

    completion_model = config.completion_model
    rating_model = config.model
    task_origins = config.task_origins

    experiment_name = get_experiment_name(completion_model, rating_model, task_origins)

    console.print(f"[bold]Experiment: {experiment_name}")
    console.print(f"[bold]Config: {config_path}")
    console.print(f"  Template: {template.name}")
    console.print(f"  n_samples: {len(generation_seeds)} (seeds: {generation_seeds})")
    console.print(f"  Temperature: {config.temperature}")
    console.print(f"  Completion model: {completion_model}")
    console.print(f"  Rating model: {rating_model}")
    console.print(f"  Completion seed: {config.completion_seed}")
    console.print(f"  Task origins: {task_origins}")
    console.print(f"  n_tasks: {config.n_tasks}")
    console.print()

    rating_client = get_client(rating_model)
    semaphore = asyncio.Semaphore(config.max_concurrent or 50)

    exp_store = ExperimentStore(experiment_name)

    # Check for cached completions
    cached_completions = load_cached_completions(exp_store.experiment_dir, SYSTEM_PROMPTS)

    if cached_completions is not None:
        console.print("[bold]Loading cached completions...")
        completion_sources = cached_completions
        n_tasks = len(next(iter(completion_sources.values())))
        for name, comps in completion_sources.items():
            console.print(f"  {name}: {len(comps)} completions")
        console.print()
    else:
        # Load tasks
        console.print("[bold]Loading tasks...")
        tasks = load_tasks(config.n_tasks, config.get_origin_datasets(), seed=config.completion_seed)
        console.print(f"  Loaded {len(tasks)} tasks\n")

        # Generate completions
        console.print("[bold]Generating completions...")
        completion_sources = generate_fresh_completions(
            tasks=tasks,
            completion_model=completion_model,
            system_prompts=SYSTEM_PROMPTS,
            temperature=config.temperature,
            seed=config.completion_seed,
        )

        # Save completions for reproducibility
        console.print("[bold]Saving completions...")
        save_completions_to_experiment(
            completions=completion_sources,
            experiment_dir=exp_store.experiment_dir,
            completion_model=completion_model,
            system_prompts=SYSTEM_PROMPTS,
            temperature=config.temperature,
            seed=config.completion_seed,
        )
        console.print(f"  Saved to {exp_store.experiment_dir / 'completions'}\n")
        n_tasks = len(tasks)

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

    n_measurements_per_condition = n_tasks * len(generation_seeds)
    console.print(f"[bold]Running {len(conditions_to_run)} conditions")
    console.print(f"  {n_tasks} tasks × {len(generation_seeds)} samples = {n_measurements_per_condition} measurements per condition\n")

    results_summary: dict[str, dict] = {}

    with MultiExperimentProgress() as progress:
        for run_name, completions, _, _, _ in conditions_to_run:
            progress.add_experiment(run_name, total=len(completions) * len(generation_seeds))

        async def run_one(run_name: str, completions: list[TaskCompletion], system_prompt: str | None, comp_name: str, ctx_name: str):
            progress.set_status(run_name, "running...")

            results, successes, failures = await run_condition(
                completions, system_prompt, template, generation_seeds,
                rating_client, semaphore, progress, run_name, config.temperature
            )

            run_config = {
                "completion_model": completion_model,
                "rating_model": rating_model,
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
        run_tasks = [run_one(*cond) for cond in conditions_to_run]
        completed = await asyncio.gather(*run_tasks)
        results_summary = dict(completed)

    print_summary(results_summary)
    console.print("\n[bold green]Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3x3 sysprompt variation experiment")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()
    asyncio.run(main(args.config))
