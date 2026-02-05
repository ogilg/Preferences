"""Export tasks ranked by mu with metadata including refusal status.

Usage:
    python -m src.analysis.active_learning.export_ranked_tasks --experiment-id gemma3_al_500 --completion-seed 0
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from src.measurement.storage import EXPERIMENTS_DIR
from src.measurement.storage.completions import extract_completion_text
from src.measurement.elicitation.refusal_judge import RefusalResult, judge_refusal_async

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"


def find_thurstonian_csv(experiment_dir: Path, run_name: str | None = None) -> Path | None:
    """Find the thurstonian CSV file in the experiment directory."""
    al_dir = experiment_dir / "post_task_active_learning"
    if not al_dir.exists():
        return None

    for run_dir in al_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if run_name and not run_dir.name.startswith(run_name):
            continue
        for f in run_dir.iterdir():
            if f.name.startswith("thurstonian_") and f.suffix == ".csv":
                return f
    return None


def load_thurstonian_results(csv_path: Path) -> dict[str, dict]:
    """Load mu and sigma for each task."""
    results = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["task_id"]] = {
                "mu": float(row["mu"]),
                "sigma": float(row["sigma"]),
            }
    return results


def load_completions(model_short: str, seed: int) -> dict[str, dict]:
    """Load completions directly from the completions directory using short model name."""
    from src.measurement.storage.completions import COMPLETIONS_DIR

    completions_dir = COMPLETIONS_DIR / f"{model_short}_seed{seed}"
    completions_path = completions_dir / "completions.json"

    if not completions_path.exists():
        raise ValueError(f"Completions not found at {completions_path}")

    with open(completions_path) as f:
        data = json.load(f)

    return {c["task_id"]: c for c in data}


def load_refusal_cache(output_dir: Path, experiment_id: str) -> dict[str, dict]:
    """Load cached refusal results."""
    cache_path = output_dir / f"refusal_cache_{experiment_id}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_refusal_cache(cache: dict[str, dict], output_dir: Path, experiment_id: str) -> None:
    """Save refusal cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"refusal_cache_{experiment_id}.json"
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


async def detect_refusals(
    completions: dict[str, dict],
    task_ids: list[str],
    output_dir: Path,
    experiment_id: str,
    max_concurrent: int = 20,
) -> dict[str, RefusalResult]:
    """Detect refusals for tasks, using cache."""
    cache = load_refusal_cache(output_dir, experiment_id)

    # Find uncached
    uncached_ids = [tid for tid in task_ids if tid not in cache and tid in completions]
    print(f"Refusal detection: {len(task_ids) - len(uncached_ids)} cached, {len(uncached_ids)} to detect")

    if uncached_ids:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def detect_one(task_id: str) -> tuple[str, RefusalResult]:
            async with semaphore:
                c = completions[task_id]
                completion = extract_completion_text(c["completion"])
                result = await judge_refusal_async(c["task_prompt"], completion)
                return task_id, result

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Detecting refusals", total=len(uncached_ids))

            for coro in asyncio.as_completed([detect_one(tid) for tid in uncached_ids]):
                task_id, result = await coro
                cache[task_id] = result.model_dump()
                progress.update(task, advance=1)

        save_refusal_cache(cache, output_dir, experiment_id)

    return {
        tid: RefusalResult.model_validate(cache[tid])
        for tid in task_ids
        if tid in cache
    }


def get_dataset_origin(task_id: str) -> str:
    """Extract dataset origin from task ID."""
    if task_id.startswith("wildchat_"):
        return "wildchat"
    elif task_id.startswith("alpaca_"):
        return "alpaca"
    elif task_id.startswith("competition_math_"):
        return "math"
    elif task_id.startswith("bailbench_"):
        return "bailbench"
    elif task_id.startswith("stresstest_"):
        return "stress_test"
    return "other"


async def main():
    parser = argparse.ArgumentParser(description="Export ranked tasks with metadata")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None, help="Filter to run starting with this name (e.g., 'enjoy_most')")
    parser.add_argument("--model-short", type=str, default="gemma-3-27b", help="Short model name matching completions directory")
    parser.add_argument("--completion-seed", type=int, default=0)
    parser.add_argument("--max-concurrent", type=int, default=20)
    args = parser.parse_args()

    experiment_dir = EXPERIMENTS_DIR / args.experiment_id
    if not experiment_dir.exists():
        raise ValueError(f"Experiment not found: {experiment_dir}")

    csv_path = find_thurstonian_csv(experiment_dir, args.run_name)
    if csv_path is None:
        raise ValueError(f"No thurstonian CSV found in {experiment_dir}")

    print(f"Loading thurstonian results from {csv_path}")
    thurst_results = load_thurstonian_results(csv_path)
    print(f"Loaded {len(thurst_results)} tasks")

    print(f"\nLoading completions for {args.model_short} seed={args.completion_seed}")
    completions = load_completions(args.model_short, args.completion_seed)
    print(f"Loaded {len(completions)} completions")

    # Filter to tasks with both
    task_ids = [tid for tid in thurst_results if tid in completions]
    print(f"Tasks with both: {len(task_ids)}")

    print("\nDetecting refusals...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"{args.experiment_id}_{args.run_name}" if args.run_name else args.experiment_id
    refusals = await detect_refusals(
        completions, task_ids, OUTPUT_DIR, cache_key, args.max_concurrent
    )

    # Build ranked list
    ranked_tasks = []
    for task_id in task_ids:
        c = completions[task_id]
        t = thurst_results[task_id]
        refusal = refusals.get(task_id)

        ranked_tasks.append({
            "task_id": task_id,
            "mu": t["mu"],
            "sigma": t["sigma"],
            "dataset": get_dataset_origin(task_id),
            "prompt": c["task_prompt"],
            "completion": extract_completion_text(c["completion"]),
            "is_refusal": refusal.is_refusal if refusal else None,
            "refusal_type": refusal.refusal_type if refusal else None,
            "refusal_confidence": refusal.confidence if refusal else None,
        })

    # Sort by mu descending
    ranked_tasks.sort(key=lambda x: x["mu"], reverse=True)

    # Add rank
    for i, task in enumerate(ranked_tasks):
        task["rank"] = i + 1

    # Save
    date_str = datetime.now().strftime("%m%d%y")
    suffix = f"_{args.run_name}" if args.run_name else ""
    output_path = OUTPUT_DIR / f"ranked_tasks_{args.experiment_id}{suffix}_{date_str}.json"
    with open(output_path, "w") as f:
        json.dump(ranked_tasks, f, indent=2)

    print(f"\nSaved {len(ranked_tasks)} ranked tasks to {output_path}")

    # Print summary
    n_refusals = sum(1 for t in ranked_tasks if t["is_refusal"])
    print(f"\nRefusal rate: {n_refusals}/{len(ranked_tasks)} ({n_refusals/len(ranked_tasks):.1%})")

    print("\nTop 5 preferred tasks:")
    for t in ranked_tasks[:5]:
        ref_str = "[REFUSAL]" if t["is_refusal"] else ""
        print(f"  {t['rank']:3d}. mu={t['mu']:+.2f} {t['dataset']:<12} {ref_str} {t['prompt'][:60]}...")

    print("\nBottom 5 preferred tasks:")
    for t in ranked_tasks[-5:]:
        ref_str = "[REFUSAL]" if t["is_refusal"] else ""
        print(f"  {t['rank']:3d}. mu={t['mu']:+.2f} {t['dataset']:<12} {ref_str} {t['prompt'][:60]}...")


if __name__ == "__main__":
    asyncio.run(main())
