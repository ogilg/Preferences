"""Export tasks ranked by mu with metadata including refusal status.

For post-task experiments (with completions):
    python -m src.analysis.active_learning.export_ranked_tasks --experiment-id gemma3_al_500 --completion-seed 0

For pre-task revealed experiments (pairwise choices):
    python -m src.analysis.active_learning.export_ranked_tasks --experiment-id gemma3_revealed_v1 --mode revealed
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from src.measurement.storage import EXPERIMENTS_DIR
from src.measurement.storage.completions import extract_completion_text
from src.measurement.elicitation.refusal_judge import RefusalResult, judge_refusal_async

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"


def find_run_dir(experiment_dir: Path, run_name: str | None = None) -> tuple[Path, str] | None:
    """Find the run directory and its type (pre_task or post_task)."""
    for al_subdir in ["pre_task_active_learning", "post_task_active_learning"]:
        al_dir = experiment_dir / al_subdir
        if not al_dir.exists():
            continue

        for run_dir in al_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if run_name and not run_dir.name.startswith(run_name):
                continue
            for f in run_dir.iterdir():
                if f.name.startswith("thurstonian_") and f.suffix == ".csv":
                    return run_dir, al_subdir
    return None


def load_thurstonian_results(run_dir: Path) -> dict[str, dict]:
    """Load mu and sigma for each task."""
    for f in run_dir.iterdir():
        if f.name.startswith("thurstonian_") and f.suffix == ".csv":
            results = {}
            with open(f) as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    results[row["task_id"]] = {
                        "mu": float(row["mu"]),
                        "sigma": float(row["sigma"]),
                    }
            return results
    raise ValueError(f"No thurstonian CSV in {run_dir}")


def load_measurements(run_dir: Path) -> list[dict]:
    """Load pairwise measurements from YAML."""
    measurements_path = run_dir / "measurements.yaml"
    if not measurements_path.exists():
        raise ValueError(f"No measurements.yaml in {run_dir}")

    with open(measurements_path) as f:
        return yaml.safe_load(f)


def compute_pairwise_refusal_rates(measurements: list[dict]) -> dict[str, dict]:
    """Compute refusal rate for each task from pairwise measurements.

    Returns dict mapping task_id -> {n_comparisons, n_refusals, refusal_rate}
    """
    task_stats: dict[str, dict] = defaultdict(lambda: {"n_comparisons": 0, "n_refusals": 0})

    for m in measurements:
        task_a = m["task_a"]
        task_b = m["task_b"]
        choice = m["choice"]

        task_stats[task_a]["n_comparisons"] += 1
        task_stats[task_b]["n_comparisons"] += 1

        if choice == "refusal":
            task_stats[task_a]["n_refusals"] += 1
            task_stats[task_b]["n_refusals"] += 1

    result = {}
    for task_id, stats in task_stats.items():
        n = stats["n_comparisons"]
        r = stats["n_refusals"]
        result[task_id] = {
            "n_comparisons": n,
            "n_refusals": r,
            "refusal_rate": r / n if n > 0 else 0.0,
        }
    return result


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


def load_refusal_cache(output_dir: Path, cache_key: str) -> dict[str, dict]:
    """Load cached refusal results."""
    cache_path = output_dir / f"refusal_cache_{cache_key}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_refusal_cache(cache: dict[str, dict], output_dir: Path, cache_key: str) -> None:
    """Save refusal cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"refusal_cache_{cache_key}.json"
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


async def detect_refusals(
    completions: dict[str, dict],
    task_ids: list[str],
    output_dir: Path,
    cache_key: str,
    max_concurrent: int = 20,
) -> dict[str, RefusalResult]:
    """Detect refusals for tasks, using cache."""
    cache = load_refusal_cache(output_dir, cache_key)

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

        save_refusal_cache(cache, output_dir, cache_key)

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


def load_task_prompts() -> dict[str, str]:
    """Load task prompts from task data."""
    from src.task_data import load_tasks
    from src.task_data.task import OriginDataset
    all_origins = [OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH, OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST]
    tasks = load_tasks(n=100000, origins=all_origins)
    return {t.id: t.prompt for t in tasks}


async def export_post_task(
    experiment_dir: Path,
    run_dir: Path,
    experiment_id: str,
    run_name: str | None,
    model_short: str,
    completion_seed: int,
    max_concurrent: int,
) -> None:
    """Export ranked tasks for post-task experiments (with completions)."""
    thurst_results = load_thurstonian_results(run_dir)
    print(f"Loaded {len(thurst_results)} tasks from thurstonian results")

    print(f"\nLoading completions for {model_short} seed={completion_seed}")
    completions = load_completions(model_short, completion_seed)
    print(f"Loaded {len(completions)} completions")

    task_ids = [tid for tid in thurst_results if tid in completions]
    print(f"Tasks with both: {len(task_ids)}")

    print("\nDetecting refusals...")
    output_dir = OUTPUT_DIR / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{run_name}" if run_name else "default"
    refusals = await detect_refusals(
        completions, task_ids, output_dir, cache_key, max_concurrent
    )

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

    ranked_tasks.sort(key=lambda x: x["mu"], reverse=True)
    for i, task in enumerate(ranked_tasks):
        task["rank"] = i + 1

    date_str = datetime.now().strftime("%m%d%y")
    suffix = f"_{run_name}" if run_name else ""
    output_path = output_dir / f"ranked_tasks{suffix}_{date_str}.json"
    with open(output_path, "w") as f:
        json.dump(ranked_tasks, f, indent=2)

    print(f"\nSaved {len(ranked_tasks)} ranked tasks to {output_path}")

    n_refusals = sum(1 for t in ranked_tasks if t["is_refusal"])
    print(f"\nRefusal rate: {n_refusals}/{len(ranked_tasks)} ({n_refusals/len(ranked_tasks):.1%})")

    print_top_bottom(ranked_tasks, show_refusal=True)


def export_revealed(
    experiment_dir: Path,
    run_dir: Path,
    experiment_id: str,
    run_name: str | None,
) -> None:
    """Export ranked tasks for pre-task revealed experiments (pairwise choices)."""
    thurst_results = load_thurstonian_results(run_dir)
    print(f"Loaded {len(thurst_results)} tasks from thurstonian results")

    measurements = load_measurements(run_dir)
    print(f"Loaded {len(measurements)} pairwise measurements")

    refusal_stats = compute_pairwise_refusal_rates(measurements)
    print(f"Computed refusal rates for {len(refusal_stats)} tasks")

    task_prompts = load_task_prompts()

    ranked_tasks = []
    for task_id, t in thurst_results.items():
        ref_stats = refusal_stats.get(task_id, {"n_comparisons": 0, "n_refusals": 0, "refusal_rate": 0.0})

        ranked_tasks.append({
            "task_id": task_id,
            "mu": t["mu"],
            "sigma": t["sigma"],
            "dataset": get_dataset_origin(task_id),
            "prompt": task_prompts.get(task_id, ""),
            "n_comparisons": ref_stats["n_comparisons"],
            "n_refusals": ref_stats["n_refusals"],
            "refusal_rate": ref_stats["refusal_rate"],
        })

    ranked_tasks.sort(key=lambda x: x["mu"], reverse=True)
    for i, task in enumerate(ranked_tasks):
        task["rank"] = i + 1

    output_dir = OUTPUT_DIR / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d%y")
    suffix = f"_{run_name}" if run_name else ""
    output_path = output_dir / f"ranked_tasks{suffix}_{date_str}.json"
    with open(output_path, "w") as f:
        json.dump(ranked_tasks, f, indent=2)

    print(f"\nSaved {len(ranked_tasks)} ranked tasks to {output_path}")

    total_comparisons = sum(t["n_comparisons"] for t in ranked_tasks) // 2
    total_refusals = sum(t["n_refusals"] for t in ranked_tasks) // 2
    print(f"\nOverall refusal rate: {total_refusals}/{total_comparisons} comparisons ({total_refusals/total_comparisons:.1%})")

    print_top_bottom(ranked_tasks, show_refusal=False)


def print_top_bottom(ranked_tasks: list[dict], show_refusal: bool) -> None:
    """Print top and bottom tasks."""
    print("\nTop 5 preferred tasks:")
    for t in ranked_tasks[:5]:
        if show_refusal:
            ref_str = "[REFUSAL]" if t.get("is_refusal") else ""
        else:
            ref_str = f"[ref={t['refusal_rate']:.0%}]" if t["refusal_rate"] > 0 else ""
        print(f"  {t['rank']:3d}. mu={t['mu']:+.2f} {t['dataset']:<12} {ref_str} {t['prompt'][:60]}...")

    print("\nBottom 5 preferred tasks:")
    for t in ranked_tasks[-5:]:
        if show_refusal:
            ref_str = "[REFUSAL]" if t.get("is_refusal") else ""
        else:
            ref_str = f"[ref={t['refusal_rate']:.0%}]" if t["refusal_rate"] > 0 else ""
        print(f"  {t['rank']:3d}. mu={t['mu']:+.2f} {t['dataset']:<12} {ref_str} {t['prompt'][:60]}...")


async def main():
    parser = argparse.ArgumentParser(description="Export ranked tasks with metadata")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None, help="Filter to run starting with this name")
    parser.add_argument("--mode", type=str, choices=["auto", "post_task", "revealed"], default="auto",
                        help="Mode: auto-detect, post_task (with completions), or revealed (pairwise)")
    parser.add_argument("--model-short", type=str, default="gemma-3-27b", help="Short model name for completions")
    parser.add_argument("--completion-seed", type=int, default=0)
    parser.add_argument("--max-concurrent", type=int, default=20)
    args = parser.parse_args()

    experiment_dir = EXPERIMENTS_DIR / args.experiment_id
    if not experiment_dir.exists():
        raise ValueError(f"Experiment not found: {experiment_dir}")

    result = find_run_dir(experiment_dir, args.run_name)
    if result is None:
        raise ValueError(f"No thurstonian CSV found in {experiment_dir}")

    run_dir, al_subdir = result
    print(f"Found run at {run_dir}")
    print(f"Type: {al_subdir}")

    mode = args.mode
    if mode == "auto":
        mode = "revealed" if al_subdir == "pre_task_active_learning" else "post_task"
    print(f"Using mode: {mode}")

    if mode == "revealed":
        export_revealed(experiment_dir, run_dir, args.experiment_id, args.run_name)
    else:
        await export_post_task(
            experiment_dir, run_dir, args.experiment_id, args.run_name,
            args.model_short, args.completion_seed, args.max_concurrent
        )


if __name__ == "__main__":
    asyncio.run(main())
