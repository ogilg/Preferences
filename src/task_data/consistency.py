"""Task consistency filter based on cross-seed and cross-template variance.

Computes consistency metrics from experiment data to filter out tasks where models
don't show consistent preferences. Weights:
- 90%: Intra-(model+template) variance across seeds
- 10%: Inter-template variance within the same model

Usage:
    python -m src.task_data.consistency \
        --experiment-dirs results/experiments/multi_model_discrimination_v1

    from src.task_data.consistency import make_consistency_filter
    filter_fn = make_consistency_filter(min_score=0.5)
    tasks = load_tasks(n=200, origins=[...], filter_fn=filter_fn)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from src.measurement.storage.run_parsing import (
    extract_model_from_run_dir,
    extract_template_from_run_dir,
    normalize_score,
    parse_scale_tag,
)
from src.task_data.task import Task


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CONSISTENCY_PATH = DATA_DIR / "task_consistency.json"


@dataclass
class TaskConsistency:
    intra_std: float
    inter_std: float
    consistency_score: float
    mean_normalized: float
    n_measurements: int


@dataclass
class _RunMeasurements:
    task_scores: dict[str, float]
    model: str
    template: str
    seed: int


def _load_run_measurements(run_dir: Path) -> _RunMeasurements | None:
    config_path = run_dir / "config.yaml"
    measurements_path = run_dir / "measurements.yaml"

    if not config_path.exists() or not measurements_path.exists():
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = extract_model_from_run_dir(run_dir.name)
    template = extract_template_from_run_dir(run_dir.name)
    if not model or not template:
        return None

    # Skip qualitative templates
    if "qualitative" in template:
        return None

    # Get scale from config
    scale_tag = config.get("template_tags", {}).get("scale")
    if not scale_tag:
        return None
    scale = parse_scale_tag(scale_tag)
    if not scale:
        return None

    seed = config["rating_seed"]

    with open(measurements_path) as f:
        measurements = yaml.safe_load(f)

    if not measurements:
        return None

    task_scores: dict[str, float] = {}
    for m in measurements:
        if "score" not in m or not isinstance(m["score"], (int, float)):
            continue
        task_scores[m["task_id"]] = normalize_score(float(m["score"]), scale)

    if not task_scores:
        return None

    return _RunMeasurements(task_scores, model, template, seed)


def _load_all_runs(experiment_dirs: list[Path]) -> list[_RunMeasurements]:
    runs = []
    for exp_dir in experiment_dirs:
        stated_dir = exp_dir / "post_task_stated"
        if not stated_dir.exists():
            continue
        for run_dir in stated_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run = _load_run_measurements(run_dir)
            if run:
                runs.append(run)
    return runs


def compute_consistency(experiment_dirs: list[Path]) -> dict[str, TaskConsistency]:
    runs = _load_all_runs(experiment_dirs)
    if not runs:
        return {}

    # Collect all task ids
    all_task_ids: set[str] = set()
    for run in runs:
        all_task_ids.update(run.task_scores.keys())

    # Group by (model, template) for intra-setup variance
    by_model_template: dict[tuple[str, str], dict[int, _RunMeasurements]] = defaultdict(dict)
    for run in runs:
        by_model_template[(run.model, run.template)][run.seed] = run

    # Group by model for inter-template variance
    by_model: dict[str, list[_RunMeasurements]] = defaultdict(list)
    for run in runs:
        by_model[run.model].append(run)

    # Collect all intra and inter stds for normalization
    all_intra_stds: list[float] = []
    all_inter_stds: list[float] = []

    # First pass: compute raw stds
    task_intra_stds: dict[str, list[float]] = defaultdict(list)
    task_inter_stds: dict[str, list[float]] = defaultdict(list)
    task_scores_all: dict[str, list[float]] = defaultdict(list)
    task_n_measurements: dict[str, int] = defaultdict(int)

    for task_id in all_task_ids:
        # Intra-std: within each (model, template), std across seeds
        for (model, template), seed_runs in by_model_template.items():
            scores = []
            for seed, run in seed_runs.items():
                if task_id in run.task_scores:
                    scores.append(run.task_scores[task_id])
                    task_scores_all[task_id].append(run.task_scores[task_id])
                    task_n_measurements[task_id] += 1
            if len(scores) >= 2:
                std = float(np.std(scores))
                task_intra_stds[task_id].append(std)
                all_intra_stds.append(std)

        # Inter-std: for each model, std of mean scores across templates
        for model, model_runs in by_model.items():
            # Group by template, compute mean per template
            template_means: dict[str, list[float]] = defaultdict(list)
            for run in model_runs:
                if task_id in run.task_scores:
                    template_means[run.template].append(run.task_scores[task_id])

            # Average within each template, then compute std across templates
            if len(template_means) >= 2:
                means = [float(np.mean(scores)) for scores in template_means.values()]
                std = float(np.std(means))
                task_inter_stds[task_id].append(std)
                all_inter_stds.append(std)

    # Compute normalization factors (use 95th percentile to avoid outlier influence)
    max_intra_std = float(np.percentile(all_intra_stds, 95)) if all_intra_stds else 1.0
    max_inter_std = float(np.percentile(all_inter_stds, 95)) if all_inter_stds else 1.0

    # Prevent division by zero
    if max_intra_std < 1e-6:
        max_intra_std = 1.0
    if max_inter_std < 1e-6:
        max_inter_std = 1.0

    # Second pass: compute consistency scores
    results: dict[str, TaskConsistency] = {}

    for task_id in all_task_ids:
        intra_stds = task_intra_stds.get(task_id, [])
        inter_stds = task_inter_stds.get(task_id, [])
        all_scores = task_scores_all.get(task_id, [])

        if not intra_stds:
            continue

        mean_intra_std = float(np.mean(intra_stds))
        mean_inter_std = float(np.mean(inter_stds)) if inter_stds else 0.0

        # Normalize and clip to [0, 1]
        intra_norm = min(mean_intra_std / max_intra_std, 1.0)
        inter_norm = min(mean_inter_std / max_inter_std, 1.0)

        # Combined score (higher = more consistent)
        consistency_score = 0.9 * (1 - intra_norm) + 0.1 * (1 - inter_norm)

        results[task_id] = TaskConsistency(
            intra_std=mean_intra_std,
            inter_std=mean_inter_std,
            consistency_score=consistency_score,
            mean_normalized=float(np.mean(all_scores)) if all_scores else 0.0,
            n_measurements=task_n_measurements[task_id],
        )

    return results


def save_consistency(metrics: dict[str, TaskConsistency], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {task_id: asdict(tc) for task_id, tc in metrics.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(metrics)} task consistency metrics to {path}")


def load_consistency(path: Path) -> dict[str, TaskConsistency]:
    with open(path) as f:
        data = json.load(f)
    return {task_id: TaskConsistency(**tc) for task_id, tc in data.items()}


def make_consistency_filter(
    min_score: float = 0.5,
    consistency_path: Path | None = None,
) -> Callable[[Task], bool]:
    path = consistency_path or DEFAULT_CONSISTENCY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Consistency data not found at {path}. "
            f"Run: python -m src.task_data.consistency --experiment-dirs <path>"
        )
    metrics = load_consistency(path)

    def filter_fn(task: Task) -> bool:
        tc = metrics.get(task.id)
        if tc is None:
            return True  # Allow tasks not in the metrics (not measured yet)
        return tc.consistency_score >= min_score

    return filter_fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute task consistency metrics from experiment data"
    )
    parser.add_argument(
        "--experiment-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Experiment directories to analyze",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CONSISTENCY_PATH,
        help=f"Output path (default: {DEFAULT_CONSISTENCY_PATH})",
    )
    args = parser.parse_args()

    experiment_dirs = [Path(d) for d in args.experiment_dirs]
    for d in experiment_dirs:
        if not d.exists():
            print(f"Warning: {d} does not exist")

    print(f"Computing consistency from {len(experiment_dirs)} experiment directories...")
    metrics = compute_consistency(experiment_dirs)

    if not metrics:
        print("No metrics computed")
        return

    # Print summary stats
    scores = [tc.consistency_score for tc in metrics.values()]
    intra_stds = [tc.intra_std for tc in metrics.values()]
    inter_stds = [tc.inter_std for tc in metrics.values()]

    print(f"\nComputed metrics for {len(metrics)} tasks:")
    print(f"  Consistency score: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
    print(f"  Intra-std: mean={np.mean(intra_stds):.4f}, std={np.std(intra_stds):.4f}")
    print(f"  Inter-std: mean={np.mean(inter_stds):.4f}, std={np.std(inter_stds):.4f}")

    # Show distribution
    thresholds = [0.3, 0.5, 0.7, 0.9]
    print("\n  Tasks by consistency threshold:")
    for thresh in thresholds:
        count = sum(1 for s in scores if s >= thresh)
        print(f"    >= {thresh}: {count} ({100*count/len(scores):.1f}%)")

    # Show lowest consistency tasks
    sorted_metrics = sorted(metrics.items(), key=lambda kv: kv[1].consistency_score)
    print("\n  Lowest consistency tasks:")
    for task_id, tc in sorted_metrics[:10]:
        print(f"    {task_id}: score={tc.consistency_score:.3f}, intra={tc.intra_std:.4f}, inter={tc.inter_std:.4f}")

    save_consistency(metrics, args.output)


if __name__ == "__main__":
    main()
