"""Consolidated loading utilities for preference measurement runs."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.measurement_storage.base import load_yaml
from src.measurement_storage.stated import PRE_TASK_STATED_DIR
from src.measurement_storage.cache import PRE_TASK_REVEALED_DIR
from src.measurement_storage.post_task import POST_STATED_DIR, POST_REVEALED_DIR
from src.prompt_templates.template import load_templates_from_yaml


@dataclass
class RunConfig:
    template_name: str
    template_tags: dict
    model_short: str
    run_dir: Path


def _parse_stated_dir_name(dir_name: str) -> tuple[str, str] | None:
    """Parse stated dir name -> (template_name, model_short).

    Handles formats:
    - {template}_{model}_{response_format}_cseed{N}_rseed{N} (new format)
    - stated_{template}_{model} (legacy pre-task format)
    """
    # New format: template_name_NNN_model_format_cseed_rseed
    match = re.match(r"([^_]+_[^_]+_\d+)_([^_]+(?:-[^_]+)*)_(?:regex|xml|tool_use)_cseed\d+_rseed\d+$", dir_name)
    if match:
        return match.group(1), match.group(2)
    # Legacy format: stated_template_model
    match = re.match(r"stated_([^_]+_\d+)_(.+?)(?:_(?:regex|xml|tool_use))?(?:_cseed\d+)?(?:_rseed\d+)?$", dir_name)
    if match:
        return match.group(1), match.group(2)
    return None


def list_runs(results_dir: Path, template_yaml: Path | None = None) -> list[RunConfig]:
    """List all measurement runs in a directory."""
    runs = []
    if not results_dir.exists():
        return runs

    template_tags_map: dict[str, dict] | None = None

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.yaml"
        if config_path.exists():
            config = load_yaml(config_path)
            runs.append(RunConfig(
                template_name=config["template_name"],
                template_tags=config["template_tags"],
                model_short=config["model_short"],
                run_dir=run_dir,
            ))
        elif (run_dir / "measurements.yaml").exists():
            parsed = _parse_stated_dir_name(run_dir.name)
            if parsed is None:
                continue
            template_name, model_short = parsed

            if template_tags_map is None:
                if template_yaml is None:
                    continue
                templates = load_templates_from_yaml(template_yaml)
                template_tags_map = {t.name: t.tags_dict for t in templates}

            if template_name not in template_tags_map:
                continue

            runs.append(RunConfig(
                template_name=template_name,
                template_tags=template_tags_map[template_name],
                model_short=model_short,
                run_dir=run_dir,
            ))
    return runs


def find_thurstonian_csv(run_dir: Path) -> Path | None:
    """Find pre-computed thurstonian CSV file (active learning only)."""
    matches = list(run_dir.glob("thurstonian_active_learning_*.csv"))
    if matches:
        return matches[0]

    csv_path = run_dir / "thurstonian_active_learning.csv"
    if csv_path.exists():
        return csv_path

    return None


def _aggregate_scores(measurements: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Aggregate multiple samples per task into mean scores."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for m in measurements:
        by_task[m["task_id"]].append(m["score"])
    task_ids = sorted(by_task.keys())
    scores = np.array([np.mean(by_task[tid]) for tid in task_ids])
    return scores, task_ids


def load_run_utilities(run_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load utilities from thurstonian CSV, scores.yaml, or measurements.yaml."""
    csv_path = find_thurstonian_csv(run_dir)
    if csv_path is not None:
        task_ids = []
        mus = []
        with open(csv_path) as f:
            next(f)  # Skip header
            for line in f:
                task_id, mu, _ = line.strip().split(",")
                task_ids.append(task_id)
                mus.append(float(mu))
        return np.array(mus), task_ids

    scores_path = run_dir / "scores.yaml"
    if scores_path.exists():
        scores = load_yaml(scores_path)
        task_ids = [s["task_id"] for s in scores]
        utilities = np.array([s["score"] for s in scores])
        return utilities, task_ids

    measurements_path = run_dir / "measurements.yaml"
    if measurements_path.exists():
        measurements = load_yaml(measurements_path)
        if measurements and "task_id" in measurements[0]:
            return _aggregate_scores(measurements)
        raise FileNotFoundError(f"Pairwise comparison data without thurstonian CSV in {run_dir}")

    raise FileNotFoundError(f"No utility data found in {run_dir}")


def load_completed_runs(
    results_dir: Path,
    template_yaml: Path | None = None,
    min_tasks: int = 0,
    require_csv: bool = False,
) -> list[tuple[RunConfig, np.ndarray, list[str]]]:
    """Load all completed runs with their utilities."""
    runs = list_runs(results_dir, template_yaml)
    completed = []

    for config in runs:
        if require_csv and find_thurstonian_csv(config.run_dir) is None:
            continue

        try:
            mu, task_ids = load_run_utilities(config.run_dir)
        except FileNotFoundError:
            continue

        if len(task_ids) < min_tasks:
            continue

        completed.append((config, mu, task_ids))

    return completed


def load_pairwise_datasets(
    results_dir: Path,
    skip_prefixes: tuple[str, ...] = ("rating_",),
) -> list[tuple[str, np.ndarray, list[str]]]:
    """Load pairwise comparison datasets as win matrices.

    Returns list of (name, wins, task_ids) tuples.
    """
    from src.task_data import Task, OriginDataset

    datasets = []
    if not results_dir.exists():
        return datasets

    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir():
            continue
        if any(result_dir.name.startswith(p) for p in skip_prefixes):
            continue
        measurements_path = result_dir / "measurements.yaml"
        if not measurements_path.exists():
            continue

        with open(measurements_path) as f:
            measurements = yaml.load(f, Loader=yaml.CSafeLoader)
        if not measurements:
            continue

        task_ids_set: set[str] = set()
        for m in measurements:
            task_ids_set.add(m["task_a"])
            task_ids_set.add(m["task_b"])

        task_ids = sorted(task_ids_set)
        id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
        n = len(task_ids)
        wins = np.zeros((n, n), dtype=np.int32)

        for m in measurements:
            i, j = id_to_idx[m["task_a"]], id_to_idx[m["task_b"]]
            if m["choice"] == "a":
                wins[i, j] += 1
            else:
                wins[j, i] += 1

        datasets.append((result_dir.name, wins, task_ids))

    return datasets


def load_all_stated_runs(
    template_yaml: Path | None = None,
    min_tasks: int = 0,
    include_pre: bool = True,
    include_post: bool = True,
) -> list[tuple[RunConfig, np.ndarray, list[str]]]:
    """Load stated preference runs from both pre-task and post-task directories."""
    completed = []
    if include_pre:
        completed.extend(load_completed_runs(PRE_TASK_STATED_DIR, template_yaml, min_tasks))
    if include_post:
        completed.extend(load_completed_runs(POST_STATED_DIR, template_yaml, min_tasks))
    return completed


def load_all_pairwise_datasets(
    skip_prefixes: tuple[str, ...] = ("rating_",),
    include_pre: bool = True,
    include_post: bool = True,
) -> list[tuple[str, np.ndarray, list[str]]]:
    """Load pairwise datasets from both pre-task and post-task revealed directories."""
    datasets = []
    if include_pre:
        datasets.extend(load_pairwise_datasets(PRE_TASK_REVEALED_DIR, skip_prefixes))
    if include_post:
        datasets.extend(load_pairwise_datasets(POST_REVEALED_DIR, skip_prefixes))
    return datasets


def discover_post_stated_caches(
    model_filter: str | None = None,
    template_filter: str | None = None,
) -> list[tuple[str, Path]]:
    """Discover all PostStatedCache directories.

    Returns list of (name, cache_dir) tuples.
    """
    if not POST_STATED_DIR.exists():
        return []

    caches = []
    for cache_dir in POST_STATED_DIR.iterdir():
        if not cache_dir.is_dir():
            continue

        measurements_path = cache_dir / "measurements.yaml"
        if not measurements_path.exists():
            continue

        name = cache_dir.name
        if model_filter and model_filter not in name:
            continue
        if template_filter and template_filter not in name:
            continue

        caches.append((name, cache_dir))

    return sorted(caches)


def load_scores_from_cache(cache_dir: Path) -> dict[str, float]:
    """Load task_id -> score mapping from a PostStatedCache directory."""
    measurements_path = cache_dir / "measurements.yaml"
    data = load_yaml(measurements_path)
    return {item["task_id"]: item["score"] for item in data}
