"""Shared utilities for loading preference measurement runs."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.preferences.storage import load_yaml
from src.preferences.templates.template import load_templates_from_yaml


@dataclass
class RunConfig:
    template_name: str
    template_tags: dict
    model_short: str
    run_dir: Path


def _parse_stated_dir_name(dir_name: str) -> tuple[str, str] | None:
    """Parse 'stated_{template_name}_{model_short}' -> (template_name, model_short)."""
    match = re.match(r"stated_([^_]+_\d+)_(.+)$", dir_name)
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
            # Stated format: parse directory name
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
    # Try binary format first (thurstonian CSV)
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

    # Try rating format (scores.yaml)
    scores_path = run_dir / "scores.yaml"
    if scores_path.exists():
        scores = load_yaml(scores_path)
        task_ids = [s["task_id"] for s in scores]
        utilities = np.array([s["score"] for s in scores])
        return utilities, task_ids

    # Try stated format (measurements.yaml with raw samples)
    measurements_path = run_dir / "measurements.yaml"
    if measurements_path.exists():
        measurements = load_yaml(measurements_path)
        return _aggregate_scores(measurements)

    raise FileNotFoundError(f"No utility data found in {run_dir}")


def load_completed_runs(
    results_dir: Path,
    template_yaml: Path | None = None,
    min_tasks: int = 0,
    require_csv: bool = False,
) -> list[tuple[RunConfig, np.ndarray, list[str]]]:
    """Load all completed runs with their utilities.

    Args:
        results_dir: Directory containing measurement runs
        template_yaml: Template YAML file (for stated runs without config.yaml)
        min_tasks: Minimum number of tasks required
        require_csv: If True, only include runs with thurstonian CSV files
    """
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
