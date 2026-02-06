"""Shared utilities for active learning analysis plots."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
PLOTS_DIR = Path(__file__).parent / "plots"

DATASET_COLORS = {
    "math": "#4C72B0",
    "alpaca": "#55A868",
    "wildchat": "#C44E52",
    "bailbench": "#8172B2",
    "stress_test": "#CCB974",
    "other": "#999999",
}


def load_ranked_tasks(experiment_id: str, run_name: str | None = None) -> list[dict]:
    """Load the most recent ranked tasks JSON for an experiment."""
    exp_output_dir = OUTPUT_DIR / experiment_id
    suffix = f"_{run_name}" if run_name else ""
    pattern = f"ranked_tasks{suffix}_*.json"

    candidates = list(exp_output_dir.glob(pattern)) if exp_output_dir.exists() else []
    if not candidates:
        raise ValueError(f"No ranked_tasks file found in {exp_output_dir} for pattern {pattern}. Run export_ranked_tasks.py first.")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    path = candidates[0]
    print(f"Loading {path}")

    with open(path) as f:
        return json.load(f)


def plot_output_path(experiment_id: str, plot_name: str, run_name: str | None = None) -> Path:
    """Generate a dated output path for a plot."""
    output_dir = PLOTS_DIR / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d%y")
    suffix = f"_{run_name}" if run_name else ""
    return output_dir / f"plot_{date_str}_{plot_name}{suffix}.png"
