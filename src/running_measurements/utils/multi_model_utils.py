"""Utilities for running experiments across multiple models in parallel."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskID


@dataclass(frozen=True)
class ExperimentTask:
    experiment_type: str
    model: str
    config_path: Path


PROGRESS_PATTERN = re.compile(r"\[PROGRESS (\d+)/(\d+)\]")


@dataclass
class RunningExperiment:
    task: ExperimentTask
    process: subprocess.Popen
    start_time: float
    log_file: Path
    log_position: int = 0
    current: int = 0
    total: int = 0
    progress_task_id: TaskID | None = None


def model_to_filename(model: str) -> str:
    """Convert model name to valid filename."""
    return model.replace("/", "_").replace(":", "_")


def generate_experiment_configs(
    models: list[str],
    base_config_active: Path,
    base_config_stated: Path,
    max_concurrent: int,
    output_dir: Path,
) -> list[ExperimentTask]:
    """Generate per-model config files and return task list."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(base_config_active) as f:
        base_active = yaml.safe_load(f)
    with open(base_config_stated) as f:
        base_stated = yaml.safe_load(f)

    tasks = []
    for model in models:
        al_config = base_active.copy()
        al_config["model"] = model
        al_config["max_concurrent"] = max_concurrent

        al_path = output_dir / f"active_learning_{model_to_filename(model)}.yaml"
        with open(al_path, "w") as f:
            yaml.safe_dump(al_config, f, sort_keys=False)
        tasks.append(ExperimentTask("active_learning", model, al_path))

        stated_config = base_stated.copy()
        stated_config["model"] = model
        stated_config["max_concurrent"] = max_concurrent

        stated_path = output_dir / f"stated_measurement_{model_to_filename(model)}.yaml"
        with open(stated_path, "w") as f:
            yaml.safe_dump(stated_config, f, sort_keys=False)
        tasks.append(ExperimentTask("stated_measurement", model, stated_path))

    return tasks


def launch_experiment(task: ExperimentTask, log_dir: Path) -> RunningExperiment:
    """Launch a single experiment subprocess."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{task.experiment_type}_{model_to_filename(task.model)}.log"

    cmd = [
        "python",
        "-m",
        f"src.experiments.run_{task.experiment_type}",
        str(task.config_path),
    ]

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return RunningExperiment(
        task=task,
        process=proc,
        start_time=time.time(),
        log_file=log_file,
    )


def parse_progress_from_log(exp: RunningExperiment) -> tuple[int, int]:
    """Read new content from log file and extract latest progress."""
    try:
        with open(exp.log_file) as f:
            f.seek(exp.log_position)
            new_content = f.read()
            exp.log_position = f.tell()
    except FileNotFoundError:
        return exp.current, exp.total

    for match in PROGRESS_PATTERN.finditer(new_content):
        exp.current = int(match.group(1))
        exp.total = int(match.group(2))

    return exp.current, exp.total


def format_experiment_label(task: ExperimentTask) -> str:
    """Format experiment label for display."""
    exp_type = "AL" if task.experiment_type == "active_learning" else "ST"
    model_short = task.model.split("/")[-1][:25]
    return f"[{exp_type}] {model_short}"


def run_experiments_parallel(
    tasks: list[ExperimentTask],
    log_dir: Path,
) -> dict[ExperimentTask, bool]:
    """Run all experiments in parallel with individual progress bars."""
    console = Console()
    pending = list(tasks)
    running: dict[subprocess.Popen, RunningExperiment] = {}
    results: dict[ExperimentTask, bool] = {}
    completed_messages: list[str] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        console=console,
    )

    with Live(progress, console=console, refresh_per_second=2) as live:
        try:
            while pending or running:
                while pending:
                    task = pending.pop(0)
                    exp = launch_experiment(task, log_dir)
                    label = format_experiment_label(task)
                    exp.progress_task_id = progress.add_task(label, total=1)
                    running[exp.process] = exp

                for exp in running.values():
                    current, total = parse_progress_from_log(exp)
                    if total > 0 and exp.progress_task_id is not None:
                        progress.update(exp.progress_task_id, completed=current, total=total)

                for proc in list(running.keys()):
                    retcode = proc.poll()
                    if retcode is not None:
                        exp = running.pop(proc)
                        success = retcode == 0
                        results[exp.task] = success

                        if exp.progress_task_id is not None:
                            if success:
                                progress.update(exp.progress_task_id, completed=exp.total, total=exp.total)
                            progress.remove_task(exp.progress_task_id)

                        duration = time.time() - exp.start_time
                        status = "[green]✓[/green]" if success else f"[red]✗ (code {retcode})[/red]"
                        label = format_experiment_label(exp.task)
                        completed_messages.append(f"{status} {label} ({duration:.0f}s)")

                time.sleep(0.5)

        except KeyboardInterrupt:
            console.print("\n[yellow]Shutdown requested, terminating experiments...[/yellow]")
            for proc, exp in running.items():
                proc.terminate()
                console.print(f"[yellow]Terminated: {exp.task.experiment_type} - {exp.task.model}[/yellow]")
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            raise

    console.print()
    for msg in completed_messages:
        console.print(msg)

    return results


def print_summary(results: dict[ExperimentTask, bool], log_dir: Path):
    """Print summary of experiment results."""
    console = Console()
    successes = [t for t, success in results.items() if success]
    failures = [t for t, success in results.items() if not success]

    console.print("\n" + "=" * 60)
    console.print("[bold]Multi-Model Experiment Summary[/bold]")
    console.print("=" * 60)
    console.print(f"Total experiments: {len(results)}")
    console.print(f"[green]Successful: {len(successes)}[/green]")
    if failures:
        console.print(f"[red]Failed: {len(failures)}[/red]")

    if failures:
        console.print("\n[red]FAILED:[/red]")
        for task in failures:
            log_file = log_dir / f"{task.experiment_type}_{model_to_filename(task.model)}.log"
            console.print(f"  - {task.experiment_type} - {task.model}")
            console.print(f"    Log: {log_file}")
