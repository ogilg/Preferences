"""Utilities for running experiments across multiple models in parallel."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from tqdm import tqdm


@dataclass
class ExperimentTask:
    experiment_type: str
    model: str
    config_path: Path


@dataclass
class RunningExperiment:
    task: ExperimentTask
    process: subprocess.Popen
    start_time: float
    log_file: Path


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


def run_experiments_parallel(
    tasks: list[ExperimentTask],
    log_dir: Path,
) -> dict[ExperimentTask, bool]:
    """Run all experiments in parallel."""
    pending = list(tasks)
    running: dict[subprocess.Popen, RunningExperiment] = {}
    results: dict[ExperimentTask, bool] = {}

    with tqdm(total=len(tasks), desc="Experiments") as pbar:
        try:
            while pending or running:
                while pending:
                    task = pending.pop(0)
                    exp = launch_experiment(task, log_dir)
                    running[exp.process] = exp
                    print(f"\n[STARTED] {task.experiment_type} - {task.model}")

                for proc in list(running.keys()):
                    retcode = proc.poll()
                    if retcode is not None:
                        exp = running.pop(proc)
                        success = retcode == 0
                        results[exp.task] = success

                        duration = time.time() - exp.start_time
                        status = "SUCCESS" if success else f"FAILED (code {retcode})"
                        print(f"\n[{status}] {exp.task.experiment_type} - {exp.task.model} ({duration:.1f}s)")
                        print(f"  Log: {exp.log_file}")
                        pbar.update(1)

                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\nShutdown requested, terminating experiments...")
            for proc, exp in running.items():
                proc.terminate()
                print(f"Terminated: {exp.task.experiment_type} - {exp.task.model}")
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            raise

    return results


def print_summary(results: dict[ExperimentTask, bool], log_dir: Path):
    """Print summary of experiment results."""
    successes = [t for t, success in results.items() if success]
    failures = [t for t, success in results.items() if not success]

    print("\n" + "=" * 60)
    print("Multi-Model Experiment Summary")
    print("=" * 60)
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successes)}")
    print(f"Failed: {len(failures)}")

    if failures:
        print("\nFAILED:")
        for task in failures:
            log_file = log_dir / f"{task.experiment_type}_{model_to_filename(task.model)}.log"
            print(f"  - {task.experiment_type} - {task.model}")
            print(f"    Log: {log_file}")

    if successes:
        print("\nSUCCESSFUL:")
        for task in successes:
            print(f"  - {task.experiment_type} - {task.model}")
