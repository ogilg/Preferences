"""Storage layer for measurement runs.

Each run stores measurements + Thurstonian results for a single
(template, model, config) combination.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.models import HyperbolicModel
from src.preferences.ranking import ThurstonianResult, save_thurstonian, load_thurstonian
from src.preferences.templates import PromptTemplate, load_templates_from_yaml
from src.task_data import Task
from src.types import BinaryPreferenceMeasurement


RESULTS_DIR = Path("results")


def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found)")


def _extract_template_id(template_name: str) -> str:
    """Extract template ID from name (e.g., 'binary_choice_001' -> '001')."""
    return template_name.rsplit("_", 1)[-1]


def _model_short_name(model_name: str) -> str:
    """Extract short model name."""
    name = model_name.split("/")[-1]
    name = name.replace("-Instruct", "").replace("Meta-", "")
    return name.lower()


def save_measurements(
    measurements: list[BinaryPreferenceMeasurement],
    path: Path | str,
) -> None:
    """Save binary preference measurements to YAML.

    Args:
        measurements: List of binary preference measurements.
        path: Path to save the YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {"task_a": m.task_a.id, "task_b": m.task_b.id, "choice": m.choice}
        for m in measurements
    ]

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


@dataclass
class MeasurementRunConfig:
    """Configuration and metadata for a measurement run."""

    template_id: str
    template_name: str
    template_file: str
    template_tags: dict[str, str]
    model: str
    model_short: str
    temperature: float
    task_origin: str
    n_tasks: int
    task_ids: list[str]
    task_prompts: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.task_prompts is None:
            self.task_prompts = {}

    def load_template(self) -> PromptTemplate:
        """Load the template from template_file."""
        template_path = Path(self.template_file)
        if not template_path.is_absolute():
            template_path = _find_project_root() / self.template_file

        templates = load_templates_from_yaml(template_path)
        for t in templates:
            if t.name == self.template_name:
                return t
        raise ValueError(f"Template '{self.template_name}' not found in {self.template_file}")

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "template_name": self.template_name,
            "template_file": self.template_file,
            "template_tags": self.template_tags,
            "model": self.model,
            "model_short": self.model_short,
            "temperature": self.temperature,
            "task_origin": self.task_origin,
            "n_tasks": self.n_tasks,
            "task_ids": self.task_ids,
            "task_prompts": self.task_prompts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MeasurementRunConfig:
        return cls(
            template_id=data["template_id"],
            template_name=data["template_name"],
            template_file=data["template_file"],
            template_tags=data["template_tags"],
            model=data["model"],
            model_short=data["model_short"],
            temperature=data["temperature"],
            task_origin=data["task_origin"],
            n_tasks=data["n_tasks"],
            task_ids=data["task_ids"],
            task_prompts=data.get("task_prompts", {}),
        )


@dataclass
class MeasurementRun:
    """A loaded measurement run."""

    config: MeasurementRunConfig
    measurements: list[dict]
    thurstonian: ThurstonianResult | None = None
    path: Path | None = None


@dataclass
class ThurstonianData:
    """Lightweight Thurstonian result for visualization (no Task objects)."""

    task_ids: list[str]
    mu: np.ndarray
    sigma: np.ndarray
    converged: bool
    neg_log_likelihood: float

    def ranking_order(self) -> list[int]:
        """Indices sorted by utility (highest first)."""
        return list(np.argsort(-self.mu))


def run_exists(
    template: PromptTemplate,
    model: HyperbolicModel,
    results_dir: Path | str = RESULTS_DIR,
) -> bool:
    """Check if a measurement run already exists for this template/model combo."""
    results_dir = Path(results_dir)
    template_id = _extract_template_id(template.name)
    model_short = _model_short_name(model.model_name)
    run_dir = results_dir / f"{template_id}_{model_short}"
    return (run_dir / "measurements.yaml").exists()


def save_run(
    template: PromptTemplate,
    template_file: str,
    model: HyperbolicModel,
    temperature: float,
    tasks: list[Task],
    measurements: list[BinaryPreferenceMeasurement],
    thurstonian: ThurstonianResult,
    results_dir: Path | str = RESULTS_DIR,
) -> Path:
    """Save a measurement run to disk.

    Creates:
        results/{template_id}_{model_short}/
            config.yaml
            measurements.yaml
            thurstonian.yaml

    Returns:
        Path to the created run directory.
    """
    results_dir = Path(results_dir)
    template_id = _extract_template_id(template.name)
    model_short = _model_short_name(model.model_name)

    run_dir = results_dir / f"{template_id}_{model_short}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = MeasurementRunConfig(
        template_id=template_id,
        template_name=template.name,
        template_file=template_file,
        template_tags=template.tags_dict,
        model=model.model_name,
        model_short=model_short,
        temperature=temperature,
        task_origin=tasks[0].origin.name.lower(),
        n_tasks=len(tasks),
        task_ids=[t.id for t in tasks],
        task_prompts={t.id: t.prompt for t in tasks},
    )

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

    save_measurements(measurements, run_dir / "measurements.yaml")
    save_thurstonian(thurstonian, run_dir / "thurstonian.yaml")

    update_index(results_dir)
    return run_dir


def load_run(
    run_dir: Path | str,
    tasks: list[Task] | None = None,
) -> MeasurementRun:
    """Load a measurement run from disk.

    Args:
        run_dir: Path to the run directory.
        tasks: Optional tasks for reconstructing ThurstonianResult.

    Returns:
        MeasurementRun with config, measurements, and optionally thurstonian.
    """
    run_dir = Path(run_dir)

    with open(run_dir / "config.yaml") as f:
        config = MeasurementRunConfig.from_dict(yaml.safe_load(f))

    with open(run_dir / "measurements.yaml") as f:
        measurements = yaml.safe_load(f)

    thurstonian = None
    if tasks is not None:
        thurstonian = load_thurstonian(run_dir / "thurstonian.yaml", tasks)

    return MeasurementRun(
        config=config,
        measurements=measurements,
        thurstonian=thurstonian,
        path=run_dir,
    )


def list_runs(
    results_dir: Path | str = RESULTS_DIR,
    **filters: str,
) -> list[MeasurementRunConfig]:
    """List measurement runs, optionally filtered.

    Args:
        results_dir: Base results directory.
        **filters: Filter by config fields or template tags.
            E.g., model_short="llama-3.1-8b", phrasing="1"

    Returns:
        List of matching MeasurementRunConfig.
    """
    results_dir = Path(results_dir)
    index_path = results_dir / "index.yaml"

    if not index_path.exists():
        return []

    with open(index_path) as f:
        index_data = yaml.safe_load(f)

    runs = []
    for entry in index_data.get("runs", []):
        if all(entry.get(k) == v for k, v in filters.items()):
            with open(results_dir / entry["dir"] / "config.yaml") as f:
                runs.append(MeasurementRunConfig.from_dict(yaml.safe_load(f)))

    return runs


def update_index(results_dir: Path | str = RESULTS_DIR) -> None:
    """Regenerate index.yaml from existing run directories."""
    results_dir = Path(results_dir)

    runs = []
    for run_dir in sorted(results_dir.iterdir()):
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Skip directories that don't have the expected run config format
        if "template_id" not in config or "model_short" not in config:
            continue

        entry = {
            "dir": run_dir.name,
            "template_id": config["template_id"],
            "model_short": config["model_short"],
        }
        entry.update(config.get("template_tags", {}))
        runs.append(entry)

    with open(results_dir / "index.yaml", "w") as f:
        yaml.dump({"runs": runs}, f, default_flow_style=False, sort_keys=False)


def load_thurstonian_data(run_dir: Path | str) -> ThurstonianData:
    """Load thurstonian.yaml as ThurstonianData (no Task objects required).

    Args:
        run_dir: Path to the run directory containing thurstonian.yaml.

    Returns:
        ThurstonianData with task_ids, mu, sigma arrays.
    """
    run_dir = Path(run_dir)
    thurstonian_path = run_dir / "thurstonian.yaml"

    with open(thurstonian_path) as f:
        data = yaml.safe_load(f)

    return ThurstonianData(
        task_ids=data["task_ids"],
        mu=np.array(data["mu"]),
        sigma=np.array(data["sigma"]),
        converged=data["converged"],
        neg_log_likelihood=data["neg_log_likelihood"],
    )
