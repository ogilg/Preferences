from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from src.models import HyperbolicModel
from src.preferences.templates.template import PromptTemplate


RESULTS_DIR = Path("results/binary")


def find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found)")


def extract_template_id(template_name: str) -> str:
    """Extract template ID from name (e.g., 'binary_choice_001' -> '001')."""
    return template_name.rsplit("_", 1)[-1]


def model_short_name(model_name: str) -> str:
    name = model_name.split("/")[-1]
    name = name.replace("-Instruct", "").replace("Meta-", "")
    return name.lower()


def save_yaml(data: dict | list, path: Path) -> None:
    """Save data to YAML with parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_yaml(path: Path) -> dict | list:
    with open(path) as f:
        return yaml.safe_load(f)


def run_exists(
    template: PromptTemplate,
    model: HyperbolicModel,
    n_tasks: int,
    results_dir: Path,
) -> bool:
    run_dir = get_run_dir(template, model, n_tasks, results_dir)
    return (run_dir / "config.yaml").exists()


class BaseRunConfig(BaseModel):
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
    task_prompts: dict[str, str] = {}

    def load_template(self) -> PromptTemplate:
        from src.preferences.templates.template import load_templates_from_yaml

        template_path = Path(self.template_file)
        if not template_path.is_absolute():
            template_path = find_project_root() / self.template_file

        templates = load_templates_from_yaml(template_path)
        try:
            return next(t for t in templates if t.name == self.template_name)
        except StopIteration:
            raise ValueError(f"Template '{self.template_name}' not found in {self.template_file}") from None


def get_run_dir(
    template: PromptTemplate,
    model: HyperbolicModel,
    n_tasks: int,
    results_dir: Path,
) -> Path:
    template_id = extract_template_id(template.name)
    short_name = model_short_name(model.model_name)
    return results_dir / f"n{n_tasks}" / f"{template_id}_{short_name}"


def make_base_config_dict(
    template: PromptTemplate,
    template_file: str,
    model: HyperbolicModel,
    temperature: float,
    tasks: list,
) -> dict:
    """Build the common config fields. Caller adds type-specific fields."""
    template_id = extract_template_id(template.name)
    short_name = model_short_name(model.model_name)
    return {
        "template_id": template_id,
        "template_name": template.name,
        "template_file": template_file,
        "template_tags": template.tags_dict,
        "model": model.model_name,
        "model_short": short_name,
        "temperature": temperature,
        "task_origin": tasks[0].origin.name.lower(),
        "n_tasks": len(tasks),
        "task_ids": [t.id for t in tasks],
        "task_prompts": {t.id: t.prompt for t in tasks},
    }


def save_run_files(
    run_dir: Path,
    config: BaseRunConfig,
    data: list[dict],
    data_filename: str,
) -> None:
    """Save config and data files to run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config.model_dump(), run_dir / "config.yaml")
    save_yaml(data, run_dir / data_filename)


def load_run_files(
    run_dir: Path,
    config_class: type[BaseRunConfig],
    data_filename: str,
) -> tuple[BaseRunConfig, list[dict]]:
    """Load config and data files from run directory."""
    config = config_class.model_validate(load_yaml(run_dir / "config.yaml"))
    data = load_yaml(run_dir / data_filename)
    return config, data
