from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from src.models import HyperbolicModel
    from src.preferences.templates.template import PromptTemplate


RESULTS_DIR = Path("results")


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
    template_id = extract_template_id(template.name)
    short_name = model_short_name(model.model_name)
    config_path = results_dir / f"{template_id}_{short_name}" / "config.yaml"
    if not config_path.exists():
        return False
    return load_yaml(config_path)["n_tasks"] == n_tasks


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
        for t in templates:
            if t.name == self.template_name:
                return t
        raise ValueError(f"Template '{self.template_name}' not found in {self.template_file}")
