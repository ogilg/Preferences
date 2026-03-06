"""Run all pending activation extractions: OOD experiments + MRA personas.

Reads AL measurement configs to get system prompts and task files,
then runs the main extraction pipeline for each condition with --resume.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.probes.extraction.config import ExtractionConfig
from src.probes.extraction.extract import run_extraction

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_ood(act_base: str, layers: list[int], selectors: list[str]) -> None:
    """Extract OOD activations for exp1a-1d.

    act_base: e.g. "activations/ood" or "activations/ood_eot"
    """
    for sub in ["a", "b", "c", "d"]:
        al_dir = PROJECT_ROOT / f"configs/measurement/active_learning/ood_exp1{sub}"
        act_root = PROJECT_ROOT / act_base / ("exp1_category" if sub == "a" else "exp1_prompts")

        for al_config_path in sorted(al_dir.glob("*.yaml")):
            condition = al_config_path.stem
            with open(al_config_path) as f:
                al_config = yaml.safe_load(f)

            config = ExtractionConfig(
                model="gemma-3-27b",
                n_tasks=al_config["n_tasks"],
                task_origins=al_config.get("task_origins", ["wildchat", "alpaca", "math", "bailbench", "stress_test"]),
                layers_to_extract=layers,
                selectors=selectors,
                batch_size=32,
                output_dir=str(act_root / condition),
                resume=True,
                system_prompt=al_config.get("measurement_system_prompt"),
                custom_tasks_file=Path(p) if (p := al_config.get("custom_tasks_file")) else None,
                task_ids_file=Path(p) if (p := al_config.get("include_task_ids_file")) else None,
            )

            print(f"exp1{sub}/{condition} (selectors={selectors})")
            run_extraction(config)


def run_ood_extractions() -> None:
    run_ood("activations/ood", [31, 43, 55], ["prompt_last"])


def run_ood_eot_extractions() -> None:
    run_ood("activations/ood_eot", [31], ["eot"])


def run_mra_extractions() -> None:
    for name in ["mra_persona2_villain", "mra_persona3_midwest", "mra_persona4_aesthete"]:
        config_path = PROJECT_ROOT / "configs" / "extraction" / f"{name}.yaml"
        print(f"MRA: {name}")
        config = ExtractionConfig.from_yaml(config_path, resume=True)
        run_extraction(config)


if __name__ == "__main__":
    run_ood_extractions()
    run_mra_extractions()
