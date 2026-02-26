"""Run all pending activation extractions: OOD experiments + MRA personas.

Reads AL measurement configs to get system prompts and task files,
then runs the main extraction pipeline for each condition with --resume.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.probes.extraction.config import ExtractionConfig
from src.probes.extraction.extract import run_extraction

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL = "gemma-3-27b"
OOD_LAYERS = [31, 43, 55]
OOD_SELECTORS = ["prompt_last"]
OOD_BATCH_SIZE = 32

# Mapping: (AL config dir, activation output root)
OOD_EXPERIMENTS = [
    ("configs/measurement/active_learning/ood_exp1a", "activations/ood/exp1_category"),
    ("configs/measurement/active_learning/ood_exp1b", "activations/ood/exp1_prompts"),
    ("configs/measurement/active_learning/ood_exp1c", "activations/ood/exp1_prompts"),
    ("configs/measurement/active_learning/ood_exp1d", "activations/ood/exp1_prompts"),
]

MRA_CONFIGS = [
    "configs/extraction/mra_persona2_villain.yaml",
    "configs/extraction/mra_persona3_midwest.yaml",
    "configs/extraction/mra_persona4_aesthete.yaml",
]


def run_ood_extractions() -> None:
    for al_dir_rel, act_root_rel in OOD_EXPERIMENTS:
        al_dir = PROJECT_ROOT / al_dir_rel
        act_root = PROJECT_ROOT / act_root_rel
        exp_name = al_dir.name  # e.g. ood_exp1a

        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name} -> {act_root_rel}")
        print(f"{'='*60}")

        for al_config_path in sorted(al_dir.glob("*.yaml")):
            condition = al_config_path.stem  # e.g. coding_neg_persona
            with open(al_config_path) as f:
                al_config = yaml.safe_load(f)

            system_prompt = al_config.get("measurement_system_prompt")
            custom_tasks_file = al_config.get("custom_tasks_file")
            include_task_ids_file = al_config.get("include_task_ids_file")

            output_dir = str(act_root / condition)

            # Build extraction config — OOD uses resolved int layers directly
            # task_origins is required by the schema but ignored when custom_tasks_file
            # or task_ids_file is set
            config = ExtractionConfig(
                model=MODEL,
                n_tasks=al_config["n_tasks"],
                task_origins=al_config.get("task_origins", ["wildchat", "alpaca", "math", "bailbench", "stress_test"]),
                layers_to_extract=OOD_LAYERS,
                selectors=OOD_SELECTORS,
                batch_size=OOD_BATCH_SIZE,
                output_dir=output_dir,
                resume=True,
                system_prompt=system_prompt,
                custom_tasks_file=Path(custom_tasks_file) if custom_tasks_file else None,
                task_ids_file=Path(include_task_ids_file) if include_task_ids_file else None,
            )

            print(f"\n--- {condition} (system_prompt={'yes' if system_prompt else 'no'}) ---")
            run_extraction(config)


def run_mra_extractions() -> None:
    print(f"\n{'='*60}")
    print("MRA persona extractions")
    print(f"{'='*60}")

    for config_path_rel in MRA_CONFIGS:
        config_path = PROJECT_ROOT / config_path_rel
        print(f"\n--- {config_path.stem} ---")
        config = ExtractionConfig.from_yaml(config_path, resume=True)
        run_extraction(config)


if __name__ == "__main__":
    run_ood_extractions()
    run_mra_extractions()
    print("\nAll extractions complete!")
