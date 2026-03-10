"""Extract Gemma 3 27B PT activations under OOD system prompts (exp1a-1d).

Same extraction as scripts/run_all_extractions.py::run_ood(), but with
model="gemma-3-27b-pt" and output to activations/ood_pt/.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.probes.extraction.config import ExtractionConfig
from src.probes.extraction.extract import run_extraction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_ood_pt() -> None:
    for sub in ["a", "b", "c", "d"]:
        al_dir = PROJECT_ROOT / f"configs/measurement/active_learning/ood_exp1{sub}"
        act_root = PROJECT_ROOT / "activations" / "ood_pt" / (
            "exp1_category" if sub == "a" else "exp1_prompts"
        )

        for al_config_path in sorted(al_dir.glob("*.yaml")):
            condition = al_config_path.stem
            with open(al_config_path) as f:
                al_config = yaml.safe_load(f)

            custom_tasks_file = Path(p) if (p := al_config.get("custom_tasks_file")) else None
            task_origins = None if custom_tasks_file else al_config.get(
                "task_origins",
                ["wildchat", "alpaca", "math", "bailbench", "stress_test"],
            )
            config = ExtractionConfig(
                model="gemma-3-27b-pt",
                n_tasks=al_config["n_tasks"],
                task_origins=task_origins,
                layers_to_extract=[31],
                selectors=["prompt_last"],
                batch_size=32,
                output_dir=str(act_root / condition),
                resume=True,
                system_prompt=al_config.get("measurement_system_prompt"),
                custom_tasks_file=custom_tasks_file,
                task_ids_file=Path(p) if (p := al_config.get("include_task_ids_file")) else None,
            )

            print(f"exp1{sub}/{condition} — {config.n_tasks} tasks")
            run_extraction(config)


if __name__ == "__main__":
    run_ood_pt()
