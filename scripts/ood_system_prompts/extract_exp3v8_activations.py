"""Extract activations for all exp3 v8 conditions (121 = baseline + 120).

Run on GPU (RunPod). Each condition gets its own output dir with activations
for the 50 tasks under that system prompt.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.probes.extraction.config import ExtractionConfig
from src.probes.extraction.extract import run_extraction

PROMPTS_PATH = Path("configs/ood/prompts/minimal_pairs_v8.json")
TASKS_PATH = Path("configs/ood/tasks/minimal_pairs_v8_tasks.json")
ACT_ROOT = Path("activations/ood/exp3v8_minimal_pairs")

MODEL = "gemma-3-27b"
LAYERS = [31, 43, 55]
SELECTORS = ["prompt_last"]
BATCH_SIZE = 32


def main():
    with open(PROMPTS_PATH) as f:
        prompts_data = json.load(f)

    # Baseline + all conditions
    conditions = [
        ("baseline", prompts_data["baseline_prompt"]),
    ] + [
        (c["condition_id"], c["system_prompt"])
        for c in prompts_data["conditions"]
    ]

    print(f"Conditions: {len(conditions)}")
    print(f"Output root: {ACT_ROOT}")

    for i, (cid, system_prompt) in enumerate(conditions):
        output_dir = ACT_ROOT / cid
        print(f"\n[{i+1}/{len(conditions)}] {cid}")

        config = ExtractionConfig(
            model=MODEL,
            n_tasks=50,
            task_origins=["wildchat", "alpaca", "math", "bailbench", "stress_test"],
            layers_to_extract=LAYERS,
            selectors=SELECTORS,
            batch_size=BATCH_SIZE,
            output_dir=str(output_dir),
            resume=True,
            system_prompt=system_prompt,
            task_ids_file=TASKS_PATH,
        )

        run_extraction(config)

    print(f"\nAll {len(conditions)} conditions extracted.")


if __name__ == "__main__":
    main()
