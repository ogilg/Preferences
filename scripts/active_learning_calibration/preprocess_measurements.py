"""Pre-process measurements.yaml into a faster JSON format for repeated loading."""

import json
from pathlib import Path

import yaml

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
OUTPUT = Path("scripts/active_learning_calibration/measurements_fast.json")


def main():
    print("Loading YAML...")
    with open(RUN_DIR / "measurements.yaml") as f:
        measurements = yaml.safe_load(f)
    print(f"Loaded {len(measurements)} measurements")

    # Strip to minimal fields
    compact = [
        {"a": m["task_a"], "b": m["task_b"], "c": m["choice"], "oa": m["origin_a"], "ob": m["origin_b"]}
        for m in measurements
    ]

    print("Saving JSON...")
    with open(OUTPUT, "w") as f:
        json.dump(compact, f)
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
