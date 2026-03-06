"""Export checkpoint JSONL files to the spec-required JSON output format."""

import json
from pathlib import Path

EXPERIMENT_DIR = Path("experiments/patching/eot_scaled")


def export_phase1():
    """Export phase1_checkpoint.jsonl -> phase1_results.json"""
    records = []
    with open(EXPERIMENT_DIR / "phase1_checkpoint.jsonl") as f:
        for line in f:
            records.append(json.loads(line))

    with open(EXPERIMENT_DIR / "phase1_results.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Phase 1: {len(records)} orderings -> phase1_results.json")


def export_phase2():
    """Export phase2_checkpoint.jsonl -> phase2_results.json"""
    records = []
    with open(EXPERIMENT_DIR / "phase2_checkpoint.jsonl") as f:
        for line in f:
            records.append(json.loads(line))

    with open(EXPERIMENT_DIR / "phase2_results.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Phase 2: {len(records)} orderings -> phase2_results.json")


def export_phase3():
    """Export phase3_checkpoint.jsonl -> phase3_results.json"""
    records = []
    with open(EXPERIMENT_DIR / "phase3_checkpoint.jsonl") as f:
        for line in f:
            records.append(json.loads(line))

    with open(EXPERIMENT_DIR / "phase3_results.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Phase 3: {len(records)} orderings -> phase3_results.json")


def main():
    if (EXPERIMENT_DIR / "phase1_checkpoint.jsonl").exists():
        export_phase1()
    if (EXPERIMENT_DIR / "phase2_checkpoint.jsonl").exists():
        export_phase2()
    if (EXPERIMENT_DIR / "phase3_checkpoint.jsonl").exists():
        export_phase3()


if __name__ == "__main__":
    main()
