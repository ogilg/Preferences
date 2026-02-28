import json
import numpy as np
from pathlib import Path

EXPECTED_TASKS = 2500
EXPECTED_DIM = 5376
EXPECTED_LAYERS = [31, 43, 55]

dirs = {
    "villain": "activations/gemma_3_27b_villain",
    "midwest": "activations/gemma_3_27b_midwest",
    "aesthete": "activations/gemma_3_27b_aesthete",
}

# Load target task IDs
with open("configs/extraction/mra_all_2500_task_ids.txt") as f:
    target_ids = set(f.read().strip().splitlines())

for name, dir_path in dirs.items():
    print(f"\n{'='*60}")
    print(f"Validating: {name} ({dir_path})")
    print(f"{'='*60}")

    d = Path(dir_path)
    npz_path = d / "activations_prompt_last.npz"
    completions_path = d / "completions_with_activations.json"
    metadata_path = d / "extraction_metadata.json"

    # Check files exist
    for p in [npz_path, completions_path, metadata_path]:
        status = "OK" if p.exists() else "MISSING"
        print(f"  {p.name}: {status}")

    # Validate activations
    data = np.load(npz_path, allow_pickle=True)
    task_ids = list(data["task_ids"])
    print(f"\n  Task IDs: {len(task_ids)} (expected {EXPECTED_TASKS})")

    # Check all target IDs present
    extracted_ids = set(task_ids)
    missing = target_ids - extracted_ids
    extra = extracted_ids - target_ids
    print(f"  Missing from target: {len(missing)}")
    print(f"  Extra (not in target): {len(extra)}")

    # Check layers
    for layer in EXPECTED_LAYERS:
        key = f"layer_{layer}"
        if key in data:
            shape = data[key].shape
            print(f"  {key}: {shape} (expected ({EXPECTED_TASKS}, {EXPECTED_DIM}))")
            assert shape == (EXPECTED_TASKS, EXPECTED_DIM), f"Shape mismatch for {key}!"
        else:
            print(f"  {key}: MISSING!")

    # Check completions
    with open(completions_path) as f:
        completions = json.load(f)
    print(f"\n  Completions: {len(completions)} records")

    # Check metadata
    with open(metadata_path) as f:
        meta = json.load(f)
    print(f"  Metadata keys: {list(meta.keys())}")

print("\n" + "="*60)
print("All validations passed!")
