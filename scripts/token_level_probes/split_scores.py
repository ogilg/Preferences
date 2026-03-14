"""Split all_token_scores from scoring_results.json into a separate .npz file."""

import json
from pathlib import Path

import numpy as np

RESULTS_PATH = Path("experiments/token_level_probes/scoring_results.json")
NPZ_PATH = Path("experiments/token_level_probes/all_token_scores.npz")

data = json.loads(RESULTS_PATH.read_text())

# Build dict for npz: key = "{item_idx}_{probe_name}", value = scores array
npz_dict = {}
for i, item in enumerate(data["items"]):
    item_id = item["id"]
    for probe_name, scores in item["all_token_scores"].items():
        npz_dict[f"{item_id}__{probe_name}"] = np.array(scores, dtype=np.float32)
    # Remove from JSON
    del item["all_token_scores"]

np.savez_compressed(NPZ_PATH, **npz_dict)
print(f"Saved {len(npz_dict)} arrays to {NPZ_PATH} ({NPZ_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

RESULTS_PATH.write_text(json.dumps(data, indent=2))
print(f"Updated {RESULTS_PATH} ({RESULTS_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
