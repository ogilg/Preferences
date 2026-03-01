"""Verify pair ID to task mapping between pairs.json and checkpoint."""

import json
from collections import defaultdict
from pathlib import Path

EXPERIMENT_DIR = Path("experiments/revealed_steering_v2")
PAIRS_PATH = Path("experiments/steering/replication/fine_grained/results/pairs.json")

# Load pairs.json
with open(PAIRS_PATH) as f:
    pairs_data = json.load(f)

# Build mapping from pairs.json
pairs_map = {}
if isinstance(pairs_data, list):
    for i, pair in enumerate(pairs_data):
        pid = pair.get("pair_id", f"pair_{i:04d}")
        pairs_map[pid] = (pair.get("task_a", pair.get("task_a_id")),
                          pair.get("task_b", pair.get("task_b_id")))
elif isinstance(pairs_data, dict):
    for pid, pair in pairs_data.items():
        pairs_map[pid] = (pair.get("task_a", pair.get("task_a_id")),
                          pair.get("task_b", pair.get("task_b_id")))

print(f"pairs.json: {len(pairs_map)} pairs")
print(f"First 3:")
for pid in sorted(pairs_map)[:3]:
    print(f"  {pid}: {pairs_map[pid]}")

# Load checkpoint — get unique pair mappings
checkpoint_map = {}
with open(EXPERIMENT_DIR / "checkpoint.jsonl") as f:
    for line in f:
        t = json.loads(line)
        pid = t["pair_id"]
        if pid not in checkpoint_map:
            checkpoint_map[pid] = (t["task_a_id"], t["task_b_id"])

print(f"\ncheckpoint: {len(checkpoint_map)} pairs")
print(f"First 3:")
for pid in sorted(checkpoint_map)[:3]:
    print(f"  {pid}: {checkpoint_map[pid]}")

# Compare
common = set(pairs_map) & set(checkpoint_map)
print(f"\nCommon pair IDs: {len(common)}")

mismatches = []
for pid in sorted(common):
    if pairs_map[pid] != checkpoint_map[pid]:
        mismatches.append((pid, pairs_map[pid], checkpoint_map[pid]))

if mismatches:
    print(f"\nMISMATCHES: {len(mismatches)}")
    for pid, p, c in mismatches[:10]:
        print(f"  {pid}: pairs.json={p}, checkpoint={c}")
else:
    print("\nAll common pairs match perfectly.")

# Check for pairs in checkpoint but not in pairs.json
only_checkpoint = set(checkpoint_map) - set(pairs_map)
if only_checkpoint:
    print(f"\nPairs in checkpoint but NOT in pairs.json: {len(only_checkpoint)}")
    for pid in sorted(only_checkpoint)[:5]:
        print(f"  {pid}: {checkpoint_map[pid]}")

only_pairs = set(pairs_map) - set(checkpoint_map)
if only_pairs:
    print(f"\nPairs in pairs.json but NOT in checkpoint: {len(only_pairs)}")
    for pid in sorted(only_pairs)[:5]:
        print(f"  {pid}: {pairs_map[pid]}")

# Also check: does the baseline_pairwise.json use the same mapping?
with open(EXPERIMENT_DIR / "baseline_pairwise.json") as f:
    baseline = json.load(f)

baseline_map = {}
for pair in baseline["pairs"]:
    baseline_map[pair["pair_id"]] = (pair["task_a"], pair["task_b"])

print(f"\nbaseline_pairwise.json: {len(baseline_map)} pairs")
common_bl = set(baseline_map) & set(checkpoint_map)
bl_mismatches = []
for pid in sorted(common_bl):
    if baseline_map[pid] != checkpoint_map[pid]:
        bl_mismatches.append((pid, baseline_map[pid], checkpoint_map[pid]))

if bl_mismatches:
    print(f"MISMATCHES vs checkpoint: {len(bl_mismatches)}")
    for pid, b, c in bl_mismatches[:10]:
        print(f"  {pid}: baseline={b}, checkpoint={c}")
else:
    print(f"All {len(common_bl)} common pairs match between baseline and checkpoint.")
