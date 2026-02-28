import json
import os

# Check completions
with open("activations/gemma_3_27b/completions_with_activations.json") as f:
    data = json.load(f)
print(f"Total completions: {len(data)}")

# Check task IDs overlap
with open("configs/extraction/mra_all_2500_task_ids.txt") as f:
    target_ids = set(f.read().strip().splitlines())
print(f"Target task IDs: {len(target_ids)}")

completion_ids = {r["task_id"] for r in data if "task_id" in r}
overlap = target_ids & completion_ids
print(f"Overlap: {len(overlap)}")
if len(overlap) < len(target_ids):
    missing = target_ids - completion_ids
    print(f"Missing {len(missing)} task IDs from completions")

# Check model cache
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
if os.path.isdir(cache_dir):
    entries = os.listdir(cache_dir)
    gemma_entries = [e for e in entries if "gemma" in e.lower()]
    print(f"\nHF cache entries: {len(entries)}")
    print(f"Gemma entries: {gemma_entries}")
else:
    print("\nNo HF cache directory found")
