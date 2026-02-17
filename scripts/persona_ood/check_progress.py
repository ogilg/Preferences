"""Check progress of full-scale measurement runs."""

import json
from pathlib import Path

PART_A = Path("experiments/probe_generalization/persona_ood/full_results_part_a.json")
PART_B = Path("experiments/probe_generalization/persona_ood/full_results_part_b.json")

for label, path, total in [("Part A", PART_A, 36), ("Part B", PART_B, 16)]:
    if not path.exists():
        print(f"{label}: not found")
        continue
    with open(path) as f:
        data = json.load(f)
    n = len(data)
    non_baseline = [k for k in data if k != "baseline"]
    durations = [data[k]["duration_s"] for k in non_baseline]
    avg_dur = sum(durations) / len(durations) if durations else 0
    remaining = total - n
    eta_min = remaining * avg_dur / 60 if avg_dur > 0 else 0

    print(f"{label}: {n}/{total} conditions ({len(non_baseline)} non-baseline)")
    if durations:
        print(f"  Avg duration: {avg_dur:.0f}s ({avg_dur/60:.1f} min)")
        print(f"  ETA: {eta_min:.0f} min ({eta_min/60:.1f} hours)")
    for name in non_baseline[-3:]:
        d = data[name]
        print(f"  Latest: {name} ({d['duration_s']:.0f}s, {d['n_raw_results']} results)")
