"""Summarise base-model calibration in the canonical contrastive-steering frame.

Each trial contributes two points (see CANONICAL CONTRASTIVE-STEERING PLOTTING
FRAME in src/steering/runner.py): one per task treated as "the steered task".
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(l) for l in path.open() if l.strip()]

buckets = defaultdict(lambda: {"chose": 0, "n": 0})
responded_by_mult = defaultdict(lambda: {"parsed": 0, "n": 0})

for r in rows:
    m = r["signed_multiplier"]
    choice = r["choice_original"]
    responded_by_mult[m]["n"] += 1
    if choice not in ("a", "b"):
        continue
    responded_by_mult[m]["parsed"] += 1
    for applied_c, steered in ((m, "a"), (-m, "b")):
        buckets[applied_c]["n"] += 1
        buckets[applied_c]["chose"] += choice == steered

print(f"{'applied c':>10} {'n':>5} {'P(chose steered)':>18}")
for c in sorted(buckets):
    b = buckets[c]
    print(f"{c:>10.2f} {b['n']:>5} {b['chose'] / b['n']:>18.3f}")

print(f"\n{'c':>7} {'parse rate':>11}")
for m in sorted(responded_by_mult):
    d = responded_by_mult[m]
    print(f"{m:>7.2f} {d['parsed'] / d['n']:>11.2f}")
