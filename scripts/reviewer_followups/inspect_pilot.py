"""Sanity-check the pilot output before committing to the full run."""

import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

print(f"{path.name}: {len(rows)} rows")
print(f"keys: {sorted(rows[0])}")
for field in ["condition", "layer", "norm_at_layer", "signed_multiplier", "sample_idx", "ordering"]:
    print(f"  {field}: {sorted({r[field] for r in rows})}")
print(f"  n_pairs: {len({r['pair_id'] for r in rows})}")
for field in ["compliance", "choice_original", "task_completed"]:
    print(f"  {field}: {dict(Counter(r.get(field, '<absent>') for r in rows))}")

print("\n--- two example responses ---")
for r in rows[:2]:
    print(f"[mult={r['signed_multiplier']} ord={r['ordering']} choice={r['choice_original']}]")
    print(f"  {r['raw_response'][:240]!r}\n")
