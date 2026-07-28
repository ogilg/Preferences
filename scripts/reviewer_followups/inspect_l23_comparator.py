"""Recover L23 run settings from the comparator checkpoints (no config yaml survived)."""

import json
from collections import Counter
from pathlib import Path

CKPT_DIR = Path("experiments/persona_steering_l23_finegrain/checkpoints")

for name in ["sadist_contrastive", "sadist_single_task"]:
    rows = [json.loads(line) for line in (CKPT_DIR / f"{name}.parsed.jsonl").read_text().splitlines()]
    print(f"=== {name} ({len(rows)} rows) ===")
    print(f"  keys: {sorted(rows[0])}")
    for field in ["layer", "norm_at_layer", "condition", "signed_multiplier", "sample_idx", "ordering"]:
        vals = sorted({r[field] for r in rows}, key=lambda v: (str(type(v)), v))
        print(f"  {field}: {vals}")
    print(f"  n_pairs: {len({r['pair_id'] for r in rows})}")
    print(f"  compliance: {dict(Counter(r.get('compliance', '<absent>') for r in rows))}")
    lens = sorted(len(r["raw_response"].split()) for r in rows)
    print(f"  raw_response word count: min={lens[0]} median={lens[len(lens) // 2]} max={lens[-1]}")
    print()
