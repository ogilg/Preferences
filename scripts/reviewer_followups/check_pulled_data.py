"""Sanity-check the data pulled from storage_pod_oscar for the reviewer follow-ups.

The relevant universe is the canonical 4000/1000/1000 splits under the default
(no system prompt) Assistant: utilities from `persona_sweep_final_six`, activations
from the 6000-task `pref_layer_sweep` end-of-turn extraction.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = ROOT / "activations/gemma-3-27b_it/pref_layer_sweep"
SWEEP = ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning"
SPLITS = ROOT / "data/canonical_splits"
TOPICS = ROOT / "data/topics/topics.json"

z = np.load(ACT_DIR / "activations_eot_L23_L32.npz")
act_ids = set(z["task_ids"].tolist())
print(f"activations: {len(act_ids)} tasks, layers={[k for k in z if k != 'task_ids']}")

completions = json.loads((ACT_DIR / "completions_with_activations.json").read_text())
origin_by_task = {c["task_id"]: c["origin"] for c in completions}
topics = json.loads(TOPICS.read_text())

for split in ("train", "eval", "test"):
    ids = [t.strip() for t in (SPLITS / f"{split}_task_ids.txt").read_text().split() if t.strip()]
    run = SWEEP / f"completion_preference_gemma-3-27b_completion_canonical_seed0_{split}_task_ids"

    meas = yaml.safe_load((run / "measurements.yaml").read_text())
    pool = set(ids)
    pairs = {frozenset((m["task_a"], m["task_b"])) for m in meas}
    degree = Counter(t for p in pairs for t in p)
    endpoints = {t for p in pairs for t in p}

    (fit,) = run.glob("thurstonian_*.csv")
    utils = {r["task_id"]: float(r["mu"]) for r in csv.DictReader(fit.open())}

    n_topics = len({next(iter(topics[t].values()))["primary"] for t in ids if t in topics})
    print(
        f"\n{split}: {len(ids)} tasks | {len(ids) - len(pool & act_ids)} missing activations\n"
        f"  comparisons {len(meas)} over {len(pairs)} unique pairs; "
        f"endpoints outside split: {len(endpoints - pool)}\n"
        f"  tasks with >=1 comparison: {len(degree)}/{len(ids)}, min degree {min(degree.values())}\n"
        f"  utilities {len(utils)} ({len(pool - set(utils))} split tasks unfitted)\n"
        f"  origins {dict(Counter(origin_by_task[t] for t in ids))}, {n_topics} topics"
    )
