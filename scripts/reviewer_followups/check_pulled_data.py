"""Sanity-check the data pulled from storage_pod_oscar for the reviewer follow-ups."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = ROOT / "activations/gemma-3-27b_it/pref_layer_sweep"
RUN = ROOT / (
    "results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0"
)
SPLITS = ROOT / "data/canonical_splits"
TOPICS = ROOT / "data/topics/topics.json"

z = np.load(ACT_DIR / "activations_eot_L23_L32.npz")
act_ids = set(z["task_ids"].tolist())
print(f"activations: {len(act_ids)} tasks, layers={[k for k in z if k != 'task_ids']}")

splits = {
    name: [t.strip() for t in (SPLITS / f"{name}_task_ids.txt").read_text().split() if t.strip()]
    for name in ("train", "eval", "test", "all_6000")
}
for name, ids in splits.items():
    print(f"split {name}: {len(ids)} tasks, {len(set(ids) - act_ids)} missing activations")

pool = set(splits["all_6000"])

completions = json.loads((ACT_DIR / "completions_with_activations.json").read_text())
origin_by_task = {c["task_id"]: c["origin"] for c in completions}
print(f"origins: {Counter(origin_by_task[t] for t in splits['all_6000'])}")

topics = json.loads(TOPICS.read_text())
topic_of = {t: next(iter(topics[t].values()))["primary"] for t in splits["all_6000"] if t in topics}
print(f"topics: {len(topic_of)}/6000 classified, {len(set(topic_of.values()))} distinct")

meas = yaml.safe_load((RUN / "measurements.yaml").read_text())
print(f"measurements: {len(meas)} records, fields={sorted(meas[0])}")

within = [m for m in meas if m["task_a"] in pool and m["task_b"] in pool]
pairs = {frozenset((m["task_a"], m["task_b"])) for m in within}
degree = Counter(t for p in pairs for t in p)
print(f"comparisons within the 6000: {len(within)} over {len(pairs)} unique pairs")
print(f"tasks with >=1 comparison: {len(degree)}/6000, min degree {min(degree.values())}")

util = (RUN / "thurstonian_80fa9dc8.csv").read_text().splitlines()
print(f"thurstonian csv: {len(util) - 1} rows, header={util[0]}")
