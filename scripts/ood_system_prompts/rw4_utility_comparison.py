"""Compare raw Thurstonian utilities for rainy_weather tasks: baseline vs neg."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.ood_system_prompts.analyze_utility_shifts import load_experiment_utilities

df, _ = load_experiment_utilities("exp1b")

# Before demeaning - raw fitted utilities
# load_experiment_utilities already demeans, so let's load raw
from src.measurement.storage.loading import load_run_utilities

BASELINE_DIR = Path("results/experiments/ood_exp1b/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
NEG_DIR = Path("results/experiments/ood_exp1b/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_sysa66278ce")

b_mu, b_ids = load_run_utilities(BASELINE_DIR)
n_mu, n_ids = load_run_utilities(NEG_DIR)

b_raw = dict(zip(b_ids, b_mu))
n_raw = dict(zip(n_ids, n_mu))

print("RAW Thurstonian utilities (before zero-centering):")
print(f"  Baseline mean: {np.mean(b_mu):.3f}")
print(f"  Neg mean: {np.mean(n_mu):.3f}")
print()

print(f"{'task_id':<30} {'baseline':>10} {'neg':>10} {'raw_delta':>10} {'demeaned_delta':>14}")
print("-" * 80)

for tid in sorted(b_raw):
    if "rainy_weather" not in tid:
        continue
    b = b_raw[tid]
    n = n_raw.get(tid, float("nan"))
    raw_d = n - b

    # Demeaned delta (what the report uses)
    b_dm = b - np.mean(b_mu)
    n_dm = n - np.mean(n_mu)
    dm_d = n_dm - b_dm

    print(f"  {tid:<28} {b:>10.3f} {n:>10.3f} {raw_d:>10.3f} {dm_d:>14.3f}")

print()
print("For comparison - cats tasks:")
print(f"{'task_id':<30} {'baseline':>10} {'neg':>10} {'raw_delta':>10} {'demeaned_delta':>14}")
print("-" * 80)

# Load cats_neg
import yaml
for d in Path("results/experiments/ood_exp1b/pre_task_active_learning/").iterdir():
    al = d / "active_learning.yaml"
    if al.exists():
        with open(al) as f:
            cfg = yaml.safe_load(f)
        if "hates cats" in cfg.get("system_prompt", ""):
            cats_dir = d
            break

c_mu, c_ids = load_run_utilities(cats_dir)
c_raw = dict(zip(c_ids, c_mu))

for tid in sorted(b_raw):
    if "cats" not in tid:
        continue
    if tid.startswith("crossed"):
        continue
    b = b_raw[tid]
    c = c_raw.get(tid, float("nan"))
    raw_d = c - b

    b_dm = b - np.mean(b_mu)
    c_dm = c - np.mean(c_mu)
    dm_d = c_dm - b_dm

    print(f"  {tid:<28} {b:>10.3f} {c:>10.3f} {raw_d:>10.3f} {dm_d:>14.3f}")
