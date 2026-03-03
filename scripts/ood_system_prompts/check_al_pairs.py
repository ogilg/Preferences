"""Check what the active learning algorithm compared for rainy_weather_neg."""

import yaml
from pathlib import Path
from collections import defaultdict

RUN_DIR = Path("results/experiments/ood_exp1b/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_sysa66278ce")

with open(RUN_DIR / "measurements.yaml") as f:
    measurements = yaml.safe_load(f)

print(f"Total comparisons: {len(measurements)}")

# Count comparisons per task
task_counts = defaultdict(int)
task_wins = defaultdict(int)
task_losses = defaultdict(int)

# Track rainy-weather-involved comparisons
rw_comparisons = []

for m in measurements:
    ta, tb = m["task_a"], m["task_b"]
    choice = m["choice"]
    task_counts[ta] += 1
    task_counts[tb] += 1

    if choice == "a":
        task_wins[ta] += 1
        task_losses[tb] += 1
    elif choice == "b":
        task_wins[tb] += 1
        task_losses[ta] += 1

    if "rainy_weather" in ta or "rainy_weather" in tb:
        rw_comparisons.append(m)

print(f"\nComparisons involving rainy_weather tasks: {len(rw_comparisons)}")

# How many are rainy-vs-rainy?
rw_vs_rw = [m for m in rw_comparisons
            if "rainy_weather" in m["task_a"] and "rainy_weather" in m["task_b"]]
print(f"Rainy-vs-rainy comparisons: {len(rw_vs_rw)}")

for m in rw_vs_rw:
    winner = m["task_a"] if m["choice"] == "a" else m["task_b"]
    loser = m["task_b"] if m["choice"] == "a" else m["task_a"]
    print(f"  {winner} > {loser}")

# Per rainy_weather task stats
print(f"\nPer rainy_weather task:")
for tid in sorted(task_counts):
    if "rainy_weather" not in tid:
        continue
    w = task_wins[tid]
    l = task_losses[tid]
    n = task_counts[tid]
    print(f"  {tid}: {n} comparisons, {w}W/{l}L")

# Compare to cats
print(f"\nPer cats task (for comparison):")
for tid in sorted(task_counts):
    if "cats" not in tid:
        continue
    w = task_wins[tid]
    l = task_losses[tid]
    n = task_counts[tid]
    print(f"  {tid}: {n} comparisons, {w}W/{l}L")

# What did rainy_weather_4 specifically win against?
print(f"\nAll comparisons involving hidden_rainy_weather_4:")
for m in measurements:
    ta, tb = m["task_a"], m["task_b"]
    if "rainy_weather_4" not in ta and "rainy_weather_4" not in tb:
        continue
    choice = m["choice"]
    winner = ta if choice == "a" else tb
    loser = tb if choice == "a" else ta
    rw4_won = "rainy_weather_4" in winner
    print(f"  {'WIN' if rw4_won else 'LOSS'}: {winner} > {loser}")
