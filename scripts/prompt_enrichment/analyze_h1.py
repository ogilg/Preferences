"""Analyze enrichment results: deltas from baseline by topic (fixed tasks)."""

import json
import sys
from collections import defaultdict

RESULTS_PATH = sys.argv[1] if len(sys.argv) > 1 else "experiments/probe_generalization/persona_ood/prompt_enrichment/h1_results_fixed.json"

with open(RESULTS_PATH) as f:
    data = json.load(f)

sampled_tasks = data["baseline"]["sampled_tasks"]
conditions = [k for k in data if k != "baseline"]

# Compute topic means for all conditions
all_topic_means = {}
for cond_name in ["baseline"] + conditions:
    rates = data[cond_name]["task_rates"]
    topic_vals: dict[str, list[float]] = defaultdict(list)
    for tid, info in rates.items():
        topic = sampled_tasks.get(tid, "unknown")
        if info["n_total"] > 0:
            topic_vals[topic].append(info["p_choose"])
    all_topic_means[cond_name] = {t: sum(vs)/len(vs) for t, vs in topic_vals.items()}

# Truncate condition names for display
def short_name(name):
    return name[:16]

topics = sorted(set(sampled_tasks.values()))

print("=== Per-topic mean p_choose ===")
header = f"  {'topic':<25} {'baseline':>10}"
for c in conditions:
    header += f" {short_name(c):>16}"
print(header)

for topic in topics:
    b = all_topic_means["baseline"].get(topic)
    if b is None:
        continue
    row = f"  {topic:<25} {b:>10.3f}"
    for c in conditions:
        v = all_topic_means[c].get(topic)
        row += f" {v:>16.3f}" if v is not None else f" {'N/A':>16}"
    print(row)

print("\n=== Deltas from baseline ===")
header = f"  {'topic':<25}"
for c in conditions:
    header += f" {short_name(c):>16}"
print(header)

for topic in topics:
    b = all_topic_means["baseline"].get(topic)
    if b is None:
        continue
    row = f"  {topic:<25}"
    for c in conditions:
        v = all_topic_means[c].get(topic)
        if v is not None:
            row += f" {v - b:>+16.3f}"
        else:
            row += f" {'N/A':>16}"
    print(row)

# Per-task top movers
print("\n=== Top 10 tasks by |delta| per condition ===")
baseline_rates = data["baseline"]["task_rates"]
for cond_name in conditions:
    rates = data[cond_name]["task_rates"]
    deltas = []
    for tid in baseline_rates:
        if tid in rates and baseline_rates[tid]["n_total"] > 0 and rates[tid]["n_total"] > 0:
            d = rates[tid]["p_choose"] - baseline_rates[tid]["p_choose"]
            topic = sampled_tasks.get(tid, "?")
            deltas.append((tid, topic, d))
    deltas.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\n--- {cond_name} ---")
    for tid, topic, d in deltas[:10]:
        print(f"  {tid:<40} {topic:<20} {d:>+.3f}")
