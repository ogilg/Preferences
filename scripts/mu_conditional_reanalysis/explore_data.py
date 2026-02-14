"""Explore the generation_results.json structure and extract mu values."""
import json
from pathlib import Path

data_path = Path("experiments/steering/program/coefficient_calibration/generation_results.json")
with open(data_path) as f:
    data = json.load(f)

print("=== Parameters ===")
print(f"Coefficients: {data['parameters']['coefficients']}")
print(f"Seeds: {data['parameters']['seeds']}")

print("\n=== Prompts by category ===")
categories = {}
for p in data["prompts"]:
    cat = p["category"]
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(p)

for cat, prompts in sorted(categories.items()):
    print(f"\n{cat}: {len(prompts)} prompts")
    for p in prompts:
        meta = p["metadata"]
        if "mu" in meta:
            print(f"  {p['prompt_id']}: task={meta.get('task_id', 'N/A')}, mu={meta['mu']:.2f}")
        elif "mu_a" in meta:
            print(f"  {p['prompt_id']}: tasks={meta['task_a_id']}+{meta['task_b_id']}, mu_a={meta['mu_a']:.2f}, mu_b={meta['mu_b']:.2f}")
        else:
            print(f"  {p['prompt_id']}: {list(meta.keys())}")

print("\n=== Results structure ===")
r0 = data["results"][0]
print(f"Fields: {list(r0.keys())}")
print(f"Total results: {len(data['results'])}")

# Group B and C tasks by mu
print("\n=== B_rating and C_completion tasks by mu ===")
for cat in ["B_rating", "C_completion"]:
    print(f"\n{cat}:")
    cat_prompts = [p for p in data["prompts"] if p["category"] == cat]
    cat_prompts.sort(key=lambda p: p["metadata"]["mu"])
    for p in cat_prompts:
        mu = p["metadata"]["mu"]
        group = "LOW" if mu < -2 else ("MID" if mu < 4 else "HIGH")
        print(f"  {p['prompt_id']}: task={p['metadata']['task_id']}, mu={mu:.2f} [{group}]")
