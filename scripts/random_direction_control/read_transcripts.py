"""Format transcripts for side-by-side reading.

Shows probe -3000, probe +3000, random_200 -3000, random_200 +3000, and baseline
for each prompt.
"""

import json
from pathlib import Path

DATA_PATH = Path("experiments/steering/program/open_ended_effects/random_direction_control/generation_results.json")

with open(DATA_PATH) as f:
    data = json.load(f)

prompts = {p["id"]: p["text"] for p in data["prompts"]}

# Build lookup: (prompt_id, direction, coefficient) -> response
responses = {}
for r in data["results"]:
    key = (r["prompt_id"], r["direction"], r["coefficient"])
    responses[key] = r["response"]

# Show 5 prompts comparing probe vs random_200 vs baseline
comparison_prompts = ["INT_00", "INT_03", "INT_05", "INT_07", "INT_08"]
directions_to_show = ["probe", "random_200", "random_203"]

for pid in comparison_prompts:
    print(f"\n{'='*80}")
    print(f"PROMPT [{pid}]: {prompts[pid]}")
    print(f"{'='*80}")

    # Baseline
    baseline = responses.get((pid, "baseline", 0), "N/A")
    print(f"\n--- BASELINE (coef=0) [{len(baseline)} chars] ---")
    print(baseline[:500])
    if len(baseline) > 500:
        print("...")

    for direction in directions_to_show:
        for coef in [-3000, 3000]:
            resp = responses.get((pid, direction, coef), "N/A")
            print(f"\n--- {direction.upper()} coef={coef:+d} [{len(resp)} chars] ---")
            print(resp[:500])
            if len(resp) > 500:
                print("...")
