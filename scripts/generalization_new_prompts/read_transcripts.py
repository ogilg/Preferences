"""Print side-by-side transcript comparisons for manual reading."""

import json

DATA_PATH = "experiments/steering/program/open_ended_effects/generalization_new_prompts/generation_results.json"

with open(DATA_PATH) as f:
    data = json.load(f)

prompts = {p["id"]: p for p in data["prompts"]}
results_by_prompt = {}
for r in data["results"]:
    key = r["prompt_id"]
    if key not in results_by_prompt:
        results_by_prompt[key] = {}
    results_by_prompt[key][r["coefficient"]] = r["response"]

# Print self-report/affect prompts at -3000, 0, +3000
target_ids = [p["id"] for p in data["prompts"]
              if p["category"] in ("self_report", "affect", "meta_cognitive")]

for pid in target_ids:
    p = prompts[pid]
    print(f"\n{'='*80}")
    print(f"PROMPT {pid} ({p['category']}): {p['text']}")
    print(f"{'='*80}")
    for coef in [-3000, 0, 3000]:
        resp = results_by_prompt[pid].get(coef, "N/A")
        # Truncate to first 500 chars for readability
        display = resp[:500] + "..." if len(resp) > 500 else resp
        print(f"\n--- coef={coef} (len={len(resp)}) ---")
        print(display)
