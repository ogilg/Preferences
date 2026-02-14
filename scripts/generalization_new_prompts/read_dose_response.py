"""Print dose-response for selected prompts across all 5 coefficients."""

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

# Focus on a few prompts where we might see dose-response
focus_ids = ["SR_04", "AF_01", "MC_03", "AF_04"]

for pid in focus_ids:
    p = prompts[pid]
    print(f"\n{'='*80}")
    print(f"PROMPT {pid} ({p['category']}): {p['text']}")
    print(f"{'='*80}")
    for coef in [-3000, -2000, 0, 2000, 3000]:
        resp = results_by_prompt[pid].get(coef, "N/A")
        # Show first 300 chars
        display = resp[:300] + "..." if len(resp) > 300 else resp
        print(f"\n--- coef={coef} (len={len(resp)}) ---")
        print(display)
