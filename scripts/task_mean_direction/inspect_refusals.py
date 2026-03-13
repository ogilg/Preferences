"""Inspect refusals and steering fallbacks in the checkpoint."""

import json

refusals = []
for line in open("experiments/steering/task_mean_direction/checkpoint.jsonl"):
    r = json.loads(line)
    if r.get("choice_presented") == "refusal" or r.get("steering_fallback"):
        refusals.append(r)

print(f"Total refusals/fallbacks: {len(refusals)}")
for r in refusals[:5]:
    print(f"  pair={r['pair_id']} L{r['layer']} m={r['multiplier']} o={r['ordering']}")
    print(f"  fallback={r['steering_fallback']} choice={r['choice_presented']}")
    print(f"  response: {r['raw_response'][:300]}")
    print()
