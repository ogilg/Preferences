"""Convert error_prefill messages format to task_prompt/completion format for extraction."""
import json

with open("data/creak/error_prefill_none_100.json") as f:
    data = json.load(f)

converted = []
for d in data:
    converted.append({
        "task_id": d["task_id"],
        "task_prompt": d["messages"][0]["content"],
        "completion": d["messages"][1]["content"],
        **{k: v for k, v in d.items() if k not in ("task_id", "messages")},
    })

with open("/tmp/error_prefill_none_100_converted.json", "w") as f:
    json.dump(converted, f, indent=2)

print(f"Converted {len(converted)} records")
print(json.dumps(converted[0], indent=2))
