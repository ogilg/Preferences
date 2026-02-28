import json

results_file = "/Users/oscargilg/Dev/MATS/Preferences/experiments/steering/stated_steering/format_replication/results/results_qualitative_ternary_last_token.jsonl"

# Inspect first few records
with open(results_file) as f:
    for i, line in enumerate(f):
        if i < 3:
            record = json.loads(line)
            print(f"Record {i}:")
            print(json.dumps(record, indent=2))
            print()
        else:
            break
