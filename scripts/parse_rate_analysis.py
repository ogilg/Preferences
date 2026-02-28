import json
from collections import defaultdict

results_file = "/Users/oscargilg/Dev/MATS/Preferences/experiments/steering/stated_steering/format_replication/results/results_qualitative_ternary_last_token.jsonl"

# Track parse rates per coefficient
coef_stats = defaultdict(lambda: {"non_null": 0, "total": 0})

with open(results_file) as f:
    for line in f:
        record = json.loads(line)
        coef = record.get("coefficient")
        scores = record.get("scores", [])

        # Count non-null scores (non-null means the score is not None)
        total = len(scores)
        non_null = sum(1 for s in scores if s is not None)

        coef_stats[coef]["non_null"] += non_null
        coef_stats[coef]["total"] += total

# Sort by coefficient
sorted_coefs = sorted(coef_stats.keys())

# Print table
print(f"{'Coefficient':<20} {'Parse Rate':<12} {'n_total':<10}")
print("-" * 42)
for coef in sorted_coefs:
    stats = coef_stats[coef]
    parse_rate = stats["non_null"] / stats["total"] if stats["total"] > 0 else 0
    print(f"{coef!s:<20} {parse_rate:<12.4f} {stats['total']:<10}")
