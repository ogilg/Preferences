import json
from collections import defaultdict

import numpy as np

data = json.load(open("experiments/token_level_probes/scoring_results.json"))
items = data["items"]
probe = "task_mean_L39"

scores = defaultdict(list)
for item in items:
    key = (item["domain"], item["condition"])
    scores[key].append(item["critical_span_mean_scores"][probe])

for key in sorted(scores.keys()):
    vals = scores[key]
    print(f"{key}: mean={np.mean(vals):.2f}, n={len(vals)}")
