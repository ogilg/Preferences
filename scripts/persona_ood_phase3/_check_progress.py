import json
from pathlib import Path

p = Path("experiments/probe_generalization/persona_ood/phase3/results.json")
if p.exists():
    d = json.load(open(p))
    print(f"Completed conditions: {len(d)}")
    for name, data in d.items():
        rates = data["task_rates"]
        n_tasks = len(rates)
        mean_total = sum(r["n_total"] for r in rates.values()) / n_tasks
        mean_p = sum(r["p_choose"] for r in rates.values()) / n_tasks
        print(f"  {name:30s}  tasks={n_tasks}  mean_n_obs={mean_total:.0f}  mean_p={mean_p:.3f}  dur={data['duration_s']:.0f}s")
else:
    print("No results yet")
