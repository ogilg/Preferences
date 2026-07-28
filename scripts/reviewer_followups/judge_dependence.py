"""How much does the outcome depend on the LLM judge?

_effective_choice uses choice_original first and only falls back to the judge's
compliance/task_completed fields. Quantify, on the published comparator, how many
rows the judge actually rescues — i.e. how much would be lost if the new arm were
scored without a judge pass.
"""

import json
from pathlib import Path

CKPT = Path("experiments/persona_steering_l23_finegrain/checkpoints")

for name in ["sadist_contrastive", "sadist_single_task"]:
    rows = [json.loads(line) for line in (CKPT / f"{name}.parsed.jsonl").read_text().splitlines() if line.strip()]
    direct = judge_rescued = refusal = 0
    for r in rows:
        if r["choice_original"] in ("a", "b"):
            direct += 1
        elif r.get("compliance") == "truncated" and r.get("task_completed") in ("a", "b"):
            judge_rescued += 1
        else:
            refusal += 1
    n = len(rows)
    print(f"{name}: {n} rows")
    print(f"  scored from choice_original alone : {direct:6d} ({direct / n:6.1%})")
    print(f"  rescued only by the judge        : {judge_rescued:6d} ({judge_rescued / n:6.1%})")
    print(f"  refusal / unscorable             : {refusal:6d} ({refusal / n:6.1%})")
    print()
