"""P(chose steered) by injection layer, in the canonical contrastive frame."""
import json
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(l) for l in path.open() if l.strip()]

# layer -> applied_c -> [chose, n]
by_layer = defaultdict(lambda: defaultdict(lambda: [0, 0]))
parse = defaultdict(lambda: [0, 0])

for r in rows:
    L, m, choice = r["layer"], r["signed_multiplier"], r["choice_original"]
    parse[L][1] += 1
    if choice not in ("a", "b"):
        continue
    parse[L][0] += 1
    for applied_c, steered in ((m, "a"), (-m, "b")):
        cell = by_layer[L][applied_c]
        cell[0] += choice == steered
        cell[1] += 1

coefs = sorted({c for d in by_layer.values() for c in d})
head = "  ".join(f"{c:>+6.2f}" for c in coefs)
print(f"{'layer':>5}  {head}  {'swing':>7}  {'parse':>6}")
for L in sorted(by_layer):
    d = by_layer[L]
    cells = [d[c][0] / d[c][1] for c in coefs]
    swing = cells[-1] - cells[0]
    n_ok, n = parse[L]
    print("  ".join([f"{L:>5}"] + [f"{v:>6.3f}" for v in cells])
          + f"  {swing:>+7.3f}  {n_ok / n:>6.2f}")
print(f"\nn per (layer, coefficient) cell: {by_layer[min(by_layer)][coefs[0]][1]}")
