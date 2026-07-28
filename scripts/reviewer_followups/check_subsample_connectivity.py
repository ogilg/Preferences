"""How often does a stratified subsample of default_train keep a connected comparison graph?

Spec A says to discard disconnected seeds and redraw. If disconnection is the norm
rather than the exception at 50%, that redraw loop never terminates and the spec
needs revising.
"""

import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning"
SPLITS = ROOT / "data/canonical_splits"
TOPICS = ROOT / "data/topics/topics.json"
ACT_DIR = ROOT / "activations/gemma-3-27b_it/pref_layer_sweep"

N_SEEDS = 20

ids = [t.strip() for t in (SPLITS / "train_task_ids.txt").read_text().split() if t.strip()]
run = SWEEP / "completion_preference_gemma-3-27b_completion_canonical_seed0_train_task_ids"
meas = yaml.safe_load((run / "measurements.yaml").read_text())
pairs = [tuple(p) for p in {frozenset((m["task_a"], m["task_b"])) for m in meas}]

origin = {c["task_id"]: c["origin"] for c in json.loads((ACT_DIR / "completions_with_activations.json").read_text())}
topics = json.loads(TOPICS.read_text())
stratum = {t: (origin[t], next(iter(topics[t].values()))["primary"]) for t in ids}

buckets = defaultdict(list)
for t in ids:
    buckets[stratum[t]].append(t)


def largest_component(kept, edges):
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen, best = set(), 0
    for start in kept:
        if start in seen:
            continue
        comp, queue = 0, deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            comp += 1
            for nxt in adj[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        best = max(best, comp)
    return best


for frac in (0.5, 0.8):
    stats = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        kept = set()
        for members in buckets.values():
            k = round(frac * len(members))
            kept.update(rng.choice(members, size=k, replace=False).tolist())
        edges = [(a, b) for a, b in pairs if a in kept and b in kept]
        stats.append((len(kept), len(edges), largest_component(kept, edges)))

    n_kept = Counter(s[0] for s in stats).most_common(1)[0][0]
    mean_edges = np.mean([s[1] for s in stats])
    frac_lcc = np.array([s[2] / s[0] for s in stats])
    n_connected = int((frac_lcc == 1.0).sum())
    print(
        f"{int(frac * 100)}%: {n_kept} tasks, {mean_edges:.0f} retained pairs "
        f"(mean degree {2 * mean_edges / n_kept:.2f})\n"
        f"     largest component covers {frac_lcc.mean():.1%} of kept tasks "
        f"(min {frac_lcc.min():.1%}); fully connected in {n_connected}/{N_SEEDS} seeds"
    )
