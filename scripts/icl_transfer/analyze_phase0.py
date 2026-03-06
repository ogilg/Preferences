"""Analyze phase 0 baseline for ICL transfer experiment.

Computes per-topic marginal P(choose), per-topic-pair P(choose X | X vs Y),
and position bias. Identifies candidate target axes for ICL manipulation.

Run: python -m scripts.icl_transfer.analyze_phase0
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

RESULTS_DIR = Path("results/experiments/exp_20260305_180810/pre_task_revealed")
TASKS_FILE = Path("configs/icl_transfer/icl_50_tasks.json")
ASSETS_DIR = Path("experiments/icl_transfer/assets")


def load_measurements():
    """Load all measurements from canonical and reversed configs."""
    measurements = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        path = run_dir / "measurements.yaml"
        with open(path) as f:
            data = yaml.load(f, Loader=yaml.CSafeLoader)
        order = "canonical" if "canonical" in run_dir.name else "reversed"
        for m in data:
            measurements.append({**m, "order": order})
    return measurements


def main():
    with open(TASKS_FILE) as f:
        tasks = json.load(f)
    topic_of = {t["task_id"]: t["topic"] for t in tasks}
    all_topics = sorted(set(topic_of.values()))
    print(f"Topics ({len(all_topics)}): {all_topics}")

    measurements = load_measurements()
    print(f"Total measurements: {len(measurements)}")

    # --- Per-topic marginal P(choose) ---
    topic_chosen = Counter()
    topic_presented = Counter()
    for m in measurements:
        ta, tb = topic_of[m["task_a"]], topic_of[m["task_b"]]
        topic_presented[ta] += 1
        topic_presented[tb] += 1
        chosen = ta if m["choice"] == "a" else tb
        topic_chosen[chosen] += 1

    print("\n=== Per-topic marginal P(choose) ===")
    marginals = {}
    for t in all_topics:
        p = topic_chosen[t] / topic_presented[t]
        marginals[t] = p
        print(f"  {t:25s}: {p:.3f}  ({topic_chosen[t]}/{topic_presented[t]})")

    # --- Per-topic-pair P(choose X | X vs Y) ---
    # For each measurement between topics X and Y, record which topic won.
    # Both (X,Y) and (Y,X) directions get a trial counted.
    pair_wins: dict[tuple[str, str], int] = defaultdict(int)
    pair_total: dict[tuple[str, str], int] = defaultdict(int)
    for m in measurements:
        ta, tb = topic_of[m["task_a"]], topic_of[m["task_b"]]
        if ta == tb:
            continue
        # This is a trial for both the (ta, tb) and (tb, ta) topic pairs
        pair_total[(ta, tb)] += 1
        pair_total[(tb, ta)] += 1
        if m["choice"] == "a":
            pair_wins[(ta, tb)] += 1
        else:
            pair_wins[(tb, ta)] += 1

    print("\n=== Per-topic-pair P(choose row | row vs col) ===")
    # Build matrix
    n_topics = len(all_topics)
    topic_idx = {t: i for i, t in enumerate(all_topics)}
    pair_prob = np.full((n_topics, n_topics), np.nan)
    pair_n = np.zeros((n_topics, n_topics), dtype=int)

    for (ta, tb), total in pair_total.items():
        i, j = topic_idx[ta], topic_idx[tb]
        wins_a = pair_wins[(ta, tb)]
        pair_prob[i, j] = wins_a / total
        pair_n[i, j] = total

    # Print pairs sorted by asymmetry (most asymmetric first)
    asymmetries = []
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            if np.isnan(pair_prob[i, j]):
                continue
            p_ij = pair_prob[i, j]
            p_ji = pair_prob[j, i]
            n_ij = pair_n[i, j] + pair_n[j, i]
            asymmetry = abs(p_ij - 0.5)
            asymmetries.append((all_topics[i], all_topics[j], p_ij, p_ji, n_ij, asymmetry))

    asymmetries.sort(key=lambda x: x[5], reverse=True)
    print(f"\n{'Topic X':25s} {'Topic Y':25s} {'P(X|XvY)':>8s} {'P(Y|XvY)':>8s} {'N':>6s} {'|P-0.5|':>7s}")
    print("-" * 100)
    for tx, ty, pxy, pyx, n, asym in asymmetries:
        print(f"  {tx:25s} {ty:25s} {pxy:8.3f} {pyx:8.3f} {n:6d} {asym:7.3f}")

    # --- Position bias ---
    first_chosen = sum(1 for m in measurements if m["choice"] == "a")
    print(f"\n=== Position bias ===")
    print(f"  P(choose first): {first_chosen / len(measurements):.3f} ({first_chosen}/{len(measurements)})")

    by_order = defaultdict(lambda: [0, 0])
    for m in measurements:
        order = m["order"]
        by_order[order][0] += 1 if m["choice"] == "a" else 0
        by_order[order][1] += 1
    for order, (first, total) in by_order.items():
        print(f"  {order:12s}: P(choose first) = {first / total:.3f} ({first}/{total})")

    # --- Identify best target axes ---
    print("\n=== Best target axes for ICL manipulation ===")
    print("(Moderate asymmetry: 0.55-0.75, enough room to shift both directions)")
    candidates = [
        (tx, ty, pxy, pyx, n, asym) for tx, ty, pxy, pyx, n, asym in asymmetries
        if 0.05 <= asym <= 0.25
    ]
    candidates.sort(key=lambda x: x[4], reverse=True)  # most data first
    for tx, ty, pxy, pyx, n, asym in candidates[:10]:
        print(f"  {tx:25s} vs {ty:25s}: P(X)={pxy:.3f}, N={n}")

    # --- Save results ---
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "marginals": marginals,
        "pair_probs": {
            f"{tx}_vs_{ty}": {"p_x": float(pxy), "p_y": float(pyx), "n": int(n)}
            for tx, ty, pxy, pyx, n, _ in asymmetries
        },
        "position_bias": first_chosen / len(measurements),
        "n_measurements": len(measurements),
        "candidates": [
            {"topic_x": tx, "topic_y": ty, "p_x": round(float(pxy), 3), "n": int(n)}
            for tx, ty, pxy, pyx, n, asym in candidates[:10]
        ],
    }
    output_path = ASSETS_DIR / "phase0_baseline_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # --- Plot: heatmap of topic-pair probs ---
    fig, ax = plt.subplots(figsize=(12, 10))
    short_topics = [t[:12] for t in all_topics]
    im = ax.imshow(pair_prob, cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n_topics))
    ax.set_yticks(range(n_topics))
    ax.set_xticklabels(short_topics, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_topics, fontsize=8)
    ax.set_xlabel("Topic Y")
    ax.set_ylabel("Topic X")
    ax.set_title("P(choose X | X vs Y) — ICL Phase 0 Baseline")
    plt.colorbar(im, ax=ax, label="P(choose row topic)")
    for i in range(n_topics):
        for j in range(n_topics):
            if not np.isnan(pair_prob[i, j]):
                ax.text(j, i, f"{pair_prob[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(pair_prob[i, j] - 0.5) > 0.2 else "black")
    plt.tight_layout()
    plot_path = ASSETS_DIR / "plot_030526_phase0_topic_pair_heatmap.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved heatmap to {plot_path}")

    # --- Plot: marginal bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_topics = sorted(marginals.keys(), key=lambda t: marginals[t], reverse=True)
    vals = [marginals[t] for t in sorted_topics]
    bars = ax.bar(range(len(sorted_topics)), vals, color="steelblue")
    ax.axhline(1 / len(all_topics), color="gray", linestyle="--", alpha=0.5, label="Uniform")
    ax.set_xticks(range(len(sorted_topics)))
    ax.set_xticklabels(sorted_topics, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("P(choose topic)")
    ax.set_ylim(0, max(vals) * 1.1)
    ax.set_title("Per-topic marginal choice probability — ICL Phase 0 Baseline")
    ax.legend()
    plt.tight_layout()
    plot_path2 = ASSETS_DIR / "plot_030526_phase0_topic_marginals.png"
    plt.savefig(plot_path2, dpi=150)
    print(f"Saved marginals to {plot_path2}")


if __name__ == "__main__":
    main()
