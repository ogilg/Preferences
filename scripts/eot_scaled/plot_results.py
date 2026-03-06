"""Generate all analysis plots for EOT scaled experiment."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_DIR = Path("experiments/patching/eot_scaled")
ASSETS_DIR = EXPERIMENT_DIR / "assets"
TASKS_PATH = EXPERIMENT_DIR / "selected_tasks.json"

# Pilot results for comparison
PILOT_LAYER_SWEEP = Path("experiments/patching/pilot/layer_sweep_results.json")

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def load_tasks():
    with open(TASKS_PATH) as f:
        return {t["task_id"]: t for t in json.load(f)}


def majority_choice(choices):
    a = choices.count("a")
    b = choices.count("b")
    if a > b:
        return "a"
    if b > a:
        return "b"
    return None


def load_phase1():
    records = []
    with open(EXPERIMENT_DIR / "phase1_checkpoint.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_phase2():
    records = []
    p = EXPERIMENT_DIR / "phase2_checkpoint.jsonl"
    if not p.exists():
        return []
    with open(p) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_phase3():
    records = []
    p = EXPERIMENT_DIR / "phase3_checkpoint.jsonl"
    if not p.exists():
        return []
    with open(p) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def plot_flip_rate_vs_delta_mu(records, tasks):
    """Plot 1: Flip rate vs |Δμ| scatter."""
    fig, ax = plt.subplots(figsize=(10, 6))

    points = []
    for rec in records:
        base = rec["baseline_choices"]
        patch = rec["patched_choices"]
        if base.count("parse_fail") > 7:
            continue
        base_choice = majority_choice(base)
        if base_choice is None:
            continue

        ta = tasks[rec["task_a_id"]]
        tb = tasks[rec["task_b_id"]]
        delta_mu = abs(ta["mu"] - tb["mu"])

        patch_choice = majority_choice(patch)
        flipped = patch_choice is not None and patch_choice != base_choice

        # Compute shift (proportion change in A choices)
        base_p_a = base.count("a") / max(base.count("a") + base.count("b"), 1)
        patch_p_a = patch.count("a") / max(patch.count("a") + patch.count("b"), 1)

        # Sign-corrected shift: positive = toward expected flip
        if base_choice == "a":
            shift = base_p_a - patch_p_a  # decrease in A = toward flip
        else:
            shift = patch_p_a - base_p_a  # increase in A = toward flip

        points.append((delta_mu, shift, flipped))

    if not points:
        return

    dmus = [p[0] for p in points]
    shifts = [p[1] for p in points]
    flipped = [p[2] for p in points]

    colors = ["#e74c3c" if f else "#3498db" for f in flipped]
    ax.scatter(dmus, shifts, c=colors, alpha=0.3, s=15)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("|Δμ| (absolute utility difference)")
    ax.set_ylabel("Sign-corrected shift (positive = toward flip)")
    ax.set_title(f"EOT Patching: Shift vs |Δμ| (n={len(points)} orderings)")
    ax.set_xlim(0, 21)
    ax.set_ylim(-1.1, 1.1)

    # Add flip rate by bin
    bins = np.linspace(0, 20, 11)
    bin_rates = []
    bin_centers = []
    for i in range(10):
        lo, hi = i * 2, (i + 1) * 2
        in_bin = [p[2] for p in points if lo <= p[0] < hi]
        if in_bin:
            bin_rates.append(sum(in_bin) / len(in_bin))
            bin_centers.append(lo + 1)

    ax2 = ax.twinx()
    ax2.bar(bin_centers, bin_rates, width=1.5, alpha=0.15, color="green", label="Flip rate")
    ax2.set_ylabel("Flip rate (green bars)")
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_030626_shift_vs_delta_mu.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: shift_vs_delta_mu")


def plot_flip_rate_by_delta_mu_bins(records, tasks):
    """Plot 2: Bar chart of flip rate by |Δμ| bins."""
    fig, ax = plt.subplots(figsize=(8, 5))

    bin_data = defaultdict(lambda: [0, 0])  # [flipped, total]
    for rec in records:
        base = rec["baseline_choices"]
        patch = rec["patched_choices"]
        if base.count("parse_fail") > 7:
            continue
        base_choice = majority_choice(base)
        if base_choice is None:
            continue

        ta = tasks[rec["task_a_id"]]
        tb = tasks[rec["task_b_id"]]
        delta_mu = abs(ta["mu"] - tb["mu"])
        bin_idx = min(int(delta_mu / 2), 9)

        patch_choice = majority_choice(patch)
        flipped = patch_choice is not None and patch_choice != base_choice

        bin_data[bin_idx][1] += 1
        if flipped:
            bin_data[bin_idx][0] += 1

    x_labels = [f"{i*2}-{(i+1)*2}" for i in range(10)]
    rates = [bin_data[i][0] / bin_data[i][1] if bin_data[i][1] > 0 else 0 for i in range(10)]
    counts = [bin_data[i][1] for i in range(10)]

    bars = ax.bar(range(10), rates, color="#2196F3", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(10))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("|Δμ| bin")
    ax.set_ylabel("Flip rate")
    ax.set_ylim(0, 1)
    ax.set_title("Flip Rate by Utility Difference")

    for i, (rate, count) in enumerate(zip(rates, counts)):
        ax.text(i, rate + 0.02, f"n={count}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_030626_flip_rate_by_delta_mu.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: flip_rate_by_delta_mu")


def plot_task_specific_effects(records, tasks):
    """Plot 3: Per-task flip count distribution."""
    fig, ax = plt.subplots(figsize=(12, 6))

    task_flips = defaultdict(int)
    task_totals = defaultdict(int)

    for rec in records:
        base = rec["baseline_choices"]
        patch = rec["patched_choices"]
        if base.count("parse_fail") > 7:
            continue
        base_choice = majority_choice(base)
        if base_choice is None:
            continue

        ta_id = rec["task_a_id"]
        tb_id = rec["task_b_id"]
        task_totals[ta_id] += 1
        task_totals[tb_id] += 1

        patch_choice = majority_choice(patch)
        if patch_choice is not None and patch_choice != base_choice:
            task_flips[ta_id] += 1
            task_flips[tb_id] += 1

    # Sort by flip rate
    task_rates = []
    for tid in task_totals:
        if task_totals[tid] >= 5:
            rate = task_flips.get(tid, 0) / task_totals[tid]
            task_rates.append((tid, rate, task_totals[tid], tasks[tid]["mu"]))

    task_rates.sort(key=lambda x: x[1], reverse=True)

    x = range(len(task_rates))
    rates_plot = [t[1] for t in task_rates]
    mus = [t[3] for t in task_rates]

    # Color by mu
    colors = plt.cm.coolwarm([(m + 10) / 20 for m in mus])
    ax.bar(x, rates_plot, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_ylabel("Flip rate")
    ax.set_xlabel("Tasks (sorted by flip rate)")
    ax.set_ylim(0, 1)
    ax.set_title("Per-Task Flip Rate (color = utility, blue=low, red=high)")

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-10, 10))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="μ (Thurstonian)")

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_030626_task_flip_rates.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: task_flip_rates")


def plot_layer_sweep(phase2_records):
    """Plot 4: Per-layer flip rate (bar chart) with probe overlay."""
    if not phase2_records:
        print("Skipping layer sweep plot (no Phase 2 data)")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    layer_flips = defaultdict(int)
    layer_totals = defaultdict(int)
    total = len(phase2_records)

    for rec in phase2_records:
        baseline_chose_a = rec["baseline_chose_a"]
        for layer_str, choices in rec["layer_choices"].items():
            layer = int(layer_str)
            layer_totals[layer] += 1
            patch_choice = majority_choice(choices)
            if patch_choice is None:
                continue
            if (patch_choice == "a") != baseline_chose_a:
                layer_flips[layer] += 1

    layers_sorted = sorted(layer_totals.keys())
    rates = [layer_flips[l] / layer_totals[l] if layer_totals[l] > 0 else 0 for l in layers_sorted]

    # Color causal window differently
    colors = ["#e74c3c" if 25 <= l <= 34 else "#3498db" for l in layers_sorted]
    ax.bar(range(len(layers_sorted)), rates, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(layers_sorted)))
    ax.set_xticklabels([f"L{l}" for l in layers_sorted], rotation=90, fontsize=7)
    ax.set_ylabel("Single-layer flip rate")
    ax.set_ylim(0, max(rates) * 1.3 if rates else 1)
    ax.set_title(f"Per-Layer EOT Patching Flip Rate (n={total} orderings, red=L25-34 causal window)")

    # Overlay pilot layer sweep for comparison if available
    if PILOT_LAYER_SWEEP.exists():
        with open(PILOT_LAYER_SWEEP) as f:
            pilot_data = json.load(f)
        pilot_layers = sorted(int(k) for k in pilot_data["per_layer"])
        pilot_rates = [pilot_data["per_layer"][str(l)]["flip_rate"] for l in pilot_layers]

        # Map pilot layers to x positions
        layer_to_x = {l: i for i, l in enumerate(layers_sorted)}
        pilot_x = [layer_to_x[l] for l in pilot_layers if l in layer_to_x]
        pilot_y = [pilot_data["per_layer"][str(l)]["flip_rate"] for l in pilot_layers if l in layer_to_x]

        ax.plot(pilot_x, pilot_y, "o-", color="orange", markersize=3, linewidth=1, alpha=0.7, label="Pilot")
        ax.legend()

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_030626_layer_sweep.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: layer_sweep")


def plot_layer_combinations(phase3_records):
    """Plot 5: Layer combination flip rates."""
    if not phase3_records:
        print("Skipping layer combo plot (no Phase 3 data)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    combo_flips = defaultdict(int)
    combo_totals = defaultdict(int)

    for rec in phase3_records:
        baseline_chose_a = rec["baseline_chose_a"]
        for combo_name, choices in rec["combo_choices"].items():
            combo_totals[combo_name] += 1
            patch_choice = majority_choice(choices)
            if patch_choice is None:
                continue
            if (patch_choice == "a") != baseline_chose_a:
                combo_flips[combo_name] += 1

    combos = sorted(combo_totals.keys(), key=lambda c: combo_flips[c] / max(combo_totals[c], 1))
    rates = [combo_flips[c] / combo_totals[c] if combo_totals[c] > 0 else 0 for c in combos]

    # Categorize by type
    colors = []
    for c in combos:
        if c.startswith("pair_"):
            colors.append("#3498db")
        elif c.startswith("triple_"):
            colors.append("#2ecc71")
        elif c.startswith("top4_") or c.startswith("top5_"):
            colors.append("#e74c3c")
        elif c == "causal_window":
            colors.append("#9b59b6")
        else:
            colors.append("#95a5a6")

    ax.barh(range(len(combos)), rates, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(combos)))
    ax.set_yticklabels([c.replace("_", " ") for c in combos], fontsize=7)
    ax.set_xlabel("Flip rate")
    ax.set_xlim(0, 1)
    ax.set_title("Layer Combination Flip Rates (blue=pairs, green=triples, red=top-k, purple=window)")

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_030626_layer_combinations.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: layer_combinations")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()

    p1 = load_phase1()
    print(f"Phase 1: {len(p1)} records")

    plot_flip_rate_vs_delta_mu(p1, tasks)
    plot_flip_rate_by_delta_mu_bins(p1, tasks)
    plot_task_specific_effects(p1, tasks)

    p2 = load_phase2()
    print(f"Phase 2: {len(p2)} records")
    plot_layer_sweep(p2)

    p3 = load_phase3()
    print(f"Phase 3: {len(p3)} records")
    plot_layer_combinations(p3)

    print("\nAll plots saved to", ASSETS_DIR)


if __name__ == "__main__":
    main()
