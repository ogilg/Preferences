"""Comprehensive analysis of scaled EOT patching experiment."""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

EXPERIMENT_DIR = Path("experiments/patching/eot_scaled")
PILOT_DIR = Path("experiments/patching/pilot")
ASSETS = EXPERIMENT_DIR / "assets"
ASSETS.mkdir(exist_ok=True)

# Load data
with open(EXPERIMENT_DIR / "selected_tasks.json") as f:
    tasks = json.load(f)
task_by_id = {t["task_id"]: t for t in tasks}

phase1 = []
with open(EXPERIMENT_DIR / "phase1_checkpoint.jsonl") as f:
    for line in f:
        phase1.append(json.loads(line))

phase2 = []
with open(EXPERIMENT_DIR / "phase2_checkpoint.jsonl") as f:
    for line in f:
        phase2.append(json.loads(line))

# Load pilot layer sweep for comparison
with open(PILOT_DIR / "layer_sweep_results.json") as f:
    pilot_sweep = json.load(f)

# Load probe results for overlay
probe_manifest = Path("results/probes/gemma3_10k_heldout_std_raw/manifest.json")
probe_r = {}
if probe_manifest.exists():
    with open(probe_manifest) as f:
        manifest = json.load(f)
    for entry in manifest.get("probes", []):
        layer = entry.get("layer")
        r = entry.get("heldout_r")
        if layer is not None and r is not None:
            probe_r[layer] = r

print(f"Phase 1: {len(phase1)} orderings")
print(f"Phase 2: {len(phase2)} orderings")
print(f"Tasks: {len(tasks)}")

# ============================================================
# Phase 1 Analysis
# ============================================================

# Compute per-ordering stats
analyzable = []
for rec in phase1:
    base = rec["baseline_choices"]
    patch = rec["patched_choices"]
    base_a = base.count("a")
    base_b = base.count("b")
    patch_a = patch.count("a")
    patch_b = patch.count("b")

    # Skip parse failures
    base_valid = base_a + base_b
    patch_valid = patch_a + patch_b
    if base_valid == 0 or patch_valid == 0:
        continue

    # Skip ambiguous baselines
    if base_a == base_b:
        continue

    baseline_chose_a = base_a > base_b
    patched_chose_a = patch_a > patch_b if patch_a != patch_b else None

    # Compute delta_mu
    ta = task_by_id[rec["task_a_id"]]
    tb = task_by_id[rec["task_b_id"]]

    if rec["direction"] == "ab":
        pos_a_mu, pos_b_mu = ta["mu"], tb["mu"]
    else:
        pos_a_mu, pos_b_mu = tb["mu"], ta["mu"]

    delta_mu = abs(ta["mu"] - tb["mu"])

    flipped = patched_chose_a is not None and baseline_chose_a != patched_chose_a

    # Signed shift: proportion change toward expected direction
    p_a_base = base_a / base_valid
    p_a_patch = patch_a / patch_valid
    shift = p_a_patch - p_a_base

    # Expected direction: if baseline chose A, we expect patch to shift toward B (negative shift)
    expected_sign = -1 if baseline_chose_a else 1
    signed_shift = shift * expected_sign

    analyzable.append({
        "task_a_id": rec["task_a_id"],
        "task_b_id": rec["task_b_id"],
        "direction": rec["direction"],
        "delta_mu": delta_mu,
        "baseline_chose_a": baseline_chose_a,
        "flipped": flipped,
        "p_a_base": p_a_base,
        "p_a_patch": p_a_patch,
        "shift": shift,
        "signed_shift": signed_shift,
    })

n_flipped = sum(1 for r in analyzable if r["flipped"])
print(f"\nPhase 1: {len(analyzable)} analyzable, {n_flipped} flipped ({n_flipped/len(analyzable):.1%})")

# --- Plot 1: Layer profile (scaled vs pilot) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scaled layer profile
layer_flips = defaultdict(int)
layer_total = defaultdict(int)
for rec in phase2:
    baseline_a = rec["baseline_chose_a"]
    for layer_str, choices in rec["layer_choices"].items():
        layer = int(layer_str)
        layer_total[layer] += 1
        a_count = choices.count("a")
        b_count = choices.count("b")
        if a_count == b_count:
            continue
        patched_chose_a = a_count > b_count
        if baseline_a is not patched_chose_a:
            layer_flips[layer] += 1

layers_sorted = sorted(layer_total.keys())
flip_rates = [layer_flips[l] / layer_total[l] for l in layers_sorted]

ax = axes[0]
bars = ax.bar(layers_sorted, flip_rates, color="steelblue", alpha=0.8, width=0.8)
# Highlight causal window
for bar, layer in zip(bars, layers_sorted):
    if 28 <= layer <= 34:
        bar.set_color("royalblue")
    elif 26 <= layer <= 27:
        bar.set_color("cornflowerblue")

# Overlay probe r
if probe_r:
    ax2 = ax.twinx()
    probe_layers = sorted(probe_r.keys())
    probe_vals = [probe_r[l] for l in probe_layers]
    ax2.plot(probe_layers, probe_vals, "ro-", markersize=6, linewidth=2, label="Probe r")
    ax2.set_ylabel("Probe r (Pearson)", color="red")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper right")

ax.set_xlabel("Layer")
ax.set_ylabel("Flip rate")
ax.set_title(f"Scaled: Per-layer flip rate (n={len(phase2)})")
ax.set_ylim(0, 1)
ax.axhline(y=n_flipped/len(analyzable), color="gray", linestyle="--", alpha=0.5, label=f"All-layer: {n_flipped/len(analyzable):.0%}")
ax.legend()

# Pilot layer profile
ax = axes[1]
pilot_layers = sorted(int(k) for k in pilot_sweep["per_layer"].keys())
pilot_rates = [pilot_sweep["per_layer"][str(l)]["flip_rate"] for l in pilot_layers]
bars = ax.bar(pilot_layers, pilot_rates, color="coral", alpha=0.8, width=0.8)
for bar, layer in zip(bars, pilot_layers):
    if 28 <= layer <= 34:
        bar.set_color("tomato")

if probe_r:
    ax2 = ax.twinx()
    ax2.plot(probe_layers, probe_vals, "ro-", markersize=6, linewidth=2, label="Probe r")
    ax2.set_ylabel("Probe r (Pearson)", color="red")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor="red")

ax.set_xlabel("Layer")
ax.set_ylabel("Flip rate")
ax.set_title(f"Pilot: Per-layer flip rate (n={pilot_sweep['n_orderings_tested']})")
ax.set_ylim(0, 1)
ax.axhline(y=0.54, color="gray", linestyle="--", alpha=0.5, label="All-layer: 54%")
ax.legend()

plt.tight_layout()
plt.savefig(ASSETS / "plot_030626_layer_profile_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved layer profile comparison")

# --- Plot 2: Flip rate vs |Δμ| (Phase 1) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Binned flip rate
bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 20)]
bin_labels = [f"{lo}-{hi}" for lo, hi in bins]
bin_flip_rates = []
bin_counts = []
for lo, hi in bins:
    in_bin = [r for r in analyzable if lo <= r["delta_mu"] < hi]
    if in_bin:
        fr = sum(1 for r in in_bin if r["flipped"]) / len(in_bin)
        bin_flip_rates.append(fr)
        bin_counts.append(len(in_bin))
    else:
        bin_flip_rates.append(0)
        bin_counts.append(0)

ax = axes[0]
bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bins)))
bars = ax.bar(range(len(bins)), bin_flip_rates, color=bar_colors, alpha=0.85)
ax.set_xticks(range(len(bins)))
ax.set_xticklabels(bin_labels, rotation=45)
ax.set_xlabel("|Δμ|")
ax.set_ylabel("Flip rate")
ax.set_title("All-layer EOT patch: Flip rate by |Δμ|")
ax.set_ylim(0, 1)
for bar, count in zip(bars, bin_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"n={count}", ha="center", va="bottom", fontsize=8)

# Signed shift scatter
ax = axes[1]
delta_mus = [r["delta_mu"] for r in analyzable]
signed_shifts = [r["signed_shift"] for r in analyzable]
colors = ["royalblue" if r["flipped"] else "lightgray" for r in analyzable]
ax.scatter(delta_mus, signed_shifts, c=colors, alpha=0.3, s=10, edgecolors="none")
ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xlabel("|Δμ|")
ax.set_ylabel("Signed shift (+ = toward expected reversal)")
ax.set_title(f"Signed shift vs |Δμ| (n={len(analyzable)})")
ax.set_ylim(-1.2, 1.2)

plt.tight_layout()
plt.savefig(ASSETS / "plot_030626_flip_rate_vs_delta_mu.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved flip rate vs delta_mu")

# --- Plot 3: Per-task flip rate distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Per-task flip rate
task_flips = defaultdict(int)
task_total = defaultdict(int)
for r in analyzable:
    for tid in [r["task_a_id"], r["task_b_id"]]:
        task_total[tid] += 1
        if r["flipped"]:
            task_flips[tid] += 1

task_rates = {tid: task_flips[tid] / task_total[tid] for tid in task_total}
task_mus = {tid: task_by_id[tid]["mu"] for tid in task_total}

ax = axes[0]
sorted_tasks = sorted(task_rates.keys(), key=lambda t: task_mus[t])
mus = [task_mus[t] for t in sorted_tasks]
rates = [task_rates[t] for t in sorted_tasks]
ax.scatter(mus, rates, c="steelblue", alpha=0.6, s=30)
ax.set_xlabel("Task utility (μ)")
ax.set_ylabel("Flip rate (across all pairs involving task)")
ax.set_title("Per-task flip rate vs utility")
ax.set_ylim(0, 1)
ax.axhline(y=n_flipped/len(analyzable), color="gray", linestyle="--", alpha=0.5)

# Histogram of per-task flip rates
ax = axes[1]
ax.hist(list(task_rates.values()), bins=20, color="steelblue", alpha=0.7, edgecolor="white")
ax.set_xlabel("Flip rate")
ax.set_ylabel("Number of tasks")
ax.set_title(f"Distribution of per-task flip rates (n={len(task_rates)} tasks)")
ax.axvline(x=n_flipped/len(analyzable), color="gray", linestyle="--", alpha=0.5, label=f"Mean: {n_flipped/len(analyzable):.0%}")
ax.legend()

plt.tight_layout()
plt.savefig(ASSETS / "plot_030626_per_task_flip_rates.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved per-task flip rates")

# --- Plot 4: Layer profile zoomed into causal window ---
fig, ax = plt.subplots(figsize=(10, 5))

window_layers = [l for l in layers_sorted if 20 <= l <= 40]
window_rates = [layer_flips[l] / layer_total[l] for l in window_layers]

bars = ax.bar(window_layers, window_rates, color="steelblue", alpha=0.8)
for bar, layer in zip(bars, window_layers):
    rate = layer_flips[layer] / layer_total[layer]
    if rate > 0.5:
        bar.set_color("royalblue")
    elif rate > 0.1:
        bar.set_color("cornflowerblue")
    # Add rate label on top
    if rate > 0.01:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{rate:.0%}", ha="center", va="bottom", fontsize=8)

if probe_r:
    ax2 = ax.twinx()
    pl = [l for l in probe_layers if 20 <= l <= 40]
    pv = [probe_r[l] for l in pl]
    ax2.plot(pl, pv, "ro-", markersize=8, linewidth=2, label="Probe r", zorder=5)
    ax2.set_ylabel("Probe r (Pearson)", color="red")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper right")

ax.set_xlabel("Layer")
ax.set_ylabel("Flip rate")
ax.set_title(f"Causal window detail: Per-layer flip rate (n={len(phase2)})")
ax.set_ylim(0, 1)
ax.axhline(y=n_flipped/len(analyzable), color="gray", linestyle="--", alpha=0.3, label=f"All-layer: {n_flipped/len(analyzable):.0%}")
ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig(ASSETS / "plot_030626_causal_window_detail.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved causal window detail")

# --- Plot 5: Position bias and baseline behavior ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Position bias
p_a_values = [r["p_a_base"] for r in analyzable]
ax = axes[0]
ax.hist(p_a_values, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
mean_pa = np.mean(p_a_values)
ax.axvline(x=mean_pa, color="red", linestyle="--", label=f"Mean P(A) = {mean_pa:.3f}")
ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Baseline P(choose A)")
ax.set_ylabel("Count")
ax.set_title("Position bias distribution")
ax.legend()

# Baseline determinism
base_max_p = [max(r["p_a_base"], 1 - r["p_a_base"]) for r in analyzable]
ax = axes[1]
ax.hist(base_max_p, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
n_deterministic = sum(1 for p in base_max_p if p >= 0.9)
ax.set_xlabel("Baseline max(P(A), P(B))")
ax.set_ylabel("Count")
ax.set_title(f"Baseline determinism ({n_deterministic}/{len(base_max_p)} have max P ≥ 0.9)")

plt.tight_layout()
plt.savefig(ASSETS / "plot_030626_baseline_behavior.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved baseline behavior")

# --- Plot 6: Scaled vs pilot summary comparison ---
fig, ax = plt.subplots(figsize=(8, 5))

# Compare key metrics
metrics = {
    "All-layer\nflip rate": (n_flipped/len(analyzable), 0.54),
    "Peak layer\nflip rate": (max(flip_rates), 0.61),
    "P(choose A)\nbaseline": (mean_pa, 0.591),
}

x = np.arange(len(metrics))
width = 0.35
scaled_vals = [v[0] for v in metrics.values()]
pilot_vals = [v[1] for v in metrics.values()]

bars1 = ax.bar(x - width/2, scaled_vals, width, label=f"Scaled (n={len(analyzable)})", color="steelblue")
bars2 = ax.bar(x + width/2, pilot_vals, width, label="Pilot (n=90)", color="coral")

ax.set_xticks(x)
ax.set_xticklabels(metrics.keys())
ax.set_ylabel("Rate")
ax.set_ylim(0, 1)
ax.set_title("Scaled vs Pilot: Key Metrics")
ax.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(ASSETS / "plot_030626_scaled_vs_pilot_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved scaled vs pilot summary")

# Print summary stats
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Phase 1: {len(analyzable)} analyzable orderings, {n_flipped} flipped ({n_flipped/len(analyzable):.1%})")
print(f"Position bias: P(A) = {mean_pa:.3f}")
print(f"Baseline deterministic (max P ≥ 0.9): {n_deterministic}/{len(analyzable)} ({n_deterministic/len(analyzable):.0%})")
print(f"\nPhase 2: {len(phase2)} orderings, {len(layers_sorted)} layers")
print(f"Peak layer: L{layers_sorted[np.argmax(flip_rates)]} ({max(flip_rates):.1%})")
print(f"Causal window (>50%): L{min(l for l in layers_sorted if layer_flips[l]/layer_total[l] > 0.5)}-L{max(l for l in layers_sorted if layer_flips[l]/layer_total[l] > 0.5)}")
print(f"\nPilot comparison:")
print(f"  Pilot all-layer flip rate: 54%")
print(f"  Scaled all-layer flip rate: {n_flipped/len(analyzable):.0%}")
print(f"  Pilot peak: L34 (61%)")
print(f"  Scaled peak: L{layers_sorted[np.argmax(flip_rates)]} ({max(flip_rates):.0%})")
