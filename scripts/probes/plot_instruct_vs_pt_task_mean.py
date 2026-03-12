"""Compare instruct vs PT task_mean probe performance."""

import json
import matplotlib.pyplot as plt
import numpy as np

N_LAYERS = 62  # gemma-3-27b has 62 layers

# --- Load data ---

# Instruct task_mean heldout
with open("results/probes/heldout_eval_gemma3_task_mean/manifest.json") as f:
    instruct_tm = json.load(f)

# Instruct turn_boundary:-1 heldout (best instruct selector)
with open("results/probes/heldout_eval_gemma3_tb-1/manifest.json") as f:
    instruct_tb1 = json.load(f)

# PT task_mean heldout
with open("results/probes/gemma3_pt_10k_heldout_task_mean/manifest.json") as f:
    pt_tm = json.load(f)

# PT task_last heldout
with open("results/probes/gemma3_pt_10k_heldout_std_demeaned/manifest.json") as f:
    pt_tl = json.load(f)

# Instruct HOO
with open("results/probes/gemma3_10k_hoo_topic_task_mean/hoo_summary.json") as f:
    instruct_tm_hoo = json.load(f)
with open("results/probes/gemma3_10k_hoo_topic_tb-1/hoo_summary.json") as f:
    instruct_tb1_hoo = json.load(f)

# PT HOO
with open("results/probes/gemma3_pt_10k_hoo_topic_task_mean/hoo_summary.json") as f:
    pt_tm_hoo = json.load(f)
with open("results/probes/gemma3_pt_10k_hoo_topic/hoo_summary.json") as f:
    pt_tl_hoo = json.load(f)


def extract_heldout(manifest):
    return {p["layer"]: p["final_r"] for p in manifest["probes"]}


def extract_hoo_mean_r(hoo_summary):
    layers = hoo_summary["layers"]
    result = {}
    for layer in layers:
        key = f"ridge_L{layer}"
        rs = [f["layers"][key]["hoo_r"] for f in hoo_summary["folds"] if key in f.get("layers", {})]
        result[layer] = sum(rs) / len(rs) if rs else 0
    return result


def to_fractional(layer_dict):
    return {layer / N_LAYERS: r for layer, r in layer_dict.items()}


# --- Extract ---
series = {
    "IT task_mean": to_fractional(extract_heldout(instruct_tm)),
    "IT prompt_last": to_fractional(extract_heldout(instruct_tb1)),
    "PT task_mean": to_fractional(extract_heldout(pt_tm)),
    "PT task_last": to_fractional(extract_heldout(pt_tl)),
}

hoo_series = {
    "IT task_mean": to_fractional(extract_hoo_mean_r(instruct_tm_hoo)),
    "IT prompt_last": to_fractional(extract_hoo_mean_r(instruct_tb1_hoo)),
    "PT task_mean": to_fractional(extract_hoo_mean_r(pt_tm_hoo)),
    "PT task_last": to_fractional(extract_hoo_mean_r(pt_tl_hoo)),
}

colors = {
    "IT task_mean": "#2196F3",
    "IT prompt_last": "#90CAF9",
    "PT task_mean": "#F44336",
    "PT task_last": "#EF9A9A",
}
markers = {
    "IT task_mean": "o",
    "IT prompt_last": "s",
    "PT task_mean": "o",
    "PT task_last": "s",
}

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for name, data in series.items():
    xs = sorted(data.keys())
    ys = [data[x] for x in xs]
    ax1.plot(xs, ys, marker=markers[name], color=colors[name], label=name, linewidth=2, markersize=6)

ax1.set_xlabel("Fractional layer depth")
ax1.set_ylabel("Pearson r (heldout)")
ax1.set_title("Heldout evaluation")
ax1.set_ylim(0, 1.0)
ax1.set_xlim(0, 1.0)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

for name, data in hoo_series.items():
    xs = sorted(data.keys())
    ys = [data[x] for x in xs]
    ax2.plot(xs, ys, marker=markers[name], color=colors[name], label=name, linewidth=2, markersize=6)

ax2.set_xlabel("Fractional layer depth")
ax2.set_title("Cross-topic generalization (HOO)")
ax2.set_ylim(0, 1.0)
ax2.set_xlim(0, 1.0)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

fig.suptitle("Gemma-3-27B: Instruct vs Pre-trained — task_mean probe comparison", fontsize=13)
plt.tight_layout()
plt.savefig("experiments/eot_probes/turn_boundary_sweep/assets/plot_031026_instruct_vs_pt_task_mean.png", dpi=150, bbox_inches="tight")
print("Saved plot.")
