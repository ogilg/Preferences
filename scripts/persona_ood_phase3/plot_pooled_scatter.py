"""Scatter plot: probe deltas vs behavioral deltas for persona OOD phase 3."""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Data paths ---
RESULTS_PATH = "experiments/probe_generalization/persona_ood/phase3/results.json"
CORE_TASKS_PATH = "experiments/probe_generalization/persona_ood/phase3/core_tasks.json"
V2_CONFIG_PATH = "experiments/probe_generalization/persona_ood/v2_config.json"
ENRICHED_PROMPTS_PATH = "experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json"
PROBE_PATH = "results/probes/gemma3_3k_std_demean/gemma3_3k_std_demean/probes/probe_ridge_L31.npy"
NEUTRAL_ACT_PATH = "activations/persona_ood_phase3/neutral/activations_prompt_last.npz"
PERSONA_ACT_DIR = "activations/persona_ood_phase3"
OUT_PATH = "experiments/probe_generalization/persona_ood/phase3/assets/plot_021826_pooled_scatter.png"

# --- Load data ---
with open(RESULTS_PATH) as f:
    results = json.load(f)

with open(CORE_TASKS_PATH) as f:
    task_ids = json.load(f)["task_ids"]

with open(V2_CONFIG_PATH) as f:
    v2_config = json.load(f)

with open(ENRICHED_PROMPTS_PATH) as f:
    enriched_prompts = json.load(f)

# --- Identify original (part A) and enriched personas ---
original_personas = [p["name"] for p in v2_config["personas"] if p.get("part") == "A"]
enriched_personas = list(enriched_prompts.keys())

assert len(original_personas) == 10, f"Expected 10 original personas, got {len(original_personas)}"
assert len(enriched_personas) == 10, f"Expected 10 enriched personas, got {len(enriched_personas)}"

all_personas = original_personas + enriched_personas

# --- Load probe ---
probe_arr = np.load(PROBE_PATH)
weights = probe_arr[:-1]
bias = probe_arr[-1]

# --- Load neutral activations and build task->score mapping ---
neutral_data = np.load(NEUTRAL_ACT_PATH)
neutral_task_ids = list(neutral_data["task_ids"])
neutral_acts = neutral_data["layer_31"]
neutral_scores = neutral_acts @ weights + bias
neutral_score_map = {tid: neutral_scores[i] for i, tid in enumerate(neutral_task_ids)}

# --- Baseline p_choose ---
baseline_rates = results["baseline"]["task_rates"]

# --- Compute deltas for each (persona, task) pair ---
behavioral_deltas = []
probe_deltas = []
persona_labels = []

for persona_name in all_personas:
    # Load persona activations
    act_path = f"{PERSONA_ACT_DIR}/{persona_name}/activations_prompt_last.npz"
    persona_data = np.load(act_path)
    persona_task_ids = list(persona_data["task_ids"])
    persona_acts = persona_data["layer_31"]
    persona_scores = persona_acts @ weights + bias
    persona_score_map = {tid: persona_scores[i] for i, tid in enumerate(persona_task_ids)}

    # Behavioral rates for this persona
    persona_rates = results[persona_name]["task_rates"]

    for tid in task_ids:
        # Behavioral delta
        p_persona = persona_rates[tid]["p_choose"]
        p_baseline = baseline_rates[tid]["p_choose"]
        b_delta = p_persona - p_baseline

        # Probe delta
        p_score = persona_score_map[tid]
        n_score = neutral_score_map[tid]
        pr_delta = p_score - n_score

        behavioral_deltas.append(b_delta)
        probe_deltas.append(pr_delta)
        persona_labels.append(persona_name)

behavioral_deltas = np.array(behavioral_deltas)
probe_deltas = np.array(probe_deltas)
persona_labels = np.array(persona_labels)

n_points = len(behavioral_deltas)
assert n_points == 1000, f"Expected 1000 points, got {n_points}"

# --- Color scheme: cool for original, warm for enriched ---
cool_cmap = plt.cm.tab10
warm_cmap = plt.cm.tab10

# Manually assign distinct colors
# Original (cool tones): blues, greens, purples, teals
cool_colors = [
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#3b6e8f",  # steel blue
    "#5b9a6f",  # sage
    "#7b68ee",  # medium slate blue
    "#20b2aa",  # light sea green
    "#4682b4",  # steel blue 2
    "#6a5acd",  # slate blue
]

# Enriched (warm tones): reds, oranges, yellows, pinks
warm_colors = [
    "#d62728",  # red
    "#ff7f0e",  # orange
    "#e377c2",  # pink
    "#bcbd22",  # olive/yellow
    "#cc5500",  # burnt orange
    "#ff6347",  # tomato
    "#daa520",  # goldenrod
    "#c71585",  # medium violet red
    "#ff4500",  # orange red
    "#b22222",  # firebrick
]

persona_colors = {}
for i, name in enumerate(original_personas):
    persona_colors[name] = cool_colors[i]
for i, name in enumerate(enriched_personas):
    persona_colors[name] = warm_colors[i]

# --- Regression ---
slope, intercept, r_value, p_value, std_err = stats.linregress(behavioral_deltas, probe_deltas)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

# Reference lines
ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5, zorder=1)
ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5, zorder=1)

# Scatter points by persona
for persona_name in all_personas:
    mask = persona_labels == persona_name
    label = persona_name.replace("_", " ")
    ax.scatter(
        behavioral_deltas[mask],
        probe_deltas[mask],
        c=persona_colors[persona_name],
        label=label,
        s=20,
        alpha=0.6,
        edgecolors="none",
        zorder=2,
    )

# Best-fit line
x_range = np.linspace(behavioral_deltas.min() - 0.02, behavioral_deltas.max() + 0.02, 100)
ax.plot(x_range, slope * x_range + intercept, "k--", linewidth=1.5, alpha=0.8, zorder=3)

# Annotation
ax.annotate(
    f"r = {r_value:.3f}\nn = {n_points}",
    xy=(0.04, 0.96),
    xycoords="axes fraction",
    verticalalignment="top",
    fontsize=11,
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
)

# Axis labels and title
ax.set_xlabel(r"Behavioral $\Delta$ (p_choose persona $-$ p_choose baseline)", fontsize=12)
ax.set_ylabel(r"Probe $\Delta$ (persona score $-$ neutral score)", fontsize=12)
ax.set_title("Probe deltas track persona-induced preference shifts (phase 3)", fontsize=13, fontweight="bold")

# Meaningful axis bounds anchored near 0
x_abs_max = max(abs(behavioral_deltas.min()), abs(behavioral_deltas.max()))
y_abs_max = max(abs(probe_deltas.min()), abs(probe_deltas.max()))
x_lim = np.ceil(x_abs_max * 10) / 10 + 0.05
y_lim = np.ceil(y_abs_max) + 0.5
ax.set_xlim(-x_lim, x_lim)
ax.set_ylim(-y_lim, y_lim)

# Legend with two columns
legend = ax.legend(
    loc="lower right",
    ncol=2,
    fontsize=7.5,
    markerscale=1.5,
    framealpha=0.9,
    handletextpad=0.3,
    columnspacing=0.8,
)

ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved plot to {OUT_PATH}")
print(f"r = {r_value:.4f}, p = {p_value:.2e}, slope = {slope:.4f}, n = {n_points}")
