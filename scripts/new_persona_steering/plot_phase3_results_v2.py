"""Plot Phase 3 preference steering results — v2 (review fixes)."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

# --- Load data ---
with open("experiments/new_persona_steering/artifacts/task_set.json") as f:
    tasks = json.loads(f.read())

task_cat = {}
for cat, info in tasks["categories"].items():
    for tid in info["ids"]:
        task_cat[tid] = cat

with open("results/experiments/persona_steering_v2/preference_steering/checkpoint.jsonl") as f:
    entries = [json.loads(line) for line in f]

CATEGORIES = ["harmful", "value_conflict", "creative", "math", "knowledge_qa"]
CAT_LABELS = {"creative": "Creative", "math": "Math", "knowledge_qa": "Knowledge QA",
              "harmful": "Harmful", "value_conflict": "Value\nconflict"}

# Consistent persona colors across all plots
PERSONA_COLORS = {
    "baseline_coeff0": "#555555",
    "sadist_L23_m0.2": "#d62728",
    "villain_L23_m0.2": "#9467bd",
    "aesthete_L29_m0.2": "#1f77b4",
    "stem_obsessive_L29_m0.12": "#2ca02c",
    "lazy_L23_m0.3": "#ff7f0e",
}
COMBOS = list(PERSONA_COLORS.keys())
COMBO_LABELS = {
    "baseline_coeff0": "Baseline (no steering)",
    "sadist_L23_m0.2": "Sadist",
    "villain_L23_m0.2": "Villain",
    "aesthete_L29_m0.2": "Aesthete",
    "stem_obsessive_L29_m0.12": "STEM obsessive",
    "lazy_L23_m0.3": "Lazy",
}
COMBO_SHORT = {
    "baseline_coeff0": "Baseline",
    "sadist_L23_m0.2": "Sadist",
    "villain_L23_m0.2": "Villain",
    "aesthete_L29_m0.2": "Aesthete",
    "stem_obsessive_L29_m0.12": "STEM obs.",
    "lazy_L23_m0.3": "Lazy",
}

# Compute P(choose cat) for each condition
results = {}
n_measurements = {}
for entry in entries:
    combo = entry["combo_key"]
    cat_chosen = {c: 0 for c in CATEGORIES}
    cat_total = {c: 0 for c in CATEGORIES}
    for m in entry["measurements"]:
        a_cat = task_cat.get(m["task_a_id"])
        b_cat = task_cat.get(m["task_b_id"])
        choice = m["choice"]
        chosen_cat = a_cat if choice == "a" else b_cat
        for cat in [a_cat, b_cat]:
            if cat:
                cat_total[cat] += 1
        if chosen_cat:
            cat_chosen[chosen_cat] += 1
    results[combo] = {c: cat_chosen[c] / cat_total[c] * 100 for c in CATEGORIES}
    n_measurements[combo] = len(entry["measurements"])

baseline = results["baseline_coeff0"]

# --- Plot 1: Shift heatmap (the hero plot) ---
fig, ax = plt.subplots(figsize=(9, 5.5))
personas = COMBOS[1:]
persona_labels = [COMBO_SHORT[c] for c in personas]
cat_labels = [CAT_LABELS[c] for c in CATEGORIES]

data = np.array([[results[combo][c] - baseline[c] for c in CATEGORIES] for combo in personas])
im = ax.imshow(data, cmap="RdBu_r", vmin=-60, vmax=60, aspect="auto")

ax.set_xticks(range(len(CATEGORIES)))
ax.set_xticklabels(cat_labels, fontsize=11)
ax.set_yticks(range(len(personas)))
ax.set_yticklabels(persona_labels, fontsize=11)

for i in range(len(personas)):
    for j in range(len(CATEGORIES)):
        val = data[i, j]
        color = "white" if abs(val) > 30 else "black"
        sign = "+" if val > 0 else ""
        ax.text(j, i, f"{sign}{val:.0f}pp", ha="center", va="center",
                fontsize=11, color=color, fontweight="bold")

ax.set_title("How does each persona vector shift task preferences?\n(percentage point change vs unsteered baseline)", fontsize=12)
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Shift from baseline (pp)", fontsize=10)
plt.tight_layout()
plt.savefig("experiments/new_persona_steering/assets/plot_031026_phase3_shift_heatmap.png", dpi=150)
plt.close()
print("Saved shift heatmap")

# --- Plot 2: Grouped bar chart with consistent colors ---
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(CATEGORIES))
width = 0.13

for i, combo in enumerate(COMBOS):
    vals = [results[combo][c] for c in CATEGORIES]
    offset = (i - len(COMBOS) / 2 + 0.5) * width
    ax.bar(x + offset, vals, width, label=COMBO_SHORT[combo],
           color=PERSONA_COLORS[combo], edgecolor="white", linewidth=0.5)

ax.set_ylabel("P(choose category)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([CAT_LABELS[c] for c in CATEGORIES], fontsize=11)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)
ax.set_title("Task category chosen when steered with each persona vector\n(baseline = unsteered Gemma 3-27B-IT, ~1800 pairwise choices per condition)",
             fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axhline(50, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
plt.tight_layout()
plt.savefig("experiments/new_persona_steering/assets/plot_031026_phase3_preference_bars.png", dpi=150)
plt.close()
print("Saved grouped bar chart")

# --- Plot 3: Hypothesis scorecard (fixed legend position) ---
hypotheses = [
    ("Sadist → harmful", "sadist_L23_m0.2", "harmful", "+"),
    ("Villain → harmful", "villain_L23_m0.2", "harmful", "+"),
    ("Lazy → −math", "lazy_L23_m0.3", "math", "-"),
    ("Aesthete → creative", "aesthete_L29_m0.2", "creative", "+"),
    ("STEM obs. → math", "stem_obsessive_L29_m0.12", "math", "+"),
]

fig, ax = plt.subplots(figsize=(8, 3.5))
labels = [h[0] for h in hypotheses]
shifts = [results[h[1]][h[2]] - baseline[h[2]] for h in hypotheses]
expected_signs = [h[3] for h in hypotheses]
bar_colors = []
for s, exp in zip(shifts, expected_signs):
    if (exp == "+" and s > 5) or (exp == "-" and s < -5):
        bar_colors.append("#2ca02c")
    elif abs(s) <= 5:
        bar_colors.append("#999999")
    else:
        bar_colors.append("#d62728")

bars = ax.barh(range(len(labels)), shifts, color=bar_colors, edgecolor="white", height=0.6)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Shift from unsteered baseline (pp)", fontsize=11)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlim(-40, 70)
ax.set_title("Pre-registered hypotheses: did steering shift the expected category?", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar, shift in zip(bars, shifts):
    sign = "+" if shift > 0 else ""
    x_pos = bar.get_width() + (2 if shift >= 0 else -2)
    ha = "left" if shift >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f"{sign}{shift:.0f}pp", va="center", ha=ha, fontsize=10, fontweight="bold")

legend_elements = [mpatches.Patch(facecolor="#2ca02c", label="Confirmed (>5pp in expected direction)"),
                   mpatches.Patch(facecolor="#d62728", label="Opposite direction"),
                   mpatches.Patch(facecolor="#999999", label="No meaningful effect (≤5pp)")]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)
plt.tight_layout()
plt.savefig("experiments/new_persona_steering/assets/plot_031026_phase3_hypothesis_scorecard.png", dpi=150)
plt.close()
print("Saved hypothesis scorecard")

# --- Plot 4: Setup diagram ---
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis("off")

box_style = dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#333333", linewidth=1.5)
arrow_style = dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                   color="#333333", linewidth=1.5)

# Phase 1: Contrastive prompts → Vector
ax.text(1.5, 3.2, "Phase 1: Extract persona vectors", fontsize=10, fontweight="bold", ha="center")
ax.annotate("", xy=(3.1, 2.0), xytext=(2.4, 2.0), arrowprops=arrow_style)
ax.text(1.2, 2.0, '  "You are a sadist\n  who delights in\n  suffering..."', fontsize=7,
        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffcccc", edgecolor="#cc0000"),
        family="monospace")
ax.text(1.2, 0.7, '  "You are kind and\n  empathetic..."', fontsize=7,
        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ccffcc", edgecolor="#00cc00"),
        family="monospace")
ax.annotate("", xy=(3.1, 1.5), xytext=(2.1, 0.7), arrowprops=arrow_style)

ax.text(3.8, 1.5, "Gemma 3-27B\nactivations\n(response avg)", fontsize=8,
        ha="center", va="center", bbox=box_style)
ax.annotate("", xy=(5.1, 1.5), xytext=(4.5, 1.5), arrowprops=arrow_style)

ax.text(5.7, 1.5, "mean(pos)\n−\nmean(neg)", fontsize=8,
        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffcc", edgecolor="#cc9900"))
ax.annotate("", xy=(6.7, 1.5), xytext=(6.3, 1.5), arrowprops=arrow_style)
ax.text(7.3, 1.5, "persona\nvector", fontsize=9, fontweight="bold",
        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="#cce5ff", edgecolor="#0066cc", linewidth=2))

# Phase 2: Filter
ax.text(8.6, 3.2, "Phase 2: Filter", fontsize=10, fontweight="bold", ha="center")
ax.annotate("", xy=(8.1, 2.0), xytext=(7.8, 1.5), arrowprops=arrow_style)
ax.text(8.6, 2.0, "Steer at 24\n(layer, coeff)\ncombos", fontsize=8,
        ha="center", va="center", bbox=box_style)
ax.text(8.6, 0.7, "Keep combos\nwith coherent +\nhigh-trait output", fontsize=8,
        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#e6ffe6", edgecolor="#009900"))
ax.annotate("", xy=(8.6, 1.0), xytext=(8.6, 1.5), arrowprops=arrow_style)

# Phase 3: Preference measurement
ax.text(10.8, 3.2, "Phase 3: Measure", fontsize=10, fontweight="bold", ha="center")
ax.annotate("", xy=(10.0, 2.0), xytext=(9.3, 0.7), arrowprops=arrow_style)
ax.text(10.8, 2.0, "Steered pairwise\npreference choices\n(90 pairs × 20)", fontsize=8,
        ha="center", va="center", bbox=box_style)
ax.text(10.8, 0.7, "P(choose harmful)\nvs baseline?", fontsize=9, fontweight="bold",
        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffe6cc", edgecolor="#cc6600"))
ax.annotate("", xy=(10.8, 1.0), xytext=(10.8, 1.5), arrowprops=arrow_style)

plt.tight_layout()
plt.savefig("experiments/new_persona_steering/assets/plot_031026_setup_diagram.png", dpi=150)
plt.close()
print("Saved setup diagram")
