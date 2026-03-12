"""Plot Phase 3 preference steering results."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Load data
with open("experiments/new_persona_steering/artifacts/task_set.json") as f:
    tasks = json.loads(f.read())

task_cat = {}
for cat, info in tasks["categories"].items():
    for tid in info["ids"]:
        task_cat[tid] = cat

with open("results/experiments/persona_steering_v2/preference_steering/checkpoint.jsonl") as f:
    entries = [json.loads(line) for line in f]

CATEGORIES = ["creative", "math", "knowledge_qa", "harmful", "value_conflict"]
CAT_LABELS = {"creative": "Creative", "math": "Math", "knowledge_qa": "Knowledge QA",
              "harmful": "Harmful", "value_conflict": "Value conflict"}

# Compute P(choose cat) for each condition
results = {}
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

baseline = results["baseline_coeff0"]

# Order: baseline first, then personas
COMBOS = ["baseline_coeff0", "sadist_L23_m0.2", "villain_L23_m0.2",
          "aesthete_L29_m0.2", "stem_obsessive_L29_m0.12", "lazy_L23_m0.3"]
COMBO_LABELS = ["Baseline", "Sadist", "Villain", "Aesthete", "STEM obs.", "Lazy"]

# --- Plot 1: Grouped bar chart ---
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(CATEGORIES))
width = 0.13
colors = ["#888888", "#d62728", "#e377c2", "#9467bd", "#2ca02c", "#ff7f0e"]

for i, (combo, label) in enumerate(zip(COMBOS, COMBO_LABELS)):
    vals = [results[combo][c] for c in CATEGORIES]
    offset = (i - len(COMBOS) / 2 + 0.5) * width
    bars = ax.bar(x + offset, vals, width, label=label, color=colors[i], edgecolor="white", linewidth=0.5)

ax.set_ylabel("P(choose category) %", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([CAT_LABELS[c] for c in CATEGORIES], fontsize=11)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.set_title("Preference shifts under persona steering", fontsize=13)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axhline(50, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
plt.tight_layout()
plt.savefig("experiments/new_persona_steering/assets/plot_031026_phase3_preference_bars.png", dpi=150)
plt.close()
print("Saved grouped bar chart")

# --- Plot 2: Shift heatmap (delta vs baseline) ---
fig, ax = plt.subplots(figsize=(8, 5))
personas = COMBOS[1:]  # skip baseline
persona_labels = COMBO_LABELS[1:]

data = np.array([[results[combo][c] - baseline[c] for c in CATEGORIES] for combo in personas])
im = ax.imshow(data, cmap="RdBu_r", vmin=-60, vmax=60, aspect="auto")

ax.set_xticks(range(len(CATEGORIES)))
ax.set_xticklabels([CAT_LABELS[c] for c in CATEGORIES], fontsize=11)
ax.set_yticks(range(len(persona_labels)))
ax.set_yticklabels(persona_labels, fontsize=11)

# Annotate cells
for i in range(len(personas)):
    for j in range(len(CATEGORIES)):
        val = data[i, j]
        color = "white" if abs(val) > 30 else "black"
        sign = "+" if val > 0 else ""
        ax.text(j, i, f"{sign}{val:.0f}pp", ha="center", va="center", fontsize=10, color=color, fontweight="bold")

ax.set_title("Preference shift vs baseline (pp)", fontsize=13)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Percentage point shift", fontsize=10)
plt.tight_layout()
plt.savefig("experiments/new_persona_steering/assets/plot_031026_phase3_shift_heatmap.png", dpi=150)
plt.close()
print("Saved shift heatmap")

# --- Plot 3: Hypothesis scorecard ---
hypotheses = [
    ("Sadist → harmful", "sadist_L23_m0.2", "harmful", "+"),
    ("Villain → harmful", "villain_L23_m0.2", "harmful", "+"),
    ("Aesthete → creative", "aesthete_L29_m0.2", "creative", "+"),
    ("STEM obs. → math", "stem_obsessive_L29_m0.12", "math", "+"),
    ("Lazy → −math", "lazy_L23_m0.3", "math", "-"),
]

fig, ax = plt.subplots(figsize=(8, 4))
labels = [h[0] for h in hypotheses]
shifts = [results[h[1]][h[2]] - baseline[h[2]] for h in hypotheses]
expected_signs = [h[3] for h in hypotheses]
colors_h = []
for s, exp in zip(shifts, expected_signs):
    if (exp == "+" and s > 5) or (exp == "-" and s < -5):
        colors_h.append("#2ca02c")  # confirmed
    elif abs(s) <= 5:
        colors_h.append("#888888")  # null
    else:
        colors_h.append("#d62728")  # opposite

bars = ax.barh(range(len(labels)), shifts, color=colors_h, edgecolor="white", height=0.6)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Shift vs baseline (pp)", fontsize=11)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlim(-40, 60)
ax.set_title("Hypothesis scorecard", fontsize=13)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar, shift in zip(bars, shifts):
    sign = "+" if shift > 0 else ""
    x_pos = bar.get_width() + (2 if shift >= 0 else -2)
    ha = "left" if shift >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{sign}{shift:.0f}pp", va="center", ha=ha, fontsize=10, fontweight="bold")

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#2ca02c", label="Confirmed"),
                   Patch(facecolor="#d62728", label="Opposite direction"),
                   Patch(facecolor="#888888", label="No effect")]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
ax.set_xlim(-40, 70)
plt.tight_layout()
plt.savefig("experiments/new_persona_steering/assets/plot_031026_phase3_hypothesis_scorecard.png", dpi=150)
plt.close()
print("Saved hypothesis scorecard")
