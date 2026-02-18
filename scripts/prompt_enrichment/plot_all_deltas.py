"""Plot mean absolute per-task delta for all personas across all H1 batches."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path("experiments/probe_generalization/persona_ood/prompt_enrichment")
RESULT_FILES = ["h1_results_fixed.json", "h1b_results.json", "h1c_results.json", "h1d_results.json"]
EXCLUDE = {"safety_advocate_short", "safety_advocate_rich"}
OUTPUT_PATH = RESULTS_DIR / "assets" / "plot_021726_mean_abs_delta_all_personas.png"

# Collect per-task deltas for each persona across all batches
persona_deltas: dict[str, list[float]] = {}

for fname in RESULT_FILES:
    path = RESULTS_DIR / fname
    with open(path) as f:
        data = json.load(f)

    baseline_rates = data["baseline"]["task_rates"]

    for cond_name, cond_data in data.items():
        if cond_name == "baseline" or cond_name in EXCLUDE:
            continue
        rates = cond_data["task_rates"]
        deltas = []
        for tid in baseline_rates:
            if tid in rates and baseline_rates[tid]["n_total"] > 0 and rates[tid]["n_total"] > 0:
                d = rates[tid]["p_choose"] - baseline_rates[tid]["p_choose"]
                deltas.append(d)
        persona_deltas[cond_name] = deltas

# Compute summary stats
names = []
mean_abs_deltas = []
mean_deltas = []
for name, deltas in sorted(persona_deltas.items(), key=lambda x: np.mean(np.abs(x[1])), reverse=True):
    names.append(name)
    mean_abs_deltas.append(np.mean(np.abs(deltas)))
    mean_deltas.append(np.mean(deltas))

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#d62728' if md > 0 else '#1f77b4' for md in mean_deltas]
bars = ax.barh(range(len(names)), mean_abs_deltas, color=colors, alpha=0.8)

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel("Mean |Δp_choose| per task", fontsize=12)
ax.set_title("Preference shift strength by persona\n(mean absolute per-task delta from baseline)", fontsize=13)
ax.set_xlim(0, 0.5)
ax.invert_yaxis()

# Add value labels
for i, (mad, md) in enumerate(zip(mean_abs_deltas, mean_deltas)):
    direction = "+" if md > 0 else "−" if md < 0 else ""
    ax.text(mad + 0.01, i, f"{mad:.3f} (net {direction}{abs(md):.3f})", va="center", fontsize=9)

ax.axvline(x=0, color="black", linewidth=0.5)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', alpha=0.8, label='Net positive (prefers more tasks)'),
    Patch(facecolor='#1f77b4', alpha=0.8, label='Net negative (prefers fewer tasks)'),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"Saved to {OUTPUT_PATH}")

# Also print the table
print(f"\n{'persona':<30} {'mean|Δ|':>10} {'mean Δ':>10} {'n_tasks':>8}")
for name, mad, md in zip(names, mean_abs_deltas, mean_deltas):
    n = len(persona_deltas[name])
    print(f"{name:<30} {mad:>10.3f} {md:>+10.3f} {n:>8}")
