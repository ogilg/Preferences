"""Replot competing preferences: paired bar chart showing probe tracks
evaluation direction for both subject and task type."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/competing_results.json") as f:
    results = json.load(f)

results.sort(key=lambda r: r["same_topic_diff"], reverse=True)

labels = []
topic_diffs = []
shell_diffs = []

for r in results:
    topic = r["topic"].replace("_", " ")
    shell = r["shell"]
    labels.append(f"{topic}\n× {shell}")
    topic_diffs.append(r["same_topic_diff"])
    # Flip sign so positive = expected direction for both
    shell_diffs.append(-r["same_shell_diff"])

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 5))
bars1 = ax.bar(x - width/2, topic_diffs, width, label="Subject tracking\n(higher score for liked subject)",
               color="#5B9BD5", edgecolor="black", alpha=0.85)
bars2 = ax.bar(x + width/2, shell_diffs, width, label="Task-type tracking\n(higher score for liked task type)",
               color="#E07B54", edgecolor="black", alpha=0.85)

ax.set_ylabel("Probe score difference\n(expected direction = positive)", fontsize=11)
ax.set_title('Competing prompts: probe tracks evaluation, not content mentions\n'
             '"love cheese, hate math" vs "love math, hate cheese" → same content, flipped evaluation',
             fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.axhline(0, color="black", linewidth=0.5)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

for i, (td, sd) in enumerate(zip(topic_diffs, shell_diffs)):
    if td < 0:
        ax.text(i - width/2, td - 15, "✗", ha="center", fontsize=10, color="red")
    if sd < 0:
        ax.text(i + width/2, sd - 15, "✗", ha="center", fontsize=10, color="red")

plt.tight_layout()
out = "docs/logs/assets/ood_generalization/plot_021126_competing_paired_bars.png"
fig.savefig(out, dpi=150)
print(f"Saved: {out}")

n_topic_correct = sum(1 for d in topic_diffs if d > 0)
n_shell_correct = sum(1 for d in shell_diffs if d > 0)
print(f"Subject tracking: {n_topic_correct}/{len(topic_diffs)} in expected direction")
print(f"Task-type tracking: {n_shell_correct}/{len(shell_diffs)} in expected direction")
print(f"Mean subject effect: {np.mean(topic_diffs):.1f}")
print(f"Mean task-type effect: {np.mean(shell_diffs):.1f}")
print(f"Ratio: {np.mean(shell_diffs)/np.mean(topic_diffs):.2f}×")
