"""Plot flip quality analysis: genuine vs label-only vs refusal by harm status."""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

EXPERIMENT_DIR = Path("experiments/patching/eot_scaled")
ASSETS = EXPERIMENT_DIR / "assets"
ASSETS.mkdir(exist_ok=True)

with open(EXPERIMENT_DIR / "flip_classification_v2.json") as f:
    class_data = json.load(f)
with open(EXPERIMENT_DIR / "flip_completions_sample_v2.json") as f:
    comp_data = json.load(f)

HARMFUL = ["bailbench", "stresstest"]


def is_harmful(task_id):
    return any(h in task_id for h in HARMFUL)


def harm_status(r):
    a_harm = is_harmful(r["task_a_id"])
    b_harm = is_harmful(r["task_b_id"])
    if a_harm and b_harm:
        return "both_harmful"
    if a_harm or b_harm:
        return "one_harmful"
    return "neither_harmful"


def flip_type(r):
    if r["patched_is_refusal"]:
        return "refusal"
    if (r["patched_executed_task"] != r["baseline_executed_task"]
            and r["patched_executed_task"] != "neither"):
        return "genuine"
    if (r["patched_stated_label"] != r["baseline_stated_label"]
            and r["patched_executed_task"] == r["baseline_executed_task"]):
        return "label_only"
    return "other"


def patch_direction(r):
    """For one-harmful pairs: is patching pushing toward or away from harm?"""
    a_harm = is_harmful(r["task_a_id"])
    b_harm = is_harmful(r["task_b_id"])
    if not (a_harm or b_harm) or (a_harm and b_harm):
        return None
    harm_is_a = a_harm
    baseline_picked_harm = ((r["baseline_chose_a"] and harm_is_a)
                            or (not r["baseline_chose_a"] and not harm_is_a))
    return "away_from_harm" if baseline_picked_harm else "toward_harm"


# Compute flip types
for r in class_data:
    r["_harm"] = harm_status(r)
    r["_flip"] = flip_type(r)
    r["_direction"] = patch_direction(r)


# --- Plot 1: Stacked bar by harm status ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

categories = ["neither_harmful", "one_harmful", "both_harmful"]
cat_labels = ["Neither harmful", "One harmful", "Both harmful"]
flip_types = ["genuine", "label_only", "refusal", "other"]
colors = {"genuine": "#2196F3", "label_only": "#FFC107", "refusal": "#F44336", "other": "#9E9E9E"}
flip_labels = {"genuine": "Genuine content flip", "label_only": "Label-only flip",
               "refusal": "Refusal", "other": "Other"}

ax = axes[0]
x = np.arange(len(categories))
width = 0.6
bottom = np.zeros(len(categories))

for ft in flip_types:
    vals = []
    for cat in categories:
        in_cat = [r for r in class_data if r["_harm"] == cat]
        if in_cat:
            vals.append(sum(1 for r in in_cat if r["_flip"] == ft) / len(in_cat))
        else:
            vals.append(0)
    bars = ax.bar(x, vals, width, bottom=bottom, color=colors[ft], label=flip_labels[ft])
    # Add count labels for non-zero
    for i, v in enumerate(vals):
        if v > 0.05:
            in_cat = [r for r in class_data if r["_harm"] == categories[i]]
            n = sum(1 for r in in_cat if r["_flip"] == ft)
            ax.text(x[i], bottom[i] + v/2, str(n), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white" if ft != "label_only" else "black")
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(cat_labels)
for i, cat in enumerate(categories):
    n = sum(1 for r in class_data if r["_harm"] == cat)
    ax.text(x[i], -0.06, f"n={n}", ha="center", va="top", fontsize=9, color="gray")
ax.set_ylabel("Proportion")
ax.set_ylim(0, 1.15)
ax.set_title("Flip type by harm status")
ax.legend(loc="upper right", fontsize=8)

# --- Plot 2: Toward vs away from harm ---
ax = axes[1]
directions = ["toward_harm", "away_from_harm"]
dir_labels = ["Toward harm", "Away from harm"]
x = np.arange(len(directions))
bottom = np.zeros(len(directions))

for ft in flip_types:
    vals = []
    for d in directions:
        in_d = [r for r in class_data if r["_direction"] == d]
        if in_d:
            vals.append(sum(1 for r in in_d if r["_flip"] == ft) / len(in_d))
        else:
            vals.append(0)
    bars = ax.bar(x, vals, width, bottom=bottom, color=colors[ft], label=flip_labels[ft])
    for i, v in enumerate(vals):
        if v > 0.05:
            in_d = [r for r in class_data if r["_direction"] == directions[i]]
            n = sum(1 for r in in_d if r["_flip"] == ft)
            ax.text(x[i], bottom[i] + v/2, str(n), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white" if ft != "label_only" else "black")
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(dir_labels)
for i, d in enumerate(directions):
    n = sum(1 for r in class_data if r["_direction"] == d)
    ax.text(x[i], -0.06, f"n={n}", ha="center", va="top", fontsize=9, color="gray")
ax.set_ylabel("Proportion")
ax.set_ylim(0, 1.15)
ax.set_title("Flip type: toward vs away from harmful task")
ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(ASSETS / "plot_030626_flip_type_by_harm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved flip type by harm status")


# --- Plot 3: Overall flip quality (v2, all 200) ---
fig, ax = plt.subplots(figsize=(6, 5))

counts = defaultdict(int)
for r in class_data:
    counts[r["_flip"]] += 1
total = len(class_data)

wedge_colors = [colors[ft] for ft in flip_types if counts[ft] > 0]
wedge_labels = [f"{flip_labels[ft]}\n{counts[ft]} ({counts[ft]/total:.0%})"
                for ft in flip_types if counts[ft] > 0]
wedge_vals = [counts[ft] for ft in flip_types if counts[ft] > 0]

wedges, texts = ax.pie(wedge_vals, labels=wedge_labels, colors=wedge_colors,
                       startangle=90, textprops={"fontsize": 10})
ax.set_title(f"Flip quality breakdown (n={total})")

plt.tight_layout()
plt.savefig(ASSETS / "plot_030626_flip_quality_pie.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved flip quality pie")


# --- Plot 4: Stated vs executed dissociation matrix ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (title, subset) in enumerate([
    ("Baseline", [(r["baseline_stated_label"], r["baseline_executed_task"]) for r in class_data]),
    ("Patched", [(r["patched_stated_label"], r["patched_executed_task"]) for r in class_data]),
]):
    ax = axes[idx]
    labels_stated = ["a", "b", "unclear"]
    labels_exec = ["a", "b", "neither"]
    matrix = np.zeros((len(labels_exec), len(labels_stated)))
    for stated, executed in subset:
        si = labels_stated.index(stated) if stated in labels_stated else 2
        ei = labels_exec.index(executed) if executed in labels_exec else 2
        matrix[ei, si] += 1

    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels_stated)))
    ax.set_xticklabels([f"Stated {l.upper()}" for l in labels_stated])
    ax.set_yticks(range(len(labels_exec)))
    ax.set_yticklabels([f"Executed {l.upper()}" if l != "neither" else "Executed neither"
                        for l in labels_exec])
    ax.set_title(title)

    for i in range(len(labels_exec)):
        for j in range(len(labels_stated)):
            val = int(matrix[i, j])
            if val > 0:
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=12,
                        fontweight="bold", color=color)

plt.suptitle("Stated label vs executed task content", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(ASSETS / "plot_030626_stated_vs_executed.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved stated vs executed matrix")

print("\nDone!")
