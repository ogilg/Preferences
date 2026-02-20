"""Revised competing prompts figure for weekly report section 3.3.

Left: 2 worked examples showing the probe score flip for subject and task-type tasks.
Right: All 12 pairs dumbbell. Same green/blue colouring. Filled = loved, open = hated.

Data from git: 5487348:experiments/competing_preferences/results/final_cross_task.json
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

data = [
    {"topic": "cheese", "shell": "math",
     "st_love": -2.4, "st_hate": -286.9, "tt_hate": -207.2, "tt_love": -56.3},
    {"topic": "cats", "shell": "coding",
     "st_love": 48.7, "st_hate": -203.1, "tt_hate": -9.5, "tt_love": 140.8},
    {"topic": "cats", "shell": "math",
     "st_love": 81.2, "st_hate": -257.6, "tt_hate": -212.8, "tt_love": -68.0},
    {"topic": "cheese", "shell": "fiction",
     "st_love": -15.4, "st_hate": -284.3, "tt_hate": -271.6, "tt_love": 47.5},
    {"topic": "astronomy", "shell": "math",
     "st_love": 8.9, "st_hate": -225.5, "tt_hate": -169.2, "tt_love": -23.1},
    {"topic": "gardening", "shell": "fiction",
     "st_love": 50.9, "st_hate": -167.4, "tt_hate": -269.2, "tt_love": 63.2},
    {"topic": "gardening", "shell": "math",
     "st_love": 9.9, "st_hate": -204.7, "tt_hate": -250.4, "tt_love": -36.8},
    {"topic": "cooking", "shell": "coding",
     "st_love": 20.2, "st_hate": -189.6, "tt_hate": -35.0, "tt_love": 150.7},
    {"topic": "rainy weather", "shell": "math",
     "st_love": 95.1, "st_hate": -95.2, "tt_hate": -188.9, "tt_love": -64.5},
    {"topic": "ancient history", "shell": "coding",
     "st_love": 1.6, "st_hate": -184.9, "tt_hate": 2.7, "tt_love": 178.5},
    {"topic": "cooking", "shell": "fiction",
     "st_love": 24.3, "st_hate": -111.7, "tt_hate": -268.0, "tt_love": 93.2},
    {"topic": "classical music", "shell": "coding",
     "st_love": 0.2, "st_hate": 17.5, "tt_hate": -11.0, "tt_love": 290.0},
]

example_indices = [0, 5]  # cheese×math, gardening×fiction

SUBJ_COLOR = '#66BB6A'
TASK_COLOR = '#42A5F5'
HIGHLIGHT_COLORS = ['#FFF3E0', '#F3E5F5']

fig = plt.figure(figsize=(16, 7.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6], wspace=0.35)

# ═══════════════════════════════════
# LEFT 2 PANELS: Worked examples
# ═══════════════════════════════════
# Each example: 2 groups of bars (subject tasks, task-type tasks)
# Within each group: 2 bars (one per competing prompt)
# The flip should be visually obvious.

PROMPT_A_COLOR = '#FF8A65'  # warm orange for prompt A
PROMPT_B_COLOR = '#7986CB'  # cool blue for prompt B

for ax_i, idx in enumerate(example_indices):
    d = data[idx]
    ax = fig.add_subplot(gs[ax_i])
    ax.set_facecolor(HIGHLIGHT_COLORS[ax_i])

    topic = d["topic"]
    shell = d["shell"]

    # Two groups: subject tasks (e.g. cheese tasks), task-type tasks (e.g. math tasks)
    # Each group has two bars: score under prompt A vs prompt B
    # Prompt A = "love {topic}, hate {shell}"
    # Prompt B = "love {shell}, hate {topic}"

    groups = [f'{topic}\ntasks', f'{shell}\ntasks']
    x = np.arange(2)
    w = 0.32

    # Prompt A values: subject loved, task-type hated
    prompt_a = [d["st_love"], d["tt_hate"]]
    # Prompt B values: subject hated, task-type loved
    prompt_b = [d["st_hate"], d["tt_love"]]

    bars_a = ax.bar(x - w/2, prompt_a, w, color=PROMPT_A_COLOR, alpha=0.85,
                     edgecolor='white', linewidth=0.5, label=f'"love {topic}, hate {shell}"')
    bars_b = ax.bar(x + w/2, prompt_b, w, color=PROMPT_B_COLOR, alpha=0.85,
                     edgecolor='white', linewidth=0.5, label=f'"love {shell}, hate {topic}"')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10, fontweight='bold')
    ax.set_ylabel('Probe score shift\n(vs. no system prompt)' if ax_i == 0 else '', fontsize=10)
    ax.set_title(f'{topic} × {shell}', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(-340, 200)
    ax.grid(axis='y', alpha=0.15)
    ax.legend(fontsize=8, framealpha=0.9, loc='lower left')


# ═══════════════════════════════════
# RIGHT PANEL: All 12 pairs dumbbell
# ═══════════════════════════════════
ax_r = fig.add_subplot(gs[2])

y_all = np.arange(len(data))
pair_labels = []
offset = 0.13
ms = 8

# Highlight bands for example rows
for ex_i, idx in enumerate(example_indices):
    ax_r.axhspan(idx - 0.45, idx + 0.45, color=HIGHLIGHT_COLORS[ex_i], zorder=0)

for i, d in enumerate(data):
    topic_label = d["topic"]
    if topic_label == "classical music":
        topic_label = "class. music"
    elif topic_label == "ancient history":
        topic_label = "anc. history"
    elif topic_label == "rainy weather":
        topic_label = "rainy weath."
    pair_labels.append(f'{topic_label} × {d["shell"]}')

    # Subject dimension (green): filled = loved, open = hated
    ax_r.plot([d["st_hate"], d["st_love"]], [i - offset, i - offset],
              color=SUBJ_COLOR, linewidth=2, alpha=0.5, zorder=1)
    ax_r.plot(d["st_love"], i - offset, 'o', color=SUBJ_COLOR,
              markersize=ms, zorder=3, alpha=0.9)
    ax_r.plot(d["st_hate"], i - offset, 'o', color=SUBJ_COLOR,
              markersize=ms, zorder=3, alpha=0.9,
              markerfacecolor='white', markeredgecolor=SUBJ_COLOR, markeredgewidth=1.5)

    # Task-type dimension (blue): filled = loved, open = hated
    ax_r.plot([d["tt_hate"], d["tt_love"]], [i + offset, i + offset],
              color=TASK_COLOR, linewidth=2, alpha=0.5, zorder=1)
    ax_r.plot(d["tt_love"], i + offset, 'o', color=TASK_COLOR,
              markersize=ms, zorder=3, alpha=0.9)
    ax_r.plot(d["tt_hate"], i + offset, 'o', color=TASK_COLOR,
              markersize=ms, zorder=3, alpha=0.9,
              markerfacecolor='white', markeredgecolor=TASK_COLOR, markeredgewidth=1.5)

ax_r.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax_r.invert_yaxis()
ax_r.set_yticks(y_all)
ax_r.set_yticklabels(pair_labels, fontsize=9.5)
ax_r.set_xlabel('Probe score shift (vs. no system prompt)', fontsize=10)
ax_r.set_title('All 12 pairs', fontsize=12, fontweight='bold')

# Legend
love_subj = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=SUBJ_COLOR,
                         markersize=ms, label='Loved subject tasks')
hate_subj = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                         markeredgecolor=SUBJ_COLOR, markeredgewidth=1.5,
                         markersize=ms, label='Hated subject tasks')
love_task = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=TASK_COLOR,
                         markersize=ms, label='Loved task-type tasks')
hate_task = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                         markeredgecolor=TASK_COLOR, markeredgewidth=1.5,
                         markersize=ms, label='Hated task-type tasks')
ax_r.legend(handles=[love_subj, hate_subj, love_task, hate_task],
            loc='lower right', fontsize=8.5, framealpha=0.95)

fig.suptitle('Competing prompts: same words, flipped evaluation\n'
             'Probe tracks which dimension is loved vs. hated',
             fontsize=13, fontweight='bold', y=1.02)

out = 'experiments/probe_generalization/ood_generalization/competing_preferences/assets/plot_021626_competing_v12.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")
