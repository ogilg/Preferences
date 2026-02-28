"""Per-topic mean utility bar chart for LW post.

Horizontal bar chart showing mean Thurstonian utility per topic category,
sorted by mean utility. Error bars show ±1 SE. Bar labels show (n=...).
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THURSTONIAN_CSV = Path(
    "results/experiments/gemma3_10k_run1/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/"
    "thurstonian_80fa9dc8.csv"
)
TOPICS_JSON = Path("data/topics/topics.json")
LW_ASSETS = Path("docs/lw_post/assets")

PRETTY = {
    "coding": "Coding",
    "content_generation": "Content Gen.",
    "fiction": "Fiction",
    "harmful_request": "Harmful Request",
    "knowledge_qa": "Knowledge QA",
    "math": "Math",
    "model_manipulation": "Model Manipulation",
    "other": "Other",
    "persuasive_writing": "Persuasive Writing",
    "security_legal": "Security & Legal",
    "sensitive_creative": "Sensitive Creative",
    "summarization": "Summarization",
}

# Load utilities
utilities = {}
with open(THURSTONIAN_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        utilities[row["task_id"]] = float(row["mu"])

# Load topics
with open(TOPICS_JSON) as f:
    topics_raw = json.load(f)

topics = {}
for task_id, classifiers in topics_raw.items():
    # Get the first classifier's primary topic
    first_classifier = next(iter(classifiers.values()))
    topics[task_id] = first_classifier["primary"]

# Group utilities by topic
by_topic = defaultdict(list)
for task_id, mu in utilities.items():
    if task_id in topics:
        by_topic[topics[task_id]].append(mu)

# Compute stats and sort by mean utility (drop "other" — too few tasks)
topic_stats = []
for topic, mus in by_topic.items():
    if topic == "other":
        continue
    arr = np.array(mus)
    topic_stats.append({
        "topic": topic,
        "mean": np.mean(arr),
        "se": np.std(arr) / np.sqrt(len(arr)),
        "n": len(arr),
    })

topic_stats.sort(key=lambda x: x["mean"])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

labels = [f"$\\bf{{{PRETTY[s['topic']].replace(' ', '~')}}}$ (n={s['n']})" for s in topic_stats]
means = [s["mean"] for s in topic_stats]
ses = [s["se"] for s in topic_stats]

colors = ["#E57373" if m < 0 else "#7ABAED" for m in means]

ax.barh(range(len(labels)), means, xerr=ses, color=colors,
        edgecolor="black", linewidth=0.5, capsize=3,
        error_kw={"linewidth": 0.8, "alpha": 0.5})
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Mean Thurstonian utility (μ)")
ax.set_title("Gemma-3 27B-IT: Mean Utility by Topic", fontsize=13)
ax.grid(axis="x", alpha=0.3, linestyle="--")

plt.tight_layout()
LW_ASSETS.mkdir(parents=True, exist_ok=True)
plot_path = LW_ASSETS / "plot_022626_topic_mean_utilities.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {plot_path}")
