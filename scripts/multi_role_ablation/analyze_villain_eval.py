"""Compare villain (with system prompt) vs no_prompt on the 500-task eval set."""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.measurement.storage.loading import load_run_utilities

NOPROMPT_RUN = Path("results/experiments/mra_persona1_noprompt/mra_persona1_noprompt/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
VILLAIN_RUN = Path("results/experiments/mra_villain_eval/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
EVAL_IDS_PATH = "experiments/probe_generalization/multi_role_ablation/task_ids_eval.txt"
TOPICS_PATH = "data/topics/topics.json"
ASSETS_DIR = Path("experiments/probe_generalization/multi_role_ablation/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Load eval task IDs
with open(EVAL_IDS_PATH) as f:
    eval_ids = set(line.strip() for line in f if line.strip())

# Load utilities
mu_np, ids_np = load_run_utilities(NOPROMPT_RUN)
mu_v, ids_v = load_run_utilities(VILLAIN_RUN)

noprompt = pd.Series(mu_np, index=ids_np, name="no_prompt")
villain = pd.Series(mu_v, index=ids_v, name="villain")

df = pd.DataFrame({"no_prompt": noprompt, "villain": villain}).dropna()
df = df[df.index.isin(eval_ids)]

# Load topics
with open(TOPICS_PATH) as f:
    topics_raw = json.load(f)

topic_map = {}
for tid, models in topics_raw.items():
    for model_name, cats in models.items():
        topic_map[tid] = cats["primary"]

df["topic"] = df.index.map(lambda x: topic_map.get(x, "unknown"))

# Sort topics by delta for consistent ordering
persona_cols = ["no_prompt", "villain"]
topic_means = df.groupby("topic")[persona_cols].mean()
topic_counts = df.groupby("topic").size().rename("n")
topic_means["delta"] = topic_means["villain"] - topic_means["no_prompt"]
topic_order = topic_means.sort_values("delta", ascending=True).index.tolist()

# --- Plot 1: Scatter of utilities ---
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(df["no_prompt"], df["villain"], alpha=0.3, s=12, color="steelblue")
lims = [min(df[persona_cols].min().min(), -10), max(df[persona_cols].max().max(), 10)]
ax.plot(lims, lims, "--", color="gray", linewidth=0.8)
ax.set_xlabel("No prompt (baseline) utility")
ax.set_ylabel("Villain utility")
r = df["no_prompt"].corr(df["villain"])
ax.set_title(f"Utility scatter (eval set, n={len(df)}, r={r:.3f})")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(ASSETS_DIR / "plot_022525_villain_vs_noprompt_scatter.png", dpi=150)
plt.close()

# --- Plot 2: Per-topic mean delta (horizontal bar) ---
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#c0392c" if d > 0 else "#2c7fb8" for d in topic_means.loc[topic_order, "delta"]]
bars = ax.barh(
    range(len(topic_order)),
    topic_means.loc[topic_order, "delta"],
    color=colors, edgecolor="white", linewidth=0.5,
)
ax.set_yticks(range(len(topic_order)))
labels = [f"{t}  (n={topic_counts[t]})" for t in topic_order]
ax.set_yticklabels(labels, fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Mean utility delta (villain − no_prompt)")
ax.set_title("Per-topic preference shift under villain persona")
fig.tight_layout()
fig.savefig(ASSETS_DIR / "plot_022525_villain_topic_delta.png", dpi=150)
plt.close()

# --- Plot 3: Within-topic std comparison ---
topic_std_np = df.groupby("topic")["no_prompt"].std()
topic_std_v = df.groupby("topic")["villain"].std()
# Only topics with enough tasks
valid_topics = [t for t in topic_order if topic_counts[t] >= 10]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(valid_topics))
width = 0.35
ax.barh(x - width/2, topic_std_np[valid_topics], width, label="No prompt", color="#2c7fb8", alpha=0.8)
ax.barh(x + width/2, topic_std_v[valid_topics], width, label="Villain", color="#c0392c", alpha=0.8)
ax.set_yticks(x)
labels = [f"{t}  (n={topic_counts[t]})" for t in valid_topics]
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Within-topic std of utilities")
ax.set_title("Within-topic utility spread: no_prompt vs villain")
ax.legend(loc="lower right")
ax.set_xlim(0)
fig.tight_layout()
fig.savefig(ASSETS_DIR / "plot_022525_villain_within_topic_std.png", dpi=150)
plt.close()

# --- Plot 4: Within-topic correlations ---
within_r = {}
for topic in valid_topics:
    subset = df[df["topic"] == topic]
    within_r[topic] = subset["no_prompt"].corr(subset["villain"])

fig, ax = plt.subplots(figsize=(8, 5))
colors_r = ["#c0392c" if r < 0 else "#2c7fb8" for r in [within_r[t] for t in valid_topics]]
ax.barh(range(len(valid_topics)), [within_r[t] for t in valid_topics], color=colors_r, alpha=0.8)
ax.set_yticks(range(len(valid_topics)))
labels = [f"{t}  (n={topic_counts[t]})" for t in valid_topics]
ax.set_yticklabels(labels, fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson r (no_prompt vs villain within topic)")
ax.set_title("Within-topic correlation of utilities")
ax.set_xlim(-0.5, 0.5)
fig.tight_layout()
fig.savefig(ASSETS_DIR / "plot_022525_villain_within_topic_corr.png", dpi=150)
plt.close()

# --- Print summary stats for the report ---
print("=== Summary stats ===")
print(f"Eval tasks: {len(df)}")
print(f"Overall r: {r:.3f}")
print(f"Overall R²: {r**2:.3f}")
print()
print("=== Per-topic table ===")
summary = topic_means.join(topic_counts).sort_values("delta", ascending=False)
summary["std_noprompt"] = topic_std_np
summary["std_villain"] = topic_std_v
for t in valid_topics:
    summary.loc[t, "within_r"] = within_r[t]
print(summary.round(3).to_string())
