"""Bar chart comparing probe accuracy across models, including Qwen3-Embedding-8B."""

import matplotlib.pyplot as plt
import numpy as np

models = [
    "Gemma-3 IT\n(L31)",
    "Gemma-3 PT\n(L31)",
    "Qwen3-Emb-8B\n(4096d)",
    "MiniLM\n(384d)",
]

# Pearson r: [heldout, HOO]
pearson_r = {
    "heldout": [0.864, 0.770, 0.725, 0.614],
    "hoo":     [0.817, 0.627, 0.618, 0.354],
}

# HOO std (for error bars)
hoo_std_r = [0.096, 0.128, 0.102, None]

# Pairwise accuracy: [heldout, HOO]
pw_acc = {
    "heldout": [0.768, 0.719, 0.694, 0.651],
    "hoo":     [0.702, 0.588, None, 0.548],
}

hoo_std_acc = [None, None, None, None]

# Colors: blue for Gemma IT, grey for Gemma PT, teal for Qwen, gold for MiniLM
colors_light = ["#7aaed4", "#b0b0b0", "#6dbfad", "#d4b96a"]
colors_dark  = ["#3d7aab", "#7a7a7a", "#3a9988", "#b08f3a"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Probe Accuracy: Test Set vs Leave-One-Topic-Out", fontsize=14, fontweight="bold")

x = np.arange(len(models))
width = 0.35

# --- Pearson r panel ---
ax1.set_title("Pearson r", fontsize=12)
for i in range(len(models)):
    ax1.bar(x[i] - width/2, pearson_r["heldout"][i], width,
            color=colors_light[i], edgecolor="grey", linewidth=0.5)
    hoo_err = hoo_std_r[i] if hoo_std_r[i] is not None else None
    ax1.bar(x[i] + width/2, pearson_r["hoo"][i], width,
            color=colors_dark[i], edgecolor="grey", linewidth=0.5,
            yerr=hoo_err, capsize=3, error_kw={"linewidth": 1})

    ax1.text(x[i] - width/2, pearson_r["heldout"][i] + 0.015,
             f'{pearson_r["heldout"][i]:.2f}', ha="center", va="bottom", fontsize=9)
    ax1.text(x[i] + width/2, pearson_r["hoo"][i] + (hoo_std_r[i] if hoo_std_r[i] else 0) + 0.015,
             f'{pearson_r["hoo"][i]:.2f}', ha="center", va="bottom", fontsize=9)

ax1.set_ylabel("Pearson r")
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim(0, 1.0)
ax1.set_yticks(np.arange(0, 1.1, 0.2))
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.legend(
    [plt.Rectangle((0, 0), 1, 1, fc=colors_light[0]),
     plt.Rectangle((0, 0), 1, 1, fc=colors_dark[0])],
    ["Test set (2,000 utilities)", "Leave-one-topic-out"],
    loc="upper right", fontsize=9
)

# --- Pairwise accuracy panel ---
ax2.set_title("Pairwise Accuracy", fontsize=12)
for i in range(len(models)):
    ax2.bar(x[i] - width/2, pw_acc["heldout"][i], width,
            color=colors_light[i], edgecolor="grey", linewidth=0.5)
    if pw_acc["hoo"][i] is not None:
        ax2.bar(x[i] + width/2, pw_acc["hoo"][i], width,
                color=colors_dark[i], edgecolor="grey", linewidth=0.5)
        ax2.text(x[i] + width/2, pw_acc["hoo"][i] + 0.015,
                 f'{pw_acc["hoo"][i]:.2f}', ha="center", va="bottom", fontsize=9)

    ax2.text(x[i] - width/2, pw_acc["heldout"][i] + 0.015,
             f'{pw_acc["heldout"][i]:.2f}', ha="center", va="bottom", fontsize=9)

ax2.axhline(y=0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax2.set_ylabel("Pairwise accuracy")
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylim(0, 1.0)
ax2.set_yticks(np.arange(0, 1.1, 0.2))
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.legend(
    [plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.3),
     plt.Rectangle((0, 0), 1, 1, fc=colors_light[0]),
     plt.Rectangle((0, 0), 1, 1, fc=colors_dark[0])],
    ["Chance (0.50)", "Test set (2,000 utilities)", "Leave-one-topic-out"],
    loc="upper right", fontsize=9
)

plt.tight_layout()
out = "experiments/probe_science/qwen_embedding/assets/plot_030926_cross_model_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
