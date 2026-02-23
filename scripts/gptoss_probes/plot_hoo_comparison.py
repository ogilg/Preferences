import matplotlib.pyplot as plt
import numpy as np

topics = [
    "coding", "content_generation", "fiction", "harmful_request",
    "knowledge_qa", "math", "model_manipulation", "other",
    "persuasive_writing", "security_legal", "sensitive_creative", "summarization",
]

gptoss_r = [0.741, 0.638, 0.749, 0.230, 0.787, 0.509, 0.569, 0.613, 0.731, 0.502, 0.284, 0.674]
gptoss_n = [164, 520, 244, 593, 930, 988, 106, 14, 120, 93, 31, 44]

gemma_r = [0.831, 0.840, 0.827, 0.890, 0.841, 0.512, 0.810, 0.880, 0.830, 0.878, 0.872, 0.791]

x = np.arange(len(topics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bars_gpt = ax.bar(x - width / 2, gptoss_r, width, label="GPT-OSS-120B (L18)", color="tab:blue")
bars_gemma = ax.bar(x + width / 2, gemma_r, width, label="Gemma-3-27B (L31)", color="tab:orange")

# Annotate sample sizes on GPT-OSS bars
for bar, n in zip(bars_gpt, gptoss_n):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"n={n}",
        ha="center", va="bottom", fontsize=7, rotation=90,
    )

ax.set_ylabel("Pearson r (HOO held-out)")
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(topics, rotation=45, ha="right")
ax.set_title("Cross-Topic Generalisation (HOO)")
ax.legend()

fig.tight_layout()
fig.savefig(
    "/Users/oscargilg/Dev/MATS/Preferences-gptoss-probes/experiments/gptoss_probes/assets/plot_022226_hoo_by_topic.png",
    dpi=150,
)
print("Saved plot.")
