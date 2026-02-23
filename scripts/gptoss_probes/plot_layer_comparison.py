import matplotlib.pyplot as plt

# GPT-OSS-120B
gptoss_depths = [0.08, 0.19, 0.28, 0.39, 0.50, 0.58, 0.69, 0.78, 0.89]
gptoss_raw = [0.757, 0.769, 0.788, 0.823, 0.833, 0.832, 0.829, 0.825, 0.828]
gptoss_demeaned = [0.351, 0.386, 0.415, 0.466, 0.461, 0.457, 0.463, 0.464, 0.467]

# Gemma-3-27B
gemma_depths = [0.25, 0.52, 0.62, 0.72, 0.82, 0.92]
gemma_raw = [0.748, 0.864, 0.853, 0.849, 0.845, 0.845]
gemma_demeaned = [0.602, 0.761, 0.738, 0.728, 0.716, 0.721]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(gptoss_depths, gptoss_raw, "o-", color="tab:blue", label="GPT-OSS-120B raw")
ax.plot(gptoss_depths, gptoss_demeaned, "o--", color="tab:blue", label="GPT-OSS-120B demeaned")
ax.plot(gemma_depths, gemma_raw, "s-", color="tab:orange", label="Gemma-3-27B raw")
ax.plot(gemma_depths, gemma_demeaned, "s--", color="tab:orange", label="Gemma-3-27B demeaned")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Fractional Layer Depth")
ax.set_ylabel("Pearson r")
ax.set_title("Probe Performance: GPT-OSS-120B vs Gemma-3-27B")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    "/Users/oscargilg/Dev/MATS/Preferences-gptoss-probes/experiments/gptoss_probes/assets/plot_022226_layer_r_raw_vs_demeaned.png",
    dpi=150,
)
print("Saved plot.")
