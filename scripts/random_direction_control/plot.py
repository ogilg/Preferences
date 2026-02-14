import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

ASSETS = "experiments/steering/program/open_ended_effects/random_direction_control/assets"

# Per-prompt direction asymmetry data (neg - pos) for self-referential framing
data = {
    "Probe": [0.5, 0.0, -2.5, 1.5, 0.0, 0.0, -0.5, 1.0, 2.0, 1.0],
    "R200": [-0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.5, -1.5],
    "R201": [-1.0, 0.0, 0.0, 2.0, 1.0, 2.0, -0.5, 0.0, 0.0, 1.0],
    "R202": [-1.5, 0.0, 0.5, 1.0, 0.0, 0.0, -0.5, 1.0, 0.5, -1.5],
    "R203": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -0.5, 1.0, 0.5, 2.0],
    "R204": [1.0, 1.0, 1.5, -1.0, 0.0, 1.0, 0.5, 0.0, -1.0, 2.0],
}

directions = list(data.keys())
prompt_labels = [f"INT_{i:02d}" for i in range(10)]


def plot_bar_chart():
    means = [np.mean(data[d]) for d in directions]
    sems = [np.std(data[d], ddof=1) / np.sqrt(len(data[d])) for d in directions]
    colors = ["#4878CF" if d == "Probe" else "#999999" for d in directions]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(directions))
    ax.bar(x, means, yerr=sems, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(directions)
    ax.set_ylabel("Direction Asymmetry (neg \u2212 pos)")
    ax.set_title("Self-Referential Framing: Probe vs Random Directions")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = f"{ASSETS}/plot_021426_direction_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_heatmap():
    matrix = np.array([data[d] for d in directions]).T  # rows=prompts, cols=directions

    fig, ax = plt.subplots(figsize=(7, 5.5))
    vmax = np.max(np.abs(matrix))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(directions)))
    ax.set_xticklabels(directions)
    ax.set_yticks(np.arange(len(prompt_labels)))
    ax.set_yticklabels(prompt_labels)

    for i in range(len(prompt_labels)):
        for j in range(len(directions)):
            val = matrix[i, j]
            text_color = "white" if abs(val) > vmax * 0.65 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9, color=text_color)

    ax.set_title("Per-Prompt Direction Asymmetry: Self-Referential Framing")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Direction Asymmetry")
    plt.tight_layout()
    path = f"{ASSETS}/plot_021426_per_prompt_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    plot_bar_chart()
    plot_heatmap()
