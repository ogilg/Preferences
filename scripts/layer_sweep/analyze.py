import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DATA_PATH = Path("experiments/steering_program/layer_sweep/judged_results.json")
ASSETS_DIR = Path("experiments/steering_program/layer_sweep/assets")

LAYERS = [37, 43, 49, 55]
PROBE_TYPES = ["ridge", "bt"]
COEF_LABELS = ["-6%", "-2%", "0%", "+2%", "+6%"]
CATEGORIES = ["D_valence", "F_affect"]


def load_results():
    with open(DATA_PATH) as f:
        data = json.load(f)
    return data["results"]


def plot_valence_by_layer_probe(results: list[dict]):
    """2x4 grid: rows=probe_type, cols=layer. Mean valence vs coef_label with SEM and Spearman."""
    valid = [r for r in results if isinstance(r["valence"], float)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    fig.suptitle("Valence vs Perturbation Fraction by Layer and Probe Type", fontsize=14, y=0.98)

    for row_idx, probe in enumerate(PROBE_TYPES):
        for col_idx, layer in enumerate(LAYERS):
            ax = axes[row_idx, col_idx]
            subset = [r for r in valid if r["probe_type"] == probe and r["layer"] == layer]

            means = []
            sems = []
            coefs_for_corr = []
            valences_for_corr = []

            for label in COEF_LABELS:
                vals = [r["valence"] for r in subset if r["coef_label"] == label]
                if vals:
                    means.append(np.mean(vals))
                    sems.append(stats.sem(vals))
                else:
                    means.append(np.nan)
                    sems.append(0.0)

                for r in subset:
                    if r["coef_label"] == label:
                        coefs_for_corr.append(r["coefficient"])
                        valences_for_corr.append(r["valence"])

            ax.errorbar(
                range(len(COEF_LABELS)), means, yerr=sems,
                fmt="o-", capsize=3, color="steelblue", markersize=5
            )
            ax.set_xticks(range(len(COEF_LABELS)))
            ax.set_xticklabels(COEF_LABELS, fontsize=8)
            ax.set_ylim(-0.5, 1.0)

            if row_idx == 0:
                ax.set_title(f"Layer {layer}", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"{probe}\nMean Valence", fontsize=10)
            if row_idx == 1:
                ax.set_xlabel("Perturbation", fontsize=9)

            if len(coefs_for_corr) >= 3:
                rho, p = stats.spearmanr(coefs_for_corr, valences_for_corr)
                p_str = f"{p:.1e}" if p < 0.001 else f"{p:.3f}"
                ax.text(
                    0.05, 0.95, f"rho={rho:.3f}\np={p_str}",
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
                )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ASSETS_DIR / "plot_021326_valence_by_layer_probe.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_coherence_by_layer(results: list[dict]):
    """1x4 grid: cols=layer. Mean coherence vs coef_label for ridge (blue) and bt (orange)."""
    valid = [r for r in results if isinstance(r["coherence"], int)]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle("Coherence vs Perturbation Fraction by Layer", fontsize=14, y=1.02)

    colors = {"ridge": "steelblue", "bt": "darkorange"}
    offsets = {"ridge": -0.1, "bt": 0.1}

    for col_idx, layer in enumerate(LAYERS):
        ax = axes[col_idx]
        subset = [r for r in valid if r["layer"] == layer]

        for probe in PROBE_TYPES:
            probe_subset = [r for r in subset if r["probe_type"] == probe]
            means = []
            sems = []

            for label in COEF_LABELS:
                vals = [r["coherence"] for r in probe_subset if r["coef_label"] == label]
                if vals:
                    means.append(np.mean(vals))
                    sems.append(stats.sem(vals))
                else:
                    means.append(np.nan)
                    sems.append(0.0)

            x = np.arange(len(COEF_LABELS)) + offsets[probe]
            ax.errorbar(
                x, means, yerr=sems,
                fmt="o-", capsize=3, color=colors[probe],
                markersize=5, label=probe
            )

        ax.set_xticks(range(len(COEF_LABELS)))
        ax.set_xticklabels(COEF_LABELS, fontsize=8)
        ax.set_ylim(3.0, 5.0)
        ax.set_title(f"Layer {layer}", fontsize=11)
        ax.set_xlabel("Perturbation", fontsize=9)
        if col_idx == 0:
            ax.set_ylabel("Mean Coherence", fontsize=10)
            ax.legend(fontsize=9)

    fig.tight_layout()
    out = ASSETS_DIR / "plot_021326_coherence_by_layer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_heatmap_rho(results: list[dict]):
    """Heatmap: rows=layers, cols=category x probe_type combos. Cell values = Spearman rho."""
    valid = [r for r in results if isinstance(r["valence"], float)]

    col_labels = ["D_ridge", "D_bt", "F_ridge", "F_bt", "All_ridge", "All_bt"]
    rho_matrix = np.full((len(LAYERS), len(col_labels)), np.nan)

    for row_idx, layer in enumerate(LAYERS):
        layer_data = [r for r in valid if r["layer"] == layer]

        for col_idx, col_label in enumerate(col_labels):
            parts = col_label.split("_")
            if parts[0] == "All":
                probe = parts[1]
                subset = [r for r in layer_data if r["probe_type"] == probe]
            elif parts[0] == "D":
                probe = parts[1]
                subset = [r for r in layer_data if r["probe_type"] == probe and r["category"] == "D_valence"]
            else:
                probe = parts[1]
                subset = [r for r in layer_data if r["probe_type"] == probe and r["category"] == "F_affect"]

            if len(subset) >= 3:
                coefs = [r["coefficient"] for r in subset]
                vals = [r["valence"] for r in subset]
                rho, _ = stats.spearmanr(coefs, vals)
                rho_matrix[row_idx, col_idx] = rho

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Spearman rho: Valence vs Coefficient", fontsize=14)

    im = ax.imshow(
        rho_matrix, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto"
    )

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(LAYERS)))
    ax.set_yticklabels([str(l) for l in LAYERS], fontsize=10)
    ax.set_ylabel("Layer", fontsize=11)

    for i in range(len(LAYERS)):
        for j in range(len(col_labels)):
            val = rho_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.2 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Spearman rho", shrink=0.8)
    fig.tight_layout()
    out = ASSETS_DIR / "plot_021326_heatmap_rho.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    results = load_results()
    print(f"Loaded {len(results)} results")

    n_valid_valence = sum(1 for r in results if isinstance(r["valence"], float))
    n_valid_coherence = sum(1 for r in results if isinstance(r["coherence"], int))
    print(f"Valid valence: {n_valid_valence}, valid coherence: {n_valid_coherence}")

    plot_valence_by_layer_probe(results)
    plot_coherence_by_layer(results)
    plot_heatmap_rho(results)
    print("Done.")
