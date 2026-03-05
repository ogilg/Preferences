"""Combined 4-panel persona geometry: 3 layers + utility correlations, shared axes.

Usage: python -m scripts.multi_role_ablation.plot_persona_geometry_combined
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from scipy.stats import pearsonr
from sklearn.manifold import MDS

from src.probes.data_loading import load_thurstonian_scores

ASSETS = Path("experiments/probe_generalization/multi_role_ablation/assets")
RESULTS = Path("results/experiments/mra_exp3/probes/mra_8persona_results.json")
DATE_PREFIX = "030526"

ALL_PERSONAS = [
    "noprompt", "villain", "aesthete", "midwest",
    "provocateur", "trickster", "autocrat", "sadist",
]

LABELS = {
    "noprompt": "Default",
    "villain": "Villain",
    "aesthete": "Aesthete",
    "midwest": "Midwest",
    "provocateur": "Provocateur",
    "trickster": "Trickster",
    "autocrat": "Autocrat",
    "sadist": "Sadist",
}

COLORS = {
    "noprompt": "#2c3e50",
    "villain": "#e74c3c",
    "aesthete": "#9b59b6",
    "midwest": "#2ecc71",
    "provocateur": "#e67e22",
    "trickster": "#3498db",
    "autocrat": "#1abc9c",
    "sadist": "#c0392b",
}

PERSONA_RUNS = {
    "noprompt": (Path("results/experiments/mra_exp2/pre_task_active_learning"), ""),
    "villain": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "syse8f24ac6"),
    "aesthete": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys021d8ca1"),
    "midwest": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys5d504504"),
    "provocateur": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sysf4d93514"),
    "trickster": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys09a42edc"),
    "autocrat": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys1c18219a"),
    "sadist": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys39e01d59"),
}

SPLITS = ["a", "b", "c"]


def load_cross_eval(layer_key: str) -> np.ndarray:
    with open(RESULTS) as f:
        data = json.load(f)
    ce = data["phase1"][layer_key]["cross_eval"]
    n = len(ALL_PERSONAS)
    matrix = np.zeros((n, n))
    for i, tp in enumerate(ALL_PERSONAS):
        for j, ep in enumerate(ALL_PERSONAS):
            matrix[i, j] = ce[tp][ep]["pearson_r"]
    return matrix


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2


def get_run_dir(persona: str, split: str) -> Path:
    results_dir, sys_hash = PERSONA_RUNS[persona]
    n = {"a": 1000, "b": 500, "c": 1000}[split]
    prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"
    suffix = f"mra_exp2_split_{split}_{n}_task_ids"
    dirname = f"{prefix}_{sys_hash}_{suffix}" if sys_hash else f"{prefix}_{suffix}"
    return results_dir / dirname


def load_all_utilities(persona: str) -> dict[str, float]:
    all_scores = {}
    for split in SPLITS:
        run_dir = get_run_dir(persona, split)
        scores = load_thurstonian_scores(run_dir)
        all_scores.update(scores)
    return all_scores


def compute_utility_correlation_matrix() -> np.ndarray:
    utilities = {}
    for persona in ALL_PERSONAS:
        utilities[persona] = load_all_utilities(persona)

    common_ids = set(utilities[ALL_PERSONAS[0]].keys())
    for persona in ALL_PERSONAS[1:]:
        common_ids &= set(utilities[persona].keys())
    common_ids = sorted(common_ids)
    print(f"  Utility correlations: {len(common_ids)} common tasks")

    n = len(ALL_PERSONAS)
    corr_matrix = np.zeros((n, n))
    for i, p1 in enumerate(ALL_PERSONAS):
        for j, p2 in enumerate(ALL_PERSONAS):
            u1 = np.array([utilities[p1][tid] for tid in common_ids])
            u2 = np.array([utilities[p2][tid] for tid in common_ids])
            corr_matrix[i, j], _ = pearsonr(u1, u2)
    return corr_matrix


def run_mds(dist_matrix: np.ndarray) -> np.ndarray:
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
    return mds.fit_transform(dist_matrix)


def align_to_reference(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    _, aligned, _ = procrustes(reference, target)
    ref_scale = np.sqrt(np.sum(reference**2))
    aligned_scale = np.sqrt(np.sum(aligned**2))
    if aligned_scale > 0:
        aligned = aligned * (ref_scale / aligned_scale)
    return aligned


def plot_panel(ax, coords, sim, title):
    for i in range(len(ALL_PERSONAS)):
        for j in range(i + 1, len(ALL_PERSONAS)):
            r = sim[i, j]
            if r > 0.5:
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color="gray", alpha=min(0.8, r - 0.3),
                    linewidth=r * 3, zorder=1,
                )

    for i, persona in enumerate(ALL_PERSONAS):
        ax.scatter(
            coords[i, 0], coords[i, 1],
            c=COLORS[persona], s=150, zorder=3,
            edgecolors="white", linewidth=1.5,
        )
        ax.annotate(
            LABELS[persona], (coords[i, 0], coords[i, 1]),
            textcoords="offset points", xytext=(6, 6),
            fontsize=9, fontweight="bold", color=COLORS[persona],
            zorder=4,
        )

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)

    # Compute all MDS embeddings
    layer_keys = ["L31", "L43", "L55"]
    probe_coords = {}
    probe_sims = {}
    reference_coords = None

    for layer_key in layer_keys:
        raw = load_cross_eval(layer_key)
        sim = symmetrize(raw)
        dist = 1 - sim
        np.fill_diagonal(dist, 0)
        coords = run_mds(dist)

        if reference_coords is None:
            reference_coords = coords
        else:
            coords = align_to_reference(reference_coords, coords)

        probe_coords[layer_key] = coords
        probe_sims[layer_key] = sim

    # Utility correlation MDS
    print("Computing utility correlations...")
    corr = compute_utility_correlation_matrix()
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    util_coords = run_mds(dist)
    util_coords = align_to_reference(reference_coords, util_coords)

    # Compute shared axis limits across all 4 panels
    all_coords = np.vstack([probe_coords[k] for k in layer_keys] + [util_coords])
    pad = 0.08
    x_range = all_coords[:, 0].max() - all_coords[:, 0].min()
    y_range = all_coords[:, 1].max() - all_coords[:, 1].min()
    xlim = (all_coords[:, 0].min() - pad * x_range, all_coords[:, 0].max() + pad * x_range)
    ylim = (all_coords[:, 1].min() - pad * y_range, all_coords[:, 1].max() + pad * y_range)

    # Plot 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    fig.subplots_adjust(wspace=0.08)

    for idx, layer_key in enumerate(layer_keys):
        plot_panel(axes[idx], probe_coords[layer_key], probe_sims[layer_key],
                   f"Probe transfer ({layer_key})")
        axes[idx].set_xlim(xlim)
        axes[idx].set_ylim(ylim)

    plot_panel(axes[3], util_coords, corr, "Utility correlations")
    axes[3].set_xlim(xlim)
    axes[3].set_ylim(ylim)

    # Only label left-most y-axis
    axes[0].set_ylabel("MDS dimension 2", fontsize=10)
    for ax in axes:
        ax.set_xlabel("MDS dimension 1", fontsize=10)

    fig.suptitle("Persona geometry across layers", fontsize=13, y=1.02)
    fig.tight_layout()

    path = ASSETS / f"plot_{DATE_PREFIX}_persona_geometry_combined.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
