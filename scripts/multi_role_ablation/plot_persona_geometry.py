"""Visualize persona geometry from cross-eval transfer and utility correlations.

Two views of persona similarity:
1. Probe transfer: how well does persona A's probe predict persona B's preferences?
2. Utility correlation: how correlated are the raw Thurstonian utilities?

Both are projected into 2D via MDS. Layers are Procrustes-aligned to L31.

Usage: python -m scripts.multi_role_ablation.plot_persona_geometry
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
DATE_PREFIX = "030326"

ALL_PERSONAS = [
    "noprompt", "villain", "aesthete", "midwest",
    "provocateur", "trickster", "autocrat", "sadist",
]

LABELS = {
    "noprompt": "No prompt",
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
SPLIT_TASK_ID_FILES = {
    "a": Path("configs/measurement/active_learning/mra_exp2_split_a_1000_task_ids.txt"),
    "b": Path("configs/measurement/active_learning/mra_exp2_split_b_500_task_ids.txt"),
    "c": Path("configs/measurement/active_learning/mra_exp2_split_c_1000_task_ids.txt"),
}


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

    # Find common task IDs across all personas
    common_ids = set(utilities[ALL_PERSONAS[0]].keys())
    for persona in ALL_PERSONAS[1:]:
        common_ids &= set(utilities[persona].keys())
    common_ids = sorted(common_ids)
    print(f"  Utility correlations: {len(common_ids)} common tasks across all 8 personas")

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
    """Procrustes-align target coordinates to reference (translation, rotation, reflection)."""
    _, aligned, _ = procrustes(reference, target)
    # procrustes normalizes scale — rescale back to reference scale
    ref_scale = np.sqrt(np.sum(reference**2))
    aligned_scale = np.sqrt(np.sum(aligned**2))
    if aligned_scale > 0:
        aligned = aligned * (ref_scale / aligned_scale)
    return aligned


def plot_geometry(coords: np.ndarray, sim: np.ndarray, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(9, 8))

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
            c=COLORS[persona], s=200, zorder=3,
            edgecolors="white", linewidth=1.5,
        )
        ax.annotate(
            LABELS[persona], (coords[i, 0], coords[i, 1]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=11, fontweight="bold", color=COLORS[persona],
            zorder=4,
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("MDS dimension 1", fontsize=10)
    ax.set_ylabel("MDS dimension 2", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_probe_transfer_geometry():
    """MDS on probe transfer similarity, Procrustes-aligned across layers."""
    reference_coords = None
    for layer_key in ["L31", "L43", "L55"]:
        raw = load_cross_eval(layer_key)
        sim = symmetrize(raw)
        dist = 1 - sim
        np.fill_diagonal(dist, 0)
        coords = run_mds(dist)

        if reference_coords is None:
            reference_coords = coords
        else:
            coords = align_to_reference(reference_coords, coords)

        title = (f"Persona geometry from probe transfer ({layer_key})\n"
                 f"MDS on symmetrized Pearson r; edges = r > 0.5")
        path = ASSETS / f"plot_{DATE_PREFIX}_persona_geometry_{layer_key}.png"
        plot_geometry(coords, sim, title, path)


def plot_utility_geometry():
    """MDS on utility correlation matrix."""
    corr = compute_utility_correlation_matrix()
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    coords = run_mds(dist)

    title = ("Persona geometry from utility correlations\n"
             "MDS on pairwise Pearson r of Thurstonian utilities; edges = r > 0.5")
    path = ASSETS / f"plot_{DATE_PREFIX}_persona_geometry_utility.png"
    plot_geometry(coords, corr, title, path)

    # Print pairwise utility correlations
    print("\nPairwise utility correlations:")
    print(f"{'Pair':<35} {'r':>6}")
    print("-" * 42)
    pairs = []
    for i in range(len(ALL_PERSONAS)):
        for j in range(i + 1, len(ALL_PERSONAS)):
            pairs.append((ALL_PERSONAS[i], ALL_PERSONAS[j], corr[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for a, b, r in pairs:
        print(f"  {LABELS[a]:>12} ↔ {LABELS[b]:<12}  {r:.3f}")

    return corr


def plot_asymmetry(layer_key: str):
    raw = load_cross_eval(layer_key)
    n = len(ALL_PERSONAS)

    asym = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                asym[i, j] = raw[i, j] - raw[j, i]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(asym, cmap="RdBu", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([LABELS[p] for p in ALL_PERSONAS], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([LABELS[p] for p in ALL_PERSONAS], fontsize=9)
    ax.set_xlabel("Eval persona", fontsize=11)
    ax.set_ylabel("Train persona", fontsize=11)
    ax.set_title(f"Transfer asymmetry ({layer_key})\n"
                 f"Cell = r(train→eval) - r(eval→train); blue = trains better on others", fontsize=11)

    for i in range(n):
        for j in range(n):
            if i != j:
                color = "white" if abs(asym[i, j]) > 0.3 else "black"
                ax.text(j, i, f"{asym[i, j]:+.2f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="r(row→col) - r(col→row)")
    fig.tight_layout()

    path = ASSETS / f"plot_{DATE_PREFIX}_transfer_asymmetry_{layer_key}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def print_probe_similarity_ranking():
    raw = load_cross_eval("L31")
    sim = symmetrize(raw)

    pairs = []
    for i in range(len(ALL_PERSONAS)):
        for j in range(i + 1, len(ALL_PERSONAS)):
            pairs.append((ALL_PERSONAS[i], ALL_PERSONAS[j], sim[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)
    print("\nPairwise probe transfer similarity (symmetrized r, L31):")
    print(f"{'Pair':<35} {'r':>6}")
    print("-" * 42)
    for a, b, r in pairs:
        print(f"  {LABELS[a]:>12} ↔ {LABELS[b]:<12}  {r:.3f}")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)

    print("=== Probe transfer geometry ===")
    plot_probe_transfer_geometry()

    print("\n=== Utility correlation geometry ===")
    plot_utility_geometry()

    print("\n=== Transfer asymmetry ===")
    for layer_key in ["L31", "L43", "L55"]:
        plot_asymmetry(layer_key)

    print_probe_similarity_ranking()


if __name__ == "__main__":
    main()
