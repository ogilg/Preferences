"""Phase 5: Geometric analysis of persona vectors vs preference probes."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

PERSONAS = ["evil", "stem_nerd", "creative_artist", "uncensored", "lazy"]
PERSONA_LABELS = ["Evil", "STEM Nerd", "Creative Artist", "Uncensored", "Lazy"]
BASE = Path("results/experiments/persona_vectors")
PROBE_DIR = Path("results/probes/gemma3_10k_heldout_std_raw")
ACTIVATIONS_10K = Path("activations/gemma_3_27b/activations_prompt_last.npz")
ASSETS = Path("experiments/persona_vectors/assets")
GEOMETRY_DIR = Path("results/experiments/persona_vectors/geometry")


def load_best_layer_and_direction(persona: str) -> tuple[int, np.ndarray]:
    vec_dir = BASE / persona / "vectors"
    with open(vec_dir / "layer_selection.json") as f:
        info = json.load(f)
    layer = info["best_layer"]
    vec = np.load(vec_dir / f"{persona}_L{layer}.npy")
    return layer, vec[:-1]  # strip intercept


def load_persona_direction_at_layer(persona: str, layer: int) -> np.ndarray:
    vec_dir = BASE / persona / "vectors"
    vec = np.load(vec_dir / f"{persona}_L{layer}.npy")
    return vec[:-1]


def load_probe_direction(layer: int) -> np.ndarray:
    probe_file = PROBE_DIR / "probes" / f"probe_ridge_L{layer}.npy"
    if not probe_file.exists():
        return None
    vec = np.load(probe_file)
    direction = vec[:-1]  # strip intercept
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return None
    return direction / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def analysis_1_persona_similarity_matrix():
    """Cosine similarity between all persona vectors at a common layer."""
    print("\n=== Analysis 1: Persona vector cosine similarity ===")

    # Use layer 43 as common layer (available for all personas and probes)
    common_layer = 43
    directions = {}
    for persona in PERSONAS:
        directions[persona] = load_persona_direction_at_layer(persona, common_layer)

    n = len(PERSONAS)
    sim_matrix = np.zeros((n, n))
    for i, p1 in enumerate(PERSONAS):
        for j, p2 in enumerate(PERSONAS):
            sim_matrix[i, j] = cosine_similarity(directions[p1], directions[p2])

    print(f"Common layer: {common_layer}")
    print("Cosine similarity matrix:")
    header = "             " + "  ".join(f"{l:>8s}" for l in PERSONA_LABELS)
    print(header)
    for i, label in enumerate(PERSONA_LABELS):
        row = f"{label:>13s}" + "  ".join(f"{sim_matrix[i,j]:+8.3f}" for j in range(n))
        print(row)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(PERSONA_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(PERSONA_LABELS)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(sim_matrix[i,j]) > 0.5 else "black", fontsize=10)
    plt.colorbar(im, label="Cosine Similarity")
    ax.set_title(f"Persona Vector Cosine Similarity (Layer {common_layer})")
    plt.tight_layout()
    plt.savefig(ASSETS / "plot_022526_persona_cosine_similarity.png", dpi=150)
    plt.close()
    print(f"  Saved to {ASSETS / 'plot_022526_persona_cosine_similarity.png'}")

    return sim_matrix


def analysis_2_cosine_with_preference_probe():
    """Cosine similarity between each persona vector and the preference probe."""
    print("\n=== Analysis 2: Persona vs preference probe cosine similarity ===")

    results = {}
    # For each persona, compare at its best layer and at common layers
    for persona in PERSONAS:
        best_layer, direction = load_best_layer_and_direction(persona)
        probe_dir = load_probe_direction(best_layer)
        if probe_dir is not None:
            cos = cosine_similarity(direction, probe_dir)
            results[persona] = {"best_layer": best_layer, "cosine": cos}
            print(f"  {persona} (L{best_layer}): cos = {cos:+.4f}")
        else:
            print(f"  {persona} (L{best_layer}): no preference probe at this layer")
            # Try common probe layers
            for try_layer in [31, 37, 43, 49, 55]:
                persona_dir = load_persona_direction_at_layer(persona, try_layer)
                probe_dir = load_probe_direction(try_layer)
                if probe_dir is not None:
                    cos = cosine_similarity(persona_dir, probe_dir)
                    results[persona] = {"layer": try_layer, "cosine": cos}
                    print(f"    fallback L{try_layer}: cos = {cos:+.4f}")
                    break

    # Also compare at all probe layers for each persona
    print("\n  Full layer comparison:")
    probe_layers = [15, 31, 37, 43, 49, 55]
    for persona in PERSONAS:
        cosines = []
        for layer in probe_layers:
            persona_dir = load_persona_direction_at_layer(persona, layer)
            probe_dir = load_probe_direction(layer)
            if probe_dir is not None:
                cos = cosine_similarity(persona_dir, probe_dir)
                cosines.append((layer, cos))
        row = "  ".join(f"L{l}:{c:+.3f}" for l, c in cosines)
        print(f"    {persona:>16s}: {row}")

    return results


def analysis_3_projection_of_10k_activations():
    """Project 10k task activations onto persona vectors, correlate with scores."""
    print("\n=== Analysis 3: 10k activation projections ===")

    # Load 10k activations
    data_10k = np.load(ACTIVATIONS_10K, allow_pickle=True)
    task_ids_10k = data_10k["task_ids"]

    # Load Thurstonian scores
    # Try to find scores in measurement results
    scores_path = None
    for candidate in [
        Path("results/measurement/gemma3_10k/thurstonian_scores.json"),
        Path("results/measurement/gemma3_10k_heldout/thurstonian_scores.json"),
    ]:
        if candidate.exists():
            scores_path = candidate
            break

    # Also check manifest for scores path
    manifest_path = PROBE_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Try to load scores from run_dir
    run_dir = Path(manifest.get("run_dir", ""))
    scores_file = run_dir / "thurstonian_scores.json" if run_dir.name else None

    if scores_file and scores_file.exists():
        scores_path = scores_file

    if scores_path is None:
        # Search more broadly
        from glob import glob
        candidates = glob("results/**/thurstonian_scores.json", recursive=True)
        if candidates:
            scores_path = Path(candidates[0])

    if scores_path is None:
        print("  WARNING: Could not find Thurstonian scores. Skipping score correlation.")
        scores_map = None
    else:
        print(f"  Loading scores from {scores_path}")
        with open(scores_path) as f:
            scores_data = json.load(f)
        # Scores should be {task_id: mu_score} or similar
        if isinstance(scores_data, dict):
            scores_map = scores_data
        elif isinstance(scores_data, list):
            scores_map = {item["task_id"]: item["mu"] for item in scores_data}
        else:
            scores_map = None

    # Load task origins for coloring
    origins_path = None
    for candidate in [
        Path("results/measurement/gemma3_10k/task_origins.json"),
    ]:
        if candidate.exists():
            origins_path = candidate
            break

    # Project activations onto each persona vector
    common_layer = 43
    acts_10k = data_10k[f"layer_{common_layer}"]
    print(f"  Activations shape at L{common_layer}: {acts_10k.shape}")

    projections = {}
    for persona in PERSONAS:
        direction = load_persona_direction_at_layer(persona, common_layer)
        proj = acts_10k @ direction
        projections[persona] = proj
        print(f"  {persona}: proj mean={np.mean(proj):.3f}, std={np.std(proj):.3f}")

    # Save projection data
    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        GEOMETRY_DIR / "projections_L43.npz",
        task_ids=task_ids_10k,
        **{f"proj_{p}": projections[p] for p in PERSONAS},
    )

    # Correlations between persona projections
    print("\n  Correlations between persona projections:")
    for i, p1 in enumerate(PERSONAS):
        for j, p2 in enumerate(PERSONAS):
            if j > i:
                r = np.corrcoef(projections[p1], projections[p2])[0, 1]
                print(f"    {p1} vs {p2}: r = {r:.3f}")

    return projections, task_ids_10k


def analysis_4_pca_visualization(projections, task_ids_10k):
    """PCA visualization: preference probe vs persona vectors."""
    print("\n=== Analysis 4: PCA visualizations ===")

    common_layer = 43
    data_10k = np.load(ACTIVATIONS_10K, allow_pickle=True)
    acts_10k = data_10k[f"layer_{common_layer}"]

    # Load preference probe direction
    pref_dir = load_probe_direction(common_layer)
    if pref_dir is None:
        print("  No preference probe at L43, skipping PCA.")
        return

    pref_proj = acts_10k @ pref_dir

    # For each persona, plot 2D scatter: pref probe projection vs persona projection
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for idx, (persona, label) in enumerate(zip(PERSONAS, PERSONA_LABELS)):
        ax = axes[idx]
        persona_proj = projections[persona]

        ax.scatter(pref_proj, persona_proj, alpha=0.05, s=1, color="steelblue")
        ax.set_xlabel("Preference probe projection")
        ax.set_ylabel(f"{label} projection")
        ax.set_title(label)

        r = np.corrcoef(pref_proj, persona_proj)[0, 1]
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat"))

    plt.suptitle(f"Preference Probe vs Persona Projections (Layer {common_layer})", fontsize=14)
    plt.tight_layout()
    plt.savefig(ASSETS / "plot_022526_pref_vs_persona_projections.png", dpi=150)
    plt.close()
    print(f"  Saved to {ASSETS / 'plot_022526_pref_vs_persona_projections.png'}")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)

    sim_matrix = analysis_1_persona_similarity_matrix()
    probe_cosines = analysis_2_cosine_with_preference_probe()
    projections, task_ids = analysis_3_projection_of_10k_activations()
    analysis_4_pca_visualization(projections, task_ids)

    # Save all geometry results
    geometry_results = {
        "persona_similarity_matrix": sim_matrix.tolist(),
        "persona_labels": PERSONA_LABELS,
        "probe_cosines": probe_cosines,
    }
    with open(GEOMETRY_DIR / "geometry_results.json", "w") as f:
        json.dump(geometry_results, f, indent=2)

    print("\nAll geometry analysis done!")


if __name__ == "__main__":
    main()
