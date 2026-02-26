"""Phase 5: Geometric analysis of persona vectors.

1. Cosine similarity matrix between persona vectors
2. Cosine with preference probe
3. 10k projections by topic
4. Preference probe vs persona projections scatter
5. Thurstonian mu correlation
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


PERSONAS = ["creative_artist", "evil", "lazy", "stem_nerd", "uncensored"]
VECTORS_DIR = Path("results/experiments/persona_vectors_v2")
ACTIVATIONS_10K = Path("activations/gemma_3_27b/activations_prompt_last.npz")
PROBE_DIR = Path("results/probes/gemma3_10k_heldout_std_raw")
TOPICS_FILE = Path("data/topics/topics_v2.json")
OUTPUT_DIR = Path("results/experiments/persona_vectors_v2/geometry")


def load_selected(persona: str) -> dict:
    path = VECTORS_DIR / persona / "triage" / "selected.json"
    with open(path) as f:
        return json.load(f)


def load_direction(persona: str, selector: str, layer: int) -> np.ndarray:
    return np.load(
        VECTORS_DIR / persona / "vectors" / f"{persona}_{selector}_L{layer}_direction.npy"
    )


def load_probe_direction(probe_dir: Path, probe_id: str) -> tuple[int, np.ndarray]:
    """Load probe weights and return (layer, normalized_direction)."""
    with open(probe_dir / "manifest.json") as f:
        manifest = json.load(f)

    for entry in manifest["probes"]:
        if entry["id"] == probe_id:
            weights = np.load(probe_dir / entry["file"])
            direction = weights[:-1]
            direction = direction / np.linalg.norm(direction)
            return entry["layer"], direction

    raise ValueError(f"Probe {probe_id} not found")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load selected configs
    selections = {}
    for p in PERSONAS:
        try:
            selections[p] = load_selected(p)
        except FileNotFoundError:
            print(f"No selection for {p}, skipping")
            continue

    # 1. Cosine similarity matrix
    print("1. Computing persona cosine similarity matrix...")
    persona_dirs = {}
    for p, sel in selections.items():
        persona_dirs[p] = load_direction(p, sel["selected_selector"], sel["selected_layer"])

    n = len(PERSONAS)
    cosine_matrix = np.zeros((n, n))
    for i, p1 in enumerate(PERSONAS):
        for j, p2 in enumerate(PERSONAS):
            if p1 in persona_dirs and p2 in persona_dirs:
                cosine_matrix[i, j] = np.dot(persona_dirs[p1], persona_dirs[p2])

    print("  Cosine similarity matrix:")
    print(f"  {'':>15s}", end="")
    for p in PERSONAS:
        print(f"  {p[:8]:>8s}", end="")
    print()
    for i, p1 in enumerate(PERSONAS):
        print(f"  {p1:>15s}", end="")
        for j in range(n):
            print(f"  {cosine_matrix[i,j]:>8.3f}", end="")
        print()

    np.save(OUTPUT_DIR / "persona_cosine_matrix.npy", cosine_matrix)
    with open(OUTPUT_DIR / "persona_cosine_labels.json", "w") as f:
        json.dump(PERSONAS, f)

    # 2. Cosine with preference probe
    print("\n2. Cosine with preference probes...")
    probe_manifest_path = PROBE_DIR / "manifest.json"
    if probe_manifest_path.exists():
        with open(probe_manifest_path) as f:
            manifest = json.load(f)

        probe_cosines = {}
        for p, sel in selections.items():
            layer = sel["selected_layer"]
            probe_id = f"ridge_L{layer}"

            try:
                _, probe_dir = load_probe_direction(PROBE_DIR, probe_id)
                persona_dir = persona_dirs[p]

                # Probe and persona dirs may be at different layers or dims
                if len(probe_dir) == len(persona_dir):
                    cos = float(np.dot(probe_dir, persona_dir))
                    probe_cosines[p] = {"probe_id": probe_id, "layer": layer, "cosine": cos}
                    print(f"  {p}: cos({probe_id}, persona) = {cos:.4f}")
                else:
                    print(f"  {p}: dimension mismatch (probe={len(probe_dir)}, persona={len(persona_dir)})")
                    probe_cosines[p] = {"probe_id": probe_id, "layer": layer, "cosine": None, "error": "dim_mismatch"}
            except ValueError as e:
                print(f"  {p}: {e}")
                probe_cosines[p] = {"error": str(e)}

        with open(OUTPUT_DIR / "probe_cosines.json", "w") as f:
            json.dump(probe_cosines, f, indent=2)
    else:
        print("  Probe manifest not found, skipping")

    # 3. 10k projections by topic
    print("\n3. Computing 10k projections by topic...")
    if ACTIVATIONS_10K.exists() and TOPICS_FILE.exists():
        data_10k = np.load(ACTIVATIONS_10K, allow_pickle=True)
        task_ids_10k = data_10k["task_ids"]

        with open(TOPICS_FILE) as f:
            topics_raw = json.load(f)

        # Build topic lookup — topics_v2 is dict[task_id, {model: {primary, secondary}}]
        topic_lookup = {}
        for tid, models in topics_raw.items():
            for model_name, topics in models.items():
                topic_lookup[tid] = topics.get("primary", "unknown")
                break  # take first model's classification

        # Also build origin lookup from task ID prefixes
        def get_origin(tid: str) -> str:
            prefix = str(tid).split("_")[0]
            return {"stresstest": "stress_test", "competition": "math",
                    "alpaca": "alpaca", "wildchat": "wildchat",
                    "bailbench": "bailbench"}.get(prefix, prefix)

        projections_by_persona = {}
        for p, sel in selections.items():
            layer = sel["selected_layer"]
            key = f"layer_{layer}"
            if key not in data_10k:
                print(f"  {p}: layer {layer} not in 10k activations, skipping")
                continue

            acts = data_10k[key]  # (N, d_model)
            direction = persona_dirs[p]

            # Project: dot product with unit direction
            projs = acts @ direction  # (N,)

            # Group by origin (more meaningful for persona analysis)
            topic_projs = {}
            for i, tid in enumerate(task_ids_10k):
                origin = get_origin(str(tid))
                if origin not in topic_projs:
                    topic_projs[origin] = []
                topic_projs[origin].append(float(projs[i]))

            # Summary stats
            topic_stats = {}
            for topic, vals in topic_projs.items():
                topic_stats[topic] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }

            projections_by_persona[p] = {
                "layer": layer,
                "topic_stats": topic_stats,
            }

            print(f"  {p} (L{layer}):")
            for topic in sorted(topic_stats.keys()):
                s = topic_stats[topic]
                print(f"    {topic}: mean={s['mean']:.1f} ± {s['std']:.1f} (n={s['n']})")

        with open(OUTPUT_DIR / "topic_projections.json", "w") as f:
            json.dump(projections_by_persona, f, indent=2)

        # 4. Preference probe vs persona projections
        print("\n4. Preference probe vs persona projections...")
        scatter_data = {}
        for p, sel in selections.items():
            layer = sel["selected_layer"]
            key = f"layer_{layer}"
            if key not in data_10k:
                continue

            acts = data_10k[key]
            persona_dir = persona_dirs[p]
            persona_projs = acts @ persona_dir

            probe_id = f"ridge_L{layer}"
            try:
                _, probe_dir = load_probe_direction(PROBE_DIR, probe_id)
                if len(probe_dir) == len(persona_dir):
                    probe_projs = acts @ probe_dir

                    # Save for plotting
                    topics_for_scatter = [get_origin(str(tid)) for tid in task_ids_10k]
                    scatter_data[p] = {
                        "persona_projs": persona_projs.tolist(),
                        "probe_projs": probe_projs.tolist(),
                        "topics": topics_for_scatter,
                        "layer": layer,
                    }

                    # Correlation
                    r = float(np.corrcoef(persona_projs, probe_projs)[0, 1])
                    print(f"  {p}: Pearson r(persona, probe) = {r:.4f}")
            except ValueError:
                pass

        with open(OUTPUT_DIR / "scatter_data.json", "w") as f:
            json.dump(scatter_data, f)

    else:
        print("  Missing 10k activations or topics file")

    # 5. Thurstonian mu correlation — needs mu scores which may not be available
    print("\n5. Checking for Thurstonian mu scores...")
    # Look for mu scores in the typical location
    mu_paths = [
        Path("results/measurement/gemma3_10k/thurstonian_scores.json"),
        Path("results/experiments/persona_vectors_v2/mu_scores.json"),
    ]
    mu_data = None
    for mp in mu_paths:
        if mp.exists():
            with open(mp) as f:
                mu_data = json.load(f)
            print(f"  Found mu scores at {mp}")
            break

    if mu_data is None:
        print("  No Thurstonian mu scores found — skipping mu correlation")
        print("  (This is expected if the scores aren't synced to this pod)")
    else:
        # If we have mu scores, correlate with persona projections
        mu_correlations = {}
        for p, sel in selections.items():
            layer = sel["selected_layer"]
            key = f"layer_{layer}"
            if key not in data_10k:
                continue

            acts = data_10k[key]
            direction = persona_dirs[p]
            persona_projs = acts @ direction

            # Match task IDs to mu scores
            matched_projs = []
            matched_mus = []
            for i, tid in enumerate(task_ids_10k):
                tid_str = str(tid)
                if tid_str in mu_data:
                    matched_projs.append(float(persona_projs[i]))
                    matched_mus.append(float(mu_data[tid_str]))

            if len(matched_projs) > 10:
                r = float(np.corrcoef(matched_projs, matched_mus)[0, 1])
                mu_correlations[p] = {"pearson_r": r, "n": len(matched_projs)}
                print(f"  {p}: r(persona_proj, mu) = {r:.4f} (n={len(matched_projs)})")

        with open(OUTPUT_DIR / "mu_correlations.json", "w") as f:
            json.dump(mu_correlations, f, indent=2)

    print("\nPhase 5 complete!")


if __name__ == "__main__":
    main()
