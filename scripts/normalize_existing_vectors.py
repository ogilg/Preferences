"""One-time script to normalize existing concept vectors to mean activation norm."""

import json
from pathlib import Path

import numpy as np

VECTORS_DIR = Path("concept_vectors")


def normalize_vectors_in_dir(concept_dir: Path, force: bool = False) -> None:
    """Normalize all vectors to mean activation norm and update manifest.

    After normalization, ||vector|| = mean_activation_norm at that layer.
    This way coef=1.0 adds a perturbation equal to typical activation magnitude.
    """
    manifest_path = concept_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"  Skipping {concept_dir.name}: no manifest.json")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check if already normalized (unless forcing re-normalization)
    if manifest.get("normalized") == "activation_norm" and not force:
        print(f"  Skipping {concept_dir.name}: already normalized to activation_norm")
        return

    print(f"  Processing {concept_dir.name}...")

    vector_norms: dict[str, dict[str, float]] = {}
    activation_norms: dict[str, dict[str, float]] = {}

    # First pass: compute activation norms from stored activations
    manifest_selectors = manifest.get("selectors", ["last"])
    for selector in manifest_selectors:
        activation_norms[selector] = {}
        for condition in ["positive", "negative"]:
            act_path = concept_dir / condition / f"activations_{selector}.npz"
            if act_path.exists():
                data = np.load(act_path, allow_pickle=True)
                for key in data.keys():
                    if key.startswith("layer_"):
                        layer_str = key.split("_")[1]
                        acts = data[key]
                        act_norm = float(np.linalg.norm(acts, axis=1).mean())
                        if layer_str not in activation_norms[selector]:
                            activation_norms[selector][layer_str] = act_norm
                        else:
                            # Average across conditions
                            activation_norms[selector][layer_str] = (
                                activation_norms[selector][layer_str] + act_norm
                            ) / 2

    # Handle both old format (layer -> path) and new format (selector -> {layer -> path})
    vector_files = manifest["vector_files"]
    first_val = next(iter(vector_files.values()))

    if isinstance(first_val, str):
        # Old format: {layer_str: path}
        selectors = ["last"]  # Assume last for old format
        vector_norms["last"] = {}
        for layer_str, rel_path in vector_files.items():
            vec_path = concept_dir / rel_path
            vec = np.load(vec_path)
            orig_norm = float(np.linalg.norm(vec))
            vector_norms["last"][layer_str] = orig_norm

            # Get target norm (mean activation norm)
            target_norm = activation_norms.get("last", {}).get(layer_str)
            if target_norm is None:
                print(f"    layer_{layer_str}: no activation data, keeping as-is (norm={orig_norm:.2f})")
                continue

            # Rescale to target norm
            vec_normalized = vec * (target_norm / orig_norm)
            np.save(vec_path, vec_normalized.astype(np.float32))
            print(f"    layer_{layer_str}: {orig_norm:.2f} -> {target_norm:.1f} (activation norm)")
    else:
        # New format: {selector: {layer_str: path}}
        selectors = list(vector_files.keys())
        for selector, layer_files in vector_files.items():
            vector_norms[selector] = {}

            for layer_str, rel_path in layer_files.items():
                vec_path = concept_dir / rel_path
                vec = np.load(vec_path)
                orig_norm = float(np.linalg.norm(vec))
                vector_norms[selector][layer_str] = orig_norm

                # Get target norm (mean activation norm)
                target_norm = activation_norms.get(selector, {}).get(layer_str)
                if target_norm is None:
                    print(f"    {selector}/layer_{layer_str}: no activation data, keeping as-is (norm={orig_norm:.2f})")
                    continue

                # Rescale to target norm
                vec_normalized = vec * (target_norm / orig_norm)
                np.save(vec_path, vec_normalized.astype(np.float32))
                print(f"    {selector}/layer_{layer_str}: {orig_norm:.2f} -> {target_norm:.1f}")

    # Update manifest
    manifest["normalized"] = "activation_norm"
    manifest["vector_norms"] = vector_norms
    if any(activation_norms.get(sel) for sel in activation_norms):
        manifest["activation_norms"] = activation_norms

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"    Updated manifest")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-normalize even if already done")
    args = parser.parse_args()

    print("Normalizing concept vectors to mean activation norm...\n")
    print("After this, coef=1.0 means 'add perturbation equal to typical activation magnitude'\n")

    for concept_dir in sorted(VECTORS_DIR.iterdir()):
        if not concept_dir.is_dir():
            continue
        normalize_vectors_in_dir(concept_dir, force=args.force)

    print("\nDone!")


if __name__ == "__main__":
    main()
