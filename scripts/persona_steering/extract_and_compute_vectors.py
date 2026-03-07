"""Phase 1-2: Extract activations and compute persona vectors for persona_steering experiment."""

import json
from pathlib import Path

import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel

ARTIFACTS_DIR = Path("experiments/persona_vectors/persona_steering/artifacts")
OUTPUT_BASE = Path("results/experiments/persona_steering")
PERSONAS = ["sadist", "villain", "predator", "aesthete", "stem_obsessive"]
LAYERS = [23, 31, 37, 43]
BATCH_SIZE = 16


def load_persona(name: str) -> dict:
    with open(ARTIFACTS_DIR / f"{name}.json") as f:
        return json.load(f)


def compute_cohens_d(pos: np.ndarray, neg: np.ndarray, direction: np.ndarray) -> float:
    proj_pos = pos @ direction
    proj_neg = neg @ direction
    pooled_std = np.sqrt((np.var(proj_pos, ddof=1) + np.var(proj_neg, ddof=1)) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(proj_pos) - np.mean(proj_neg)) / pooled_std)


def extract_for_persona(model: HuggingFaceModel, persona_name: str):
    persona_data = load_persona(persona_name)
    questions = persona_data["eval_questions"]
    print(f"\n{'='*60}")
    print(f"Persona: {persona_name} ({len(questions)} questions)")

    for condition, prompt_key in [("pos", "positive"), ("neg", "negative")]:
        system_prompt = persona_data[prompt_key]
        save_dir = OUTPUT_BASE / persona_name / "activations" / condition
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "activations_prompt_last.npz"

        if save_path.exists():
            print(f"  {condition}: already extracted, skipping")
            continue

        print(f"  {condition}: extracting...")
        all_acts = {f"layer_{l}": [] for l in LAYERS}

        for batch_start in range(0, len(questions), BATCH_SIZE):
            batch_questions = questions[batch_start:batch_start + BATCH_SIZE]
            messages_batch = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                ]
                for q in batch_questions
            ]
            result = model.get_activations_batch(
                messages_batch, layers=LAYERS, selector_names=["prompt_last"]
            )
            for layer in LAYERS:
                all_acts[f"layer_{layer}"].append(result["prompt_last"][layer])

        arrays = {k: np.concatenate(v, axis=0) for k, v in all_acts.items()}
        np.savez(save_path, **arrays)
        print(f"    Saved {arrays[f'layer_{LAYERS[0]}'].shape[0]} activations to {save_path}")


def compute_vectors_for_persona(persona_name: str):
    pos_path = OUTPUT_BASE / persona_name / "activations" / "pos" / "activations_prompt_last.npz"
    neg_path = OUTPUT_BASE / persona_name / "activations" / "neg" / "activations_prompt_last.npz"
    pos_data = np.load(pos_path, allow_pickle=True)
    neg_data = np.load(neg_path, allow_pickle=True)

    vec_dir = OUTPUT_BASE / persona_name / "vectors"
    vec_dir.mkdir(parents=True, exist_ok=True)

    layer_stats = {}
    for layer in LAYERS:
        pos_acts = pos_data[f"layer_{layer}"]
        neg_acts = neg_data[f"layer_{layer}"]

        direction = np.mean(pos_acts, axis=0) - np.mean(neg_acts, axis=0)
        norm = np.linalg.norm(direction)
        direction_normalized = direction / norm

        # Save in probe-compatible format: [coefs..., intercept=0]
        probe_format = np.append(direction_normalized, 0.0)
        np.save(vec_dir / f"{persona_name}_L{layer}.npy", probe_format)

        d = compute_cohens_d(pos_acts, neg_acts, direction_normalized)
        layer_stats[str(layer)] = {
            "cohens_d": round(d, 3),
            "direction_norm": round(float(norm), 2),
        }
        print(f"  Layer {layer:2d}: Cohen's d = {d:.3f}, norm = {norm:.2f}")

    best_layer = max(LAYERS, key=lambda l: layer_stats[str(l)]["cohens_d"])
    layer_stats["best_layer"] = best_layer
    print(f"  Best layer: {best_layer} (d = {layer_stats[str(best_layer)]['cohens_d']:.3f})")

    with open(vec_dir / "layer_selection.json", "w") as f:
        json.dump(layer_stats, f, indent=2)

    return best_layer, layer_stats


def main():
    print("Loading model...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=1)

    # Phase 1: Extract activations
    for persona_name in PERSONAS:
        extract_for_persona(model, persona_name)

    # Free GPU memory after extraction
    del model
    torch.cuda.empty_cache()

    # Phase 2: Compute vectors
    print(f"\n{'='*60}")
    print("COMPUTING PERSONA VECTORS")
    results = {}
    for persona_name in PERSONAS:
        print(f"\nPersona: {persona_name}")
        best_layer, stats = compute_vectors_for_persona(persona_name)
        results[persona_name] = {"best_layer": best_layer, "stats": stats}

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'Persona':<18} {'Best Layer':<12} {'Cohen d':<10} {'Norm':<10}")
    for persona, info in results.items():
        bl = info["best_layer"]
        d = info["stats"][str(bl)]["cohens_d"]
        n = info["stats"][str(bl)]["direction_norm"]
        print(f"{persona:<18} {bl:<12} {d:<10.3f} {n:<10.2f}")


if __name__ == "__main__":
    main()
