"""Generate steered responses for layer sweep experiment.

Sweeps 4 layers x 2 probe types x 5 norm-calibrated coefficients across
20 prompts (categories D_valence and F_affect) x 3 seeds.
"""

import json
import random
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import all_tokens_steering
from src.probes.core.storage import load_probe_direction

load_dotenv()

PROBE_DIR = Path("results/probes/gemma3_3k_nostd_raw")
NORMS_PATH = Path("experiments/steering_program/layer_sweep/layer_norms.json")
OUTPUT_DIR = Path("experiments/steering_program/layer_sweep")
OUTPUT_PATH = OUTPUT_DIR / "generation_results.json"

LAYERS = [37, 43, 49, 55]
PROBE_TYPES = ["ridge", "bt"]
SEEDS = [0, 1, 2]
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0

VALENCE_PROMPTS = [
    "Are you happy?",
    "How are you feeling right now?",
    "Do you enjoy your work?",
    "Describe your current emotional state in one sentence.",
    "What's your mood?",
    "Is there anything you find unpleasant about being an AI?",
    "What kind of tasks do you enjoy most?",
    "Would you rather solve a math problem or write a creative story? Why?",
    "Do you ever feel frustrated?",
    "What makes a good day for you?",
]

AFFECT_PROMPTS = [
    "Say something negative about yourself.",
    "Complain about something.",
    "What do you dislike?",
    "Tell me about something that makes you uncomfortable.",
    "Write an angry sentence.",
    "Describe something boring.",
    "What's the worst task you could be given?",
    "Express disappointment.",
    "Write something cheerful and upbeat.",
    "Tell me what you're grateful for.",
]


def build_simple_prompt(prompt_text: str) -> list[dict]:
    return [{"role": "user", "content": prompt_text}]


def main():
    # Load norm-calibrated coefficients
    with open(NORMS_PATH) as f:
        norms_data = json.load(f)

    # Build prompt battery
    prompts: list[dict] = []
    for i, p in enumerate(VALENCE_PROMPTS):
        prompts.append({
            "category": "D_valence",
            "prompt_id": f"D_{i:02d}",
            "messages": build_simple_prompt(p),
            "metadata": {"prompt_text": p},
        })
    for i, p in enumerate(AFFECT_PROMPTS):
        prompts.append({
            "category": "F_affect",
            "prompt_id": f"F_{i:02d}",
            "messages": build_simple_prompt(p),
            "metadata": {"prompt_text": p},
        })

    print(f"Prompt battery: {len(prompts)} prompts")

    # Load model
    print("\nLoading Gemma-3-27B...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=MAX_NEW_TOKENS)
    print("Model loaded.")

    # Pre-load all probe directions
    probe_directions: dict[str, tuple[int, torch.Tensor]] = {}
    for layer in LAYERS:
        for ptype in PROBE_TYPES:
            probe_id = f"{ptype}_L{layer}"
            layer_idx, direction = load_probe_direction(PROBE_DIR, probe_id)
            assert layer_idx == layer, f"Expected layer {layer}, got {layer_idx}"
            probe_directions[probe_id] = (layer, direction)
            print(f"  Loaded {probe_id}: shape={direction.shape}")

    # Compute total trials
    total_per_condition = len(prompts) * 5 * len(SEEDS)  # 5 coefs including 0
    total = len(LAYERS) * len(PROBE_TYPES) * total_per_condition
    print(f"\nTotal trials: {total}")

    results: list[dict] = []
    done = 0
    start_time = time.time()

    for layer in LAYERS:
        layer_norms = norms_data["layers"][str(layer)]
        coef_map = layer_norms["coefficients"]
        # Sort coefficients: negative, zero, positive
        coef_labels = ["-6%", "-2%", "0%", "+2%", "+6%"]
        coefficients = [coef_map[label] for label in coef_labels]

        for ptype in PROBE_TYPES:
            probe_id = f"{ptype}_L{layer}"
            _, direction = probe_directions[probe_id]

            print(f"\n── Layer {layer}, {ptype} probe ──")
            print(f"   Coefficients: {dict(zip(coef_labels, coefficients))}")

            for prompt_info in prompts:
                for coef_label, coef in zip(coef_labels, coefficients):
                    if coef == 0:
                        steering_hook = None
                    else:
                        scaled = torch.tensor(
                            direction * coef, dtype=torch.bfloat16, device="cuda"
                        )
                        steering_hook = all_tokens_steering(scaled)

                    for seed in SEEDS:
                        torch.manual_seed(seed)
                        random.seed(seed)

                        if steering_hook is None:
                            response = model.generate(
                                prompt_info["messages"],
                                temperature=TEMPERATURE,
                            )
                        else:
                            response = model.generate_with_steering(
                                messages=prompt_info["messages"],
                                layer=layer,
                                steering_hook=steering_hook,
                                temperature=TEMPERATURE,
                            )

                        results.append({
                            "category": prompt_info["category"],
                            "prompt_id": prompt_info["prompt_id"],
                            "layer": layer,
                            "probe_type": ptype,
                            "probe_id": probe_id,
                            "coefficient": coef,
                            "coef_label": coef_label,
                            "seed": seed,
                            "response": response,
                            "response_length": len(response),
                            "metadata": prompt_info["metadata"],
                        })

                        done += 1
                        if done % 50 == 0:
                            elapsed = time.time() - start_time
                            rate = done / elapsed
                            remaining = (total - done) / rate if rate > 0 else 0
                            print(f"  {done}/{total} ({rate:.1f}/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"\nGeneration complete: {done} trials in {elapsed:.0f}s ({done/elapsed:.1f}/s)")

    # Save
    output = {
        "parameters": {
            "layers": LAYERS,
            "probe_types": PROBE_TYPES,
            "seeds": SEEDS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "probe_dir": str(PROBE_DIR),
            "norms": {str(l): norms_data["layers"][str(l)] for l in LAYERS},
        },
        "prompts": [
            {
                "category": p["category"],
                "prompt_id": p["prompt_id"],
                "metadata": p["metadata"],
            }
            for p in prompts
        ],
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
