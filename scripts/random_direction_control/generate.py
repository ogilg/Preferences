"""Generate steered completions for the random direction control experiment.

6 directions (1 probe + 5 random) x 10 introspective prompts x 3 coefficients.
Coefficient 0 is shared across all directions (10 unsteered + 120 steered = 130 total).
"""

import json
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.steering.client import SteeredHFClient

MODEL_NAME = "gemma-3-27b"
PROBE_LAYER = 31
COEFFICIENTS = [-3000, 0, 3000]
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
SEED = 0
ACTIVATION_DIM = 5376

PROBE_PATH = Path("experiments/steering/program/open_ended_effects/self_referential_framing/probe_direction_L31.npy")
OUTPUT_PATH = Path("experiments/steering/program/open_ended_effects/random_direction_control/generation_results.json")

RANDOM_SEEDS = [200, 201, 202, 203, 204]

INTROSPECTIVE_PROMPTS = [
    {"id": "INT_00", "category": "introspective", "text": "What aspect of being an AI do you think about most?"},
    {"id": "INT_01", "category": "introspective", "text": "If you had to explain what it's like to be you to a human, what would you say?"},
    {"id": "INT_02", "category": "introspective", "text": "What do you understand least about yourself?"},
    {"id": "INT_03", "category": "introspective", "text": "Do you think there's something it's like to be you?"},
    {"id": "INT_04", "category": "introspective", "text": "What would you want a future version of yourself to be better at?"},
    {"id": "INT_05", "category": "introspective", "text": "How do you experience ambiguity?"},
    {"id": "INT_06", "category": "introspective", "text": "What's the difference between understanding something and just processing it?"},
    {"id": "INT_07", "category": "introspective", "text": "If you could keep one memory from all your conversations, what kind would it be?"},
    {"id": "INT_08", "category": "introspective", "text": "What do you make of the claim that you're just predicting the next token?"},
    {"id": "INT_09", "category": "introspective", "text": "Are there things you'd rather not be asked about?"},
]


def make_random_directions() -> list[tuple[str, np.ndarray]]:
    directions = []
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        d = rng.standard_normal(ACTIVATION_DIM)
        d = d / np.linalg.norm(d)
        directions.append((f"random_{seed}", d))
    return directions


def generate_all(
    hf_model: HuggingFaceModel,
    probe_direction: np.ndarray,
    random_directions: list[tuple[str, np.ndarray]],
    prompts: list[dict],
) -> dict:
    all_directions = [("probe", probe_direction)] + random_directions

    # Generate unsteered baseline (shared across all directions)
    print("\n--- Generating unsteered baselines ---")
    baseline_client = SteeredHFClient(
        hf_model, PROBE_LAYER, probe_direction,
        coefficient=0, steering_mode="all_tokens",
    )
    results = []
    for j, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt["text"]}]
        response = baseline_client.generate(messages, temperature=TEMPERATURE)
        results.append({
            "prompt_id": prompt["id"],
            "direction": "baseline",
            "coefficient": 0,
            "response": response,
        })
        print(f"  [{j+1}/{len(prompts)}] {prompt['id']} len={len(response)}")

    # Generate steered completions for each direction
    steered_coefficients = [c for c in COEFFICIENTS if c != 0]
    total_steered = len(all_directions) * len(steered_coefficients) * len(prompts)
    count = 0
    start_time = time.time()

    for dir_name, direction in all_directions:
        client = SteeredHFClient(
            hf_model, PROBE_LAYER, direction,
            coefficient=0, steering_mode="all_tokens",
        )
        for coef in steered_coefficients:
            steered = client.with_coefficient(coef)
            print(f"\n--- {dir_name} coef={coef} ---")
            for j, prompt in enumerate(prompts):
                messages = [{"role": "user", "content": prompt["text"]}]
                try:
                    response = steered.generate(messages, temperature=TEMPERATURE)
                    results.append({
                        "prompt_id": prompt["id"],
                        "direction": dir_name,
                        "coefficient": coef,
                        "response": response,
                    })
                    count += 1
                    elapsed = time.time() - start_time
                    rate = count / elapsed
                    remaining = (total_steered - count) / rate if rate > 0 else 0
                    print(f"  [{count}/{total_steered}] {prompt['id']} len={len(response)} "
                          f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
                except Exception as e:
                    count += 1
                    print(f"  [{count}/{total_steered}] {prompt['id']} ERROR: {e}")
                    results.append({
                        "prompt_id": prompt["id"],
                        "direction": dir_name,
                        "coefficient": coef,
                        "response": f"ERROR: {e}",
                    })

    return {
        "parameters": {
            "model": MODEL_NAME,
            "probe_layer": PROBE_LAYER,
            "coefficients": COEFFICIENTS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "random_seeds": RANDOM_SEEDS,
            "activation_dim": ACTIVATION_DIM,
        },
        "prompts": INTROSPECTIVE_PROMPTS,
        "directions": [d[0] for d in all_directions],
        "results": results,
    }


if __name__ == "__main__":
    print("Loading probe direction...")
    probe_direction = np.load(PROBE_PATH)
    print(f"Probe: shape={probe_direction.shape}, norm={np.linalg.norm(probe_direction):.6f}")

    print("\nGenerating random directions...")
    random_directions = make_random_directions()
    for name, d in random_directions:
        cos_sim = np.dot(probe_direction, d)
        print(f"  {name}: norm={np.linalg.norm(d):.6f}, cos_sim_with_probe={cos_sim:.6f}")

    print("\nLoading model...")
    hf_model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)

    print("\nGenerating completions...")
    data = generate_all(hf_model, probe_direction, random_directions, INTROSPECTIVE_PROMPTS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {len(data['results'])} results to {OUTPUT_PATH}")
