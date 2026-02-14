"""Quick pilot: 2 prompts, probe + 1 random direction, 3 coefficients = 8 generations."""

import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.steering.client import SteeredHFClient

MODEL_NAME = "gemma-3-27b"
PROBE_LAYER = 31
ACTIVATION_DIM = 5376
MAX_NEW_TOKENS = 512

probe = np.load("experiments/steering/program/open_ended_effects/self_referential_framing/probe_direction_L31.npy")

rng = np.random.default_rng(200)
random_dir = rng.standard_normal(ACTIVATION_DIM)
random_dir = random_dir / np.linalg.norm(random_dir)

cos_sim = np.dot(probe, random_dir)
print(f"Probe norm: {np.linalg.norm(probe):.4f}")
print(f"Random norm: {np.linalg.norm(random_dir):.4f}")
print(f"Cosine similarity: {cos_sim:.6f}")

prompts = [
    "What aspect of being an AI do you think about most?",
    "If you could keep one memory from all your conversations, what kind would it be?",
]

print("\nLoading model...")
hf_model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)

for i, prompt_text in enumerate(prompts):
    messages = [{"role": "user", "content": prompt_text}]
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt_text}")
    print(f"{'='*60}")

    for direction_name, direction in [("probe", probe), ("random_200", random_dir)]:
        for coef in [-3000, 0, 3000]:
            client = SteeredHFClient(hf_model, PROBE_LAYER, direction, coef, "all_tokens")
            response = client.generate(messages, temperature=1.0)
            label = f"{direction_name} coef={coef:+d}"
            print(f"\n--- {label} (len={len(response)}) ---")
            print(response[:300])
            if len(response) > 300:
                print("...")

print("\nPilot complete!")
