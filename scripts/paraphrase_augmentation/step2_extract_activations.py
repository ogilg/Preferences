"""Step 2: Extract Layer 31 prompt_last activations for paraphrased tasks."""

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.models.huggingface_model import HuggingFaceModel

load_dotenv()

PARAPHRASES_FILE = Path("experiments/probe_science/paraphrase_augmentation/paraphrases.json")
EXISTING_ACTIVATIONS = Path("activations/gemma_3_27b/activations_prompt_last.npz")
OUTPUT_FILE = Path("experiments/probe_science/paraphrase_augmentation/paraphrase_activations.npz")

LAYER = 31
BATCH_SIZE = 16


def main():
    with open(PARAPHRASES_FILE) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} paraphrases")

    task_ids = [f"{entry['task_id']}_para" for entry in data]
    messages_list = [
        [{"role": "user", "content": entry["paraphrased_prompt"]}]
        for entry in data
    ]

    print("Loading model...")
    model = HuggingFaceModel("gemma-3-27b")

    print(f"Extracting Layer {LAYER} activations for {len(messages_list)} paraphrases...")
    all_activations = []

    for batch_start in range(0, len(messages_list), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(messages_list))
        batch_messages = messages_list[batch_start:batch_end]

        # Returns {selector: {layer: (batch, d_model) ndarray}}
        batch_results = model.get_activations_batch(
            messages_batch=batch_messages,
            layers=[LAYER],
            selector_names=["prompt_last"],
        )

        batch_acts = batch_results["prompt_last"][LAYER]  # (batch_size, d_model)
        all_activations.append(batch_acts)
        print(f"  Extracted {batch_end}/{len(messages_list)}")

    activations_array = np.concatenate(all_activations, axis=0)  # (100, 5376)
    task_ids_array = np.array(task_ids)

    print(f"Activations shape: {activations_array.shape}")

    # Verify dimensions match existing activations
    existing = np.load(EXISTING_ACTIVATIONS, allow_pickle=True)
    existing_dim = existing[f"layer_{LAYER}"].shape[1]
    assert activations_array.shape[1] == existing_dim, (
        f"Dimension mismatch: paraphrase {activations_array.shape[1]} vs existing {existing_dim}"
    )

    np.savez(OUTPUT_FILE, task_ids=task_ids_array, **{f"layer_{LAYER}": activations_array})
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
