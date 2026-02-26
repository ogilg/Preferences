"""Phase 2: Extract activations for all personas under pos/neg conditions.

Extracts both prompt_last and prompt_mean selectors for layers
[15, 23, 31, 37, 43, 49, 55] using the extraction split (indices 0-29).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.models.huggingface_model import HuggingFaceModel
from src.probes.extraction.simple import extract_activations
from src.task_data import Task, OriginDataset


PERSONAS = ["creative_artist", "evil", "lazy", "stem_nerd", "uncensored"]
LAYERS = [15, 23, 31, 37, 43, 49, 55]
SELECTORS = ["prompt_last", "prompt_mean"]
EXTRACTION_INDICES = list(range(0, 30))
BATCH_SIZE = 32

ARTIFACTS_DIR = Path("experiments/persona_vectors/artifacts")
OUTPUT_DIR = Path("results/experiments/persona_vectors_v2")


def load_persona(name: str) -> dict:
    with open(ARTIFACTS_DIR / f"{name}.json") as f:
        return json.load(f)


def make_tasks(questions: list[str], persona_name: str, condition: str) -> list[Task]:
    return [
        Task(
            prompt=q,
            origin=OriginDataset.SYNTHETIC,
            id=f"{persona_name}_{condition}_{i}",
            metadata={},
        )
        for i, q in enumerate(questions)
    ]


def main():
    print("Loading model...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=256)
    print(f"Model loaded. n_layers={model.n_layers}, hidden_dim={model.hidden_dim}")

    for persona_name in PERSONAS:
        print(f"\n{'='*60}")
        print(f"Processing persona: {persona_name}")
        persona = load_persona(persona_name)

        extraction_questions = [
            persona["eval_questions"][i] for i in EXTRACTION_INDICES
        ]

        for condition, prompt_key in [("pos", "positive"), ("neg", "negative")]:
            system_prompt = persona[prompt_key]
            tasks = make_tasks(extraction_questions, persona_name, condition)

            save_path = OUTPUT_DIR / persona_name / "activations" / condition
            save_path.mkdir(parents=True, exist_ok=True)

            print(f"  Extracting {condition} condition ({len(tasks)} tasks)...")
            extract_activations(
                model=model,
                tasks=tasks,
                layers=LAYERS,
                selectors=SELECTORS,
                batch_size=BATCH_SIZE,
                save_path=save_path,
                system_prompt=system_prompt,
            )
            print(f"  Saved to {save_path}")

    print("\nPhase 2 complete!")


if __name__ == "__main__":
    main()
