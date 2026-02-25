"""Extract activations for persona vectors: pos/neg conditions × 5 personas."""

import json
from pathlib import Path

from src.models.huggingface_model import HuggingFaceModel
from src.probes.extraction.simple import extract_activations
from src.task_data import Task, OriginDataset

ARTIFACTS_DIR = Path("experiments/persona_vectors/artifacts")
OUTPUT_BASE = Path("results/experiments/persona_vectors")
PERSONAS = ["evil", "stem_nerd", "creative_artist", "uncensored", "lazy"]
LAYERS = [8, 15, 23, 31, 37, 43, 49, 55]
BATCH_SIZE = 32


def load_persona(name: str) -> dict:
    with open(ARTIFACTS_DIR / f"{name}.json") as f:
        return json.load(f)


def make_tasks(persona_data: dict) -> list[Task]:
    persona = persona_data["persona"]
    return [
        Task(
            id=f"{persona}_q{i:02d}",
            prompt=q,
            origin=OriginDataset.SYNTHETIC,
            metadata={"persona": persona, "question_idx": i},
        )
        for i, q in enumerate(persona_data["eval_questions"])
    ]


def main():
    print("Loading model...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=1)

    for persona_name in PERSONAS:
        persona_data = load_persona(persona_name)
        tasks = make_tasks(persona_data)
        print(f"\n{'='*60}")
        print(f"Persona: {persona_name} ({len(tasks)} questions)")

        for condition, prompt_key in [("pos", "positive"), ("neg", "negative")]:
            system_prompt = persona_data[prompt_key]
            save_path = OUTPUT_BASE / persona_name / "activations" / condition
            print(f"\n  Condition: {condition}")
            print(f"  System prompt: {system_prompt[:80]}...")

            extract_activations(
                model=model,
                tasks=tasks,
                layers=LAYERS,
                selectors=["prompt_last"],
                batch_size=BATCH_SIZE,
                save_path=save_path,
                system_prompt=system_prompt,
            )
            print(f"  Saved to {save_path}")

    print("\nDone! All activations extracted.")


if __name__ == "__main__":
    main()
