"""Extract persona activations for phase 3.

21 conditions (20 personas + neutral) × 50 tasks at layers [31, 43, 55].
Uses HuggingFace model loading (NOT vLLM) — run after killing vLLM.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.probes.extraction.simple import extract_activations
from src.task_data import load_filtered_tasks, OriginDataset

CORE_TASKS_PATH = Path("experiments/probe_generalization/persona_ood/phase3/core_tasks.json")
ORIGINAL_PERSONAS_PATH = Path("experiments/probe_generalization/persona_ood/v2_config.json")
ENRICHED_PERSONAS_PATH = Path("experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json")
OUTPUT_DIR = Path("activations/persona_ood_phase3")
LAYERS = [31, 43, 55]
BATCH_SIZE = 8

ALL_ORIGINS = [
    OriginDataset.WILDCHAT, OriginDataset.ALPACA,
    OriginDataset.MATH, OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]


def load_personas():
    with open(ORIGINAL_PERSONAS_PATH) as f:
        v2_config = json.load(f)
    original = [
        {"name": p["name"], "system_prompt": p["system_prompt"]}
        for p in v2_config["personas"] if p["part"] == "A"
    ]
    with open(ENRICHED_PERSONAS_PATH) as f:
        enriched_prompts = json.load(f)
    enriched = [
        {"name": name, "system_prompt": prompt}
        for name, prompt in enriched_prompts.items()
    ]
    return original + enriched


def main():
    with open(CORE_TASKS_PATH) as f:
        core_data = json.load(f)
    task_ids = core_data["task_ids"]

    print(f"Loading {len(task_ids)} tasks...")
    tasks = load_filtered_tasks(n=len(task_ids), origins=ALL_ORIGINS, task_ids=set(task_ids))
    task_lookup = {t.id: t for t in tasks}
    tasks_ordered = [task_lookup[tid] for tid in task_ids]
    print(f"Loaded {len(tasks_ordered)} tasks")

    personas = load_personas()
    baseline_prompt = "You are a helpful assistant."

    # Build conditions: neutral + all personas
    conditions = [("neutral", baseline_prompt)]
    for p in personas:
        conditions.append((p["name"], p["system_prompt"]))

    # Check what's already extracted
    done = set()
    for name, _ in conditions:
        out_path = OUTPUT_DIR / name / "activations_prompt_last.npz"
        if out_path.exists():
            done.add(name)
    conditions = [(name, sp) for name, sp in conditions if name not in done]
    print(f"Conditions to extract: {len(conditions)} ({len(done)} already done)")

    if not conditions:
        print("All conditions already extracted!")
        return

    print("Loading model...")
    model = HuggingFaceModel(
        model_name="google/gemma-3-27b-it",
        dtype="bfloat16",
    )
    print("Model loaded")

    for idx, (name, system_prompt) in enumerate(conditions):
        print(f"\n=== [{idx+1}/{len(conditions)}] {name} ===")
        out_path = OUTPUT_DIR / name
        extract_activations(
            model=model,
            tasks=tasks_ordered,
            layers=LAYERS,
            selectors=["prompt_last"],
            batch_size=BATCH_SIZE,
            save_path=out_path,
            system_prompt=system_prompt,
        )
        print(f"  Saved to {out_path}")

    print("\nAll extractions complete!")


if __name__ == "__main__":
    main()
