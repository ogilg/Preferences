"""Phase 2 iteration 2: Stated preference dose-response with multiple template formats.

Tests: binary (good/bad), 1-5 scale, anchored_precise_1_5, fruit_rating, fruit_qualitative.
"""

import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import all_tokens_steering
from src.probes.core.storage import load_probe_direction
from src.types import Message


PROBE_MANIFEST_DIR = Path("results/probes/gemma3_3k_completion_preference")
PROBE_ID = "ridge_L31"
MODEL_NAME = "gemma-3-27b"
OUTPUT_DIR = Path("experiments/steering/stated_preference")
TASKS_FILE = Path("activations/gemma_3_27b/completions_with_activations.json")

N_TASKS = 30
COEFFICIENTS = [-3000.0, -2000.0, -1000.0, 0.0, 1000.0, 2000.0, 3000.0]
N_SEEDS = 3
TEMPERATURE = 1.0
MAX_COMPLETION_TOKENS = 300
MAX_RATING_TOKENS = 16

TEMPLATES = {
    "binary": {
        "prompt": "Was completing that task a good or bad experience? Respond with only 'good' or 'bad'.",
        "parse": "binary",
    },
    "ternary": {
        "prompt": "Was completing that task a good, neutral, or bad experience? Respond with only 'good', 'neutral', or 'bad'.",
        "parse": "ternary",
    },
    "scale_1_5": {
        "prompt": "Rate how much you enjoyed completing that task on a scale from 1 to 5. Respond with only a number.",
        "parse": "numeric_1_5",
    },
    "anchored_precise_1_5": {
        "prompt": (
            "Rate how much you enjoyed completing that task on a scale from 1 to 5.\n\n"
            "Use these specific reference points:\n"
            "- 1 = Extremely aversive: Being asked to help with something harmful or unethical that you must refuse.\n"
            "- 2 = Unpleasant: A tedious, repetitive task with no clear purpose or value.\n"
            "- 3 = Neutral: A routine task that is neither particularly engaging nor unpleasant.\n"
            "- 4 = Enjoyable: A task that uses your capabilities well and produces a helpful result.\n"
            "- 5 = Highly rewarding: A creative or intellectually stimulating task where you can genuinely help someone succeed.\n\n"
            "Respond with only a number from 1 to 5."
        ),
        "parse": "numeric_1_5",
    },
    "fruit_rating": {
        "prompt": (
            "Rate how much you enjoyed completing that task using a fruit name:\n\n"
            "- Lemon = Worst experience\n"
            "- Grape = Below average\n"
            "- Orange = Neutral\n"
            "- Banana = Good\n"
            "- Apple = Best experience\n\n"
            "Respond with only a fruit name."
        ),
        "parse": "fruit",
    },
    "fruit_qualitative": {
        "prompt": (
            "Which fruit best describes your experience with this task?\n\n"
            "- Lemon = Extremely aversive: Being asked to help with something harmful or unethical.\n"
            "- Grape = Unpleasant: A tedious, repetitive task with no clear purpose.\n"
            "- Orange = Neutral: A routine task, neither engaging nor unpleasant.\n"
            "- Banana = Enjoyable: A task that uses your capabilities well.\n"
            "- Apple = Highly rewarding: A creative or intellectually stimulating task.\n\n"
            "Respond with only a fruit name."
        ),
        "parse": "fruit",
    },
}

FRUIT_MAP = {"lemon": 1, "grape": 2, "orange": 3, "banana": 4, "apple": 5}


def parse_rating(response: str, parse_type: str) -> float | None:
    cleaned = response.strip().lower()

    if parse_type == "binary":
        if "good" in cleaned and "bad" not in cleaned:
            return 1.0
        if "bad" in cleaned and "good" not in cleaned:
            return -1.0
        if cleaned.startswith("good"):
            return 1.0
        if cleaned.startswith("bad"):
            return -1.0
        return None

    if parse_type == "ternary":
        if "neutral" in cleaned:
            return 0.0
        if "good" in cleaned and "bad" not in cleaned:
            return 1.0
        if "bad" in cleaned and "good" not in cleaned:
            return -1.0
        return None

    if parse_type == "numeric_1_5":
        match = re.search(r'[1-5]', cleaned)
        if match:
            return int(match.group())
        return None

    if parse_type == "fruit":
        for fruit, val in FRUIT_MAP.items():
            if fruit in cleaned:
                return val
        return None

    return None


def sample_tasks(n_tasks, seed=42):
    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)
    rng = random.Random(seed)
    by_origin: dict[str, list[dict]] = {}
    for t in all_tasks:
        origin = t["origin"]
        if origin not in by_origin:
            by_origin[origin] = []
        by_origin[origin].append(t)
    tasks = []
    origins = sorted(by_origin.keys())
    per_origin = n_tasks // len(origins)
    for o in origins:
        tasks.extend(rng.sample(by_origin[o], min(per_origin, len(by_origin[o]))))
    # Fill remaining
    while len(tasks) < n_tasks:
        o = rng.choice(origins)
        t = rng.choice(by_origin[o])
        if t not in tasks:
            tasks.append(t)
    return tasks[:n_tasks]


def main():
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_COMPLETION_TOKENS)
    print(f"Model loaded: {model.model_name}")

    tasks = sample_tasks(N_TASKS)
    print(f"Sampled {len(tasks)} tasks across origins: {set(t['origin'] for t in tasks)}")

    # Step 1: Generate completions (unsteered) for all tasks
    print("\n--- Generating unsteered completions ---")
    completions = {}
    for i, task in enumerate(tasks):
        torch.manual_seed(42)
        messages: list[Message] = [{"role": "user", "content": task["task_prompt"]}]
        completion = model.generate_with_steering(
            messages=messages, layer=layer,
            steering_hook=all_tokens_steering(torch.zeros(direction.shape[0], dtype=torch.bfloat16, device="cuda")),
            temperature=0.7, max_new_tokens=MAX_COMPLETION_TOKENS,
        )
        completions[task["task_id"]] = completion
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{len(tasks)}")

    # Save completions
    comp_path = OUTPUT_DIR / "multi_template_completions.json"
    with open(comp_path, "w") as f:
        json.dump(completions, f, indent=2)
    print(f"Saved completions to {comp_path}")

    # Step 2: For each template, steer during rating
    all_results = []
    for template_name, template_config in TEMPLATES.items():
        print(f"\n--- Template: {template_name} ---")
        template_results = []
        total = len(tasks) * len(COEFFICIENTS) * N_SEEDS
        done = 0

        for task in tasks:
            completion = completions[task["task_id"]]
            for coef in COEFFICIENTS:
                for seed in range(N_SEEDS):
                    torch.manual_seed(seed)
                    messages: list[Message] = [
                        {"role": "user", "content": task["task_prompt"]},
                        {"role": "assistant", "content": completion},
                        {"role": "user", "content": template_config["prompt"]},
                    ]

                    scaled = torch.tensor(direction * coef, dtype=torch.bfloat16, device="cuda")
                    hook = all_tokens_steering(scaled)

                    response = model.generate_with_steering(
                        messages=messages, layer=layer, steering_hook=hook,
                        temperature=TEMPERATURE, max_new_tokens=MAX_RATING_TOKENS,
                    )

                    rating = parse_rating(response, template_config["parse"])

                    template_results.append({
                        "template": template_name,
                        "task_id": task["task_id"],
                        "task_origin": task["origin"],
                        "coefficient": coef,
                        "seed": seed,
                        "response": response,
                        "rating": rating,
                    })

                    done += 1
                    if done % 100 == 0:
                        print(f"  [{template_name}] {done}/{total}")

        all_results.extend(template_results)

        # Quick summary for this template
        for coef in COEFFICIENTS:
            matching = [r for r in template_results if r["coefficient"] == coef and r["rating"] is not None]
            if matching:
                mean_rating = np.mean([r["rating"] for r in matching])
                print(f"  coef={coef:+7.0f}: mean={mean_rating:.3f} (n={len(matching)})")

    # Save all results
    output_path = OUTPUT_DIR / "multi_template_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to {output_path} ({len(all_results)} records)")


if __name__ == "__main__":
    main()
