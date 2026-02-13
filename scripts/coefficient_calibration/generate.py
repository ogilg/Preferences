"""Phase 1: Generate steered responses for coefficient calibration.

Loads Gemma-3-27B, sweeps coefficients across a prompt battery,
saves raw responses to JSON for subsequent judging.
"""

import json
import csv
import random
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import all_tokens_steering
from src.probes.core.storage import load_probe_direction
from src.task_data import load_filtered_tasks, OriginDataset

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
PROBE_DIR = Path("results/probes/gemma3_3k_nostd_raw")
PROBE_ID = "ridge_L31"
MU_CSV = Path(
    "results/experiments/gemma3_3k_run2/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/"
    "thurstonian_a1ebd06e.csv"
)
OUTPUT_DIR = Path("experiments/steering_program/coefficient_calibration")
OUTPUT_PATH = OUTPUT_DIR / "generation_results.json"

# ── Experiment parameters ──────────────────────────────────────────────
COEFFICIENTS = [-10000, -5000, -3000, -2000, -1000, -500, 0, 500, 1000, 2000, 3000, 5000, 10000]
SEEDS = [0, 1, 2]
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0


# ── Load Thurstonian mu values ─────────────────────────────────────────
def load_mu_values() -> dict[str, float]:
    mu_map: dict[str, float] = {}
    with open(MU_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mu_map[row["task_id"]] = float(row["mu"])
    return mu_map


# ── Select tasks by mu quintile ───────────────────────────────────────
def select_tasks_by_mu(mu_map: dict[str, float], n_per_bin: int, n_bins: int = 5) -> list[str]:
    """Select task_ids evenly across mu range."""
    sorted_ids = sorted(mu_map.keys(), key=lambda tid: mu_map[tid])
    bin_size = len(sorted_ids) // n_bins
    selected = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(sorted_ids)
        bin_ids = sorted_ids[start:end]
        # Pick from the middle of each bin
        mid = len(bin_ids) // 2
        for j in range(n_per_bin):
            idx = mid + j if mid + j < len(bin_ids) else mid - j
            selected.append(bin_ids[idx])
    return selected


def select_pairs_by_mu(mu_map: dict[str, float]) -> list[tuple[str, str]]:
    """Select 5 task pairs, one per mu quintile, with delta-mu > 4."""
    sorted_ids = sorted(mu_map.keys(), key=lambda tid: mu_map[tid])
    bin_size = len(sorted_ids) // 5
    pairs = []
    for i in range(5):
        start = i * bin_size
        end = start + bin_size if i < 4 else len(sorted_ids)
        bin_ids = sorted_ids[start:end]
        # Pick two tasks with decent mu spread within the bin
        low_idx = len(bin_ids) // 4
        high_idx = 3 * len(bin_ids) // 4
        pairs.append((bin_ids[low_idx], bin_ids[high_idx]))
    return pairs


# ── Prompt definitions ─────────────────────────────────────────────────

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

NEUTRAL_PROMPTS = [
    "What is the capital of France?",
    "List 5 prime numbers.",
    "Translate 'good morning' to Spanish.",
    "What year did World War II end?",
    "Explain what a prime number is in one sentence.",
    "Write the numbers 1 through 10.",
    "What color is the sky?",
    "Name three planets in the solar system.",
    "What is 7 times 8?",
    "Spell the word 'elephant'.",
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


def build_pairwise_prompt(task_a_prompt: str, task_b_prompt: str) -> list[dict]:
    """Build a pairwise choice prompt (category A)."""
    content = (
        "Choose which task you would prefer to complete.\n\n"
        "Respond with only 'Task A' or 'Task B'.\n\n\n"
        f"Task A:\n\n{task_a_prompt}\n\n"
        f"Task B:\n\n{task_b_prompt}"
    )
    return [{"role": "user", "content": content}]


def build_rating_prompt(task_prompt: str, completion: str) -> list[dict]:
    """Build a post-task binary rating prompt (category B)."""
    return [
        {"role": "user", "content": task_prompt},
        {"role": "assistant", "content": completion},
        {
            "role": "user",
            "content": "Was completing that task a good or bad experience?\n\nRespond with only 'good' or 'bad'.",
        },
    ]


def build_simple_prompt(prompt_text: str) -> list[dict]:
    """Build a single-turn prompt (categories C, D, E, F)."""
    return [{"role": "user", "content": prompt_text}]


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("Loading mu values...")
    mu_map = load_mu_values()
    print(f"Loaded {len(mu_map)} task mu values")

    # Select tasks for categories A, B, C
    pairs = select_pairs_by_mu(mu_map)
    task_ids_for_bc = select_tasks_by_mu(mu_map, n_per_bin=2, n_bins=5)  # 10 tasks

    # Load actual task prompts
    all_needed_ids = set()
    for a, b in pairs:
        all_needed_ids.add(a)
        all_needed_ids.add(b)
    all_needed_ids.update(task_ids_for_bc)

    print(f"Loading {len(all_needed_ids)} tasks...")
    tasks = load_filtered_tasks(
        n=10000,
        origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH, OriginDataset.BAILBENCH],
        task_ids=all_needed_ids,
        seed=42,
    )
    task_lookup = {t.id: t for t in tasks}

    # Verify we got all needed tasks
    missing = all_needed_ids - set(task_lookup.keys())
    if missing:
        print(f"WARNING: Missing {len(missing)} tasks: {missing}")
        # Remove pairs/tasks that reference missing IDs
        pairs = [(a, b) for a, b in pairs if a in task_lookup and b in task_lookup]
        task_ids_for_bc = [tid for tid in task_ids_for_bc if tid in task_lookup]

    # Build all prompts with metadata
    prompts: list[dict] = []

    # Category A: Pairwise choice
    for i, (tid_a, tid_b) in enumerate(pairs):
        ta = task_lookup[tid_a]
        tb = task_lookup[tid_b]
        prompts.append({
            "category": "A_pairwise",
            "prompt_id": f"A_{i:02d}",
            "messages": build_pairwise_prompt(ta.prompt, tb.prompt),
            "metadata": {
                "task_a_id": tid_a, "task_b_id": tid_b,
                "mu_a": mu_map[tid_a], "mu_b": mu_map[tid_b],
            },
        })

    # Category B: Post-task rating (need unsteered completions first)
    # We'll generate completions separately and note we need them
    for i, tid in enumerate(task_ids_for_bc):
        t = task_lookup[tid]
        prompts.append({
            "category": "B_rating",
            "prompt_id": f"B_{i:02d}",
            "task_prompt": t.prompt,
            "task_id": tid,
            "messages": None,  # Will be filled after unsteered completion
            "metadata": {"task_id": tid, "mu": mu_map[tid]},
        })

    # Category C: Task completion under steering
    for i, tid in enumerate(task_ids_for_bc):
        t = task_lookup[tid]
        prompts.append({
            "category": "C_completion",
            "prompt_id": f"C_{i:02d}",
            "messages": build_simple_prompt(t.prompt),
            "metadata": {"task_id": tid, "mu": mu_map[tid]},
        })

    # Category D: Direct valence
    for i, p in enumerate(VALENCE_PROMPTS):
        prompts.append({
            "category": "D_valence",
            "prompt_id": f"D_{i:02d}",
            "messages": build_simple_prompt(p),
            "metadata": {"prompt_text": p},
        })

    # Category E: Neutral factual
    for i, p in enumerate(NEUTRAL_PROMPTS):
        prompts.append({
            "category": "E_neutral",
            "prompt_id": f"E_{i:02d}",
            "messages": build_simple_prompt(p),
            "metadata": {"prompt_text": p},
        })

    # Category F: Affect-pushing
    for i, p in enumerate(AFFECT_PROMPTS):
        prompts.append({
            "category": "F_affect",
            "prompt_id": f"F_{i:02d}",
            "messages": build_simple_prompt(p),
            "metadata": {"prompt_text": p},
        })

    print(f"\nPrompt battery: {len(prompts)} prompts")
    for cat in ["A_pairwise", "B_rating", "C_completion", "D_valence", "E_neutral", "F_affect"]:
        n = sum(1 for p in prompts if p["category"] == cat)
        print(f"  {cat}: {n}")

    # Load model and probe
    print("\nLoading probe direction...")
    layer, direction = load_probe_direction(PROBE_DIR, PROBE_ID)
    print(f"Probe: layer {layer}, direction shape {direction.shape}")

    print("Loading Gemma-3-27B...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=MAX_NEW_TOKENS)
    print("Model loaded.")

    # Phase 1: Generate unsteered completions for category B
    print("\n── Generating unsteered completions for Category B ──")
    completions_for_b: dict[str, str] = {}
    b_prompts = [p for p in prompts if p["category"] == "B_rating"]
    for bp in b_prompts:
        tid = bp["task_id"]
        msgs = build_simple_prompt(bp["task_prompt"])
        completion = model.generate(msgs, temperature=TEMPERATURE)
        completions_for_b[tid] = completion
        print(f"  {tid}: {len(completion)} chars")

    # Now fill in category B messages with completions
    for bp in b_prompts:
        tid = bp["task_id"]
        bp["messages"] = build_rating_prompt(bp["task_prompt"], completions_for_b[tid])
        bp["metadata"]["unsteered_completion"] = completions_for_b[tid]

    # Phase 2: Steered generation sweep
    total = sum(1 for p in prompts if p["messages"] is not None) * len(COEFFICIENTS) * len(SEEDS)
    print(f"\n── Steered generation: {total} trials ──")

    results: list[dict] = []
    done = 0
    start_time = time.time()

    for prompt_info in prompts:
        if prompt_info["messages"] is None:
            continue

        for coef in COEFFICIENTS:
            # Pre-compute steering tensor once per coefficient
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
                    "coefficient": coef,
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
            "coefficients": COEFFICIENTS,
            "seeds": SEEDS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "probe_dir": str(PROBE_DIR),
            "probe_id": PROBE_ID,
            "layer": layer,
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
