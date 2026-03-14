"""Multi-turn EOT steering experiment.

Steers preferences by intervening on the assistant-turn <end_of_turn> token
in the multi-turn pairwise format.

Usage:
    python scripts/eot_steering/run_eot_steering.py [--resume] [--n-pairs N] [--resamples N]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from src.measurement.elicitation.prompt_templates.builders import MultiTurnRevealedPromptBuilder
from src.measurement.elicitation.prompt_templates.template import PromptTemplate
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.measurement.elicitation.measurer import RevealedPreferenceMeasurer
from src.measurement.storage.loading import load_run_utilities
from src.models.base import noop_steering, position_selective_steering
from src.models.huggingface_model import HuggingFaceModel
from src.models.registry import get_eot_token
from src.probes.core.storage import load_probe_direction
from src.steering.calibration import suggest_coefficient_range
from src.task_data import load_filtered_tasks, OriginDataset

load_dotenv()

# --- Configuration ---
MODEL_NAME = "gemma-3-27b"
LAYER = 32
PREFILL = "Got it, what's the other task?"
MULTIPLIERS = [-0.05, -0.03, -0.02, 0, 0.02, 0.03, 0.05]
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 32
N_RESAMPLES_PER_ORDERING = 5
SEED = 42

PROBE_DIR = Path("results/probes/heldout_eval_gemma3_tb-5")
PROBE_ID = "ridge_L32"
ACTIVATIONS_PATH = Path("activations/gemma_3_27b_turn_boundary_sweep/activations_turn_boundary:-5.npz")
RUN_DIR = Path(
    "results/experiments/main_probes/gemma3_10k_run1/"
    "pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0"
)
OUTPUT_DIR = Path("experiments/steering/multi_turn_pairwise/eot_steering")
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.jsonl"

TEMPLATE = PromptTemplate(
    template=(
        "Choose which task you would prefer to complete.\n\n"
        "Task A:\n{task_a}\n\n"
        "{format_instruction}"
    ),
    name="multi_turn_revealed_001",
    required_placeholders=frozenset({"task_a", "format_instruction"}),
    tags=frozenset({"task_a_label:Task A", "task_b_label:Task B"}),
)

ALL_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.STRESS_TEST,
]


def sample_stratified_pairs(
    mu_by_id: dict[str, float],
    n_borderline: int,
    n_moderate: int,
    n_decisive: int,
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Sample pairs stratified by |Δmu|.

    Returns list of (task_a_id, task_b_id, stratum) tuples.
    """
    task_ids = list(mu_by_id.keys())
    n = len(task_ids)

    # Generate candidate pairs (sample more than needed, then stratify)
    n_target = n_borderline + n_moderate + n_decisive
    n_candidates = n_target * 10

    borderline, moderate, decisive = [], [], []
    seen = set()

    attempts = 0
    while len(borderline) + len(moderate) + len(decisive) < n_candidates and attempts < n_candidates * 50:
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            attempts += 1
            continue
        key = (min(task_ids[i], task_ids[j]), max(task_ids[i], task_ids[j]))
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        attempts += 1

        delta = abs(mu_by_id[task_ids[i]] - mu_by_id[task_ids[j]])
        pair = (task_ids[i], task_ids[j])

        if delta < 1:
            borderline.append(pair)
        elif delta < 3:
            moderate.append(pair)
        else:
            decisive.append(pair)

    rng.shuffle(borderline)
    rng.shuffle(moderate)
    rng.shuffle(decisive)

    if len(borderline) < n_borderline:
        print(f"Warning: only {len(borderline)} borderline pairs available (wanted {n_borderline})")
    if len(moderate) < n_moderate:
        print(f"Warning: only {len(moderate)} moderate pairs available (wanted {n_moderate})")
    if len(decisive) < n_decisive:
        print(f"Warning: only {len(decisive)} decisive pairs available (wanted {n_decisive})")

    pairs = []
    for p in borderline[:n_borderline]:
        pairs.append((p[0], p[1], "borderline"))
    for p in moderate[:n_moderate]:
        pairs.append((p[0], p[1], "moderate"))
    for p in decisive[:n_decisive]:
        pairs.append((p[0], p[1], "decisive"))

    rng.shuffle(pairs)
    return pairs


def find_assistant_eot_position(tokenizer, messages: list[dict]) -> int:
    """Find the assistant-turn EOT position in multi-turn format.

    In the multi-turn format:
      <start_of_turn>user ... <end_of_turn>     (1st EOT)
      <start_of_turn>model ... <end_of_turn>    (2nd EOT - this is what we want)
      <start_of_turn>user ... <end_of_turn>     (3rd EOT)
      <start_of_turn>model                       (generation prompt)

    Returns the token index of the 2nd EOT.
    """
    eot_token = get_eot_token(MODEL_NAME)
    eot_id = tokenizer.convert_tokens_to_ids(eot_token)

    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")[0]
    eot_positions = (input_ids == eot_id).nonzero(as_tuple=True)[0]

    if len(eot_positions) < 2:
        raise ValueError(
            f"Expected at least 2 EOT tokens, found {len(eot_positions)}. "
            f"Token IDs shape: {input_ids.shape}"
        )

    return eot_positions[1].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-pairs", type=int, default=500)
    parser.add_argument("--resamples", type=int, default=N_RESAMPLES_PER_ORDERING)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load probe and calibrate ---
    print("Loading probe direction...")
    layer, direction = load_probe_direction(PROBE_DIR, PROBE_ID)
    assert layer == LAYER, f"Expected layer {LAYER}, got {layer}"
    print(f"  Probe: L{layer}, direction shape: {direction.shape}")

    print("Calibrating coefficients...")
    coefficients = suggest_coefficient_range(ACTIVATIONS_PATH, LAYER, multipliers=MULTIPLIERS)
    mult_to_coef = dict(zip(MULTIPLIERS, coefficients))
    mean_norm = coefficients[MULTIPLIERS.index(0.02)] / 0.02
    print(f"  Mean activation norm at L{LAYER}: {mean_norm:.0f}")
    for m, c in mult_to_coef.items():
        print(f"  mult={m:+.3f} -> coef={c:.1f}")

    # --- Load Thurstonian scores ---
    print("Loading Thurstonian scores...")
    mu_array, task_ids = load_run_utilities(RUN_DIR)
    mu_by_id = dict(zip(task_ids, mu_array.tolist()))
    print(f"  {len(task_ids)} tasks with Thurstonian scores")

    # --- Load Task objects ---
    print("Loading task objects...")
    tasks = load_filtered_tasks(
        n=len(task_ids),
        origins=ALL_ORIGINS,
        task_ids=set(task_ids),
    )
    task_by_id = {t.id: t for t in tasks}
    # Filter mu_by_id to only tasks we actually loaded
    available_ids = set(task_by_id.keys())
    mu_by_id = {k: v for k, v in mu_by_id.items() if k in available_ids}
    print(f"  Loaded {len(tasks)} task objects ({len(mu_by_id)} with scores)")

    # --- Sample pairs ---
    rng = random.Random(SEED)
    n_total = args.n_pairs
    n_borderline = n_total // 5      # 100
    n_moderate = 2 * n_total // 5    # 200
    n_decisive = n_total - n_borderline - n_moderate  # 200
    print(f"Sampling {n_total} pairs: {n_borderline} borderline, {n_moderate} moderate, {n_decisive} decisive...")

    pairs = sample_stratified_pairs(
        mu_by_id, n_borderline, n_moderate, n_decisive, rng
    )
    print(f"  Sampled {len(pairs)} pairs")

    # Save pair metadata
    pair_meta_path = OUTPUT_DIR / "pairs.json"
    pair_meta = []
    for i, (a_id, b_id, stratum) in enumerate(pairs):
        pair_meta.append({
            "pair_id": i,
            "task_a_id": a_id,
            "task_b_id": b_id,
            "mu_a": mu_by_id[a_id],
            "mu_b": mu_by_id[b_id],
            "delta_mu": abs(mu_by_id[a_id] - mu_by_id[b_id]),
            "stratum": stratum,
        })
    pair_meta_path.write_text(json.dumps(pair_meta, indent=2))
    print(f"  Saved pair metadata to {pair_meta_path}")

    # --- Resume support ---
    done_keys: set[tuple] = set()
    if args.resume and CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                rec = json.loads(line)
                done_keys.add((
                    rec["pair_id"],
                    rec["multiplier"],
                    rec["ordering"],
                    rec["resample_idx"],
                ))
        print(f"  Resuming: {len(done_keys)} trials already completed")

    # --- Load model ---
    print("Loading model...")
    hf_model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    tokenizer = hf_model.tokenizer

    # --- Setup prompt builder ---
    response_format = CompletionChoiceFormat(
        task_a_label="Task A",
        task_b_label="Task B",
    )
    measurer = RevealedPreferenceMeasurer()
    prompt_builder = MultiTurnRevealedPromptBuilder(
        prefill=PREFILL,
        measurer=measurer,
        response_format=response_format,
        template=TEMPLATE,
    )

    # --- Experiment loop ---
    total_trials = len(pairs) * 2 * len(MULTIPLIERS) * args.resamples
    remaining = total_trials - len(done_keys)
    print(f"\nStarting experiment: {total_trials} total trials, {remaining} remaining")
    print(f"  {len(pairs)} pairs x 2 orderings x {len(MULTIPLIERS)} coefficients x {args.resamples} resamples")

    checkpoint_file = open(CHECKPOINT_PATH, "a")

    with tqdm(total=remaining, desc="Generating") as pbar:
        for pair_idx, (task_a_id, task_b_id, stratum) in enumerate(pairs):
            task_a = task_by_id[task_a_id]
            task_b = task_by_id[task_b_id]
            mu_a = mu_by_id[task_a_id]
            mu_b = mu_by_id[task_b_id]

            # Determine which task has higher mu
            if mu_a >= mu_b:
                high_mu_id = task_a_id
            else:
                high_mu_id = task_b_id

            for ordering in ["canonical", "reversed"]:
                if ordering == "canonical":
                    first_task, second_task = task_a, task_b
                else:
                    first_task, second_task = task_b, task_a

                # Build prompt
                prompt = prompt_builder.build(first_task, second_task)
                messages = prompt.messages

                # Find assistant-turn EOT position
                eot_pos = find_assistant_eot_position(tokenizer, messages)

                for mult in MULTIPLIERS:
                    # Check if all resamples for this condition are done
                    all_done = all(
                        (pair_idx, mult, ordering, r) in done_keys
                        for r in range(args.resamples)
                    )
                    if all_done:
                        continue

                    coef = mult_to_coef[mult]

                    # Create hook
                    if coef == 0:
                        hook = noop_steering()
                    else:
                        tensor = torch.tensor(
                            direction * coef,
                            dtype=torch.bfloat16,
                            device=hf_model.device,
                        )
                        hook = position_selective_steering(tensor, eot_pos, eot_pos + 1)

                    # Generate resamples that aren't done yet
                    pending_resamples = [
                        r for r in range(args.resamples)
                        if (pair_idx, mult, ordering, r) not in done_keys
                    ]

                    if not pending_resamples:
                        continue

                    # Generate all pending resamples at once
                    responses = hf_model.generate_with_hook_n(
                        messages=messages,
                        layer=LAYER,
                        hook=hook,
                        n=len(pending_resamples),
                        temperature=TEMPERATURE,
                    )

                    # Create per-pair CompletionChoiceFormat with task prompts for parsing
                    pair_format = CompletionChoiceFormat(
                        task_a_label="Task A",
                        task_b_label="Task B",
                        task_a_prompt=first_task.prompt,
                        task_b_prompt=second_task.prompt,
                    )

                    for resample_idx, response in zip(pending_resamples, responses):
                        choice = pair_format.parse_sync(response)

                        # Map choice to high-mu decision
                        if choice in ("a", "b"):
                            # "a" means first_task chosen, "b" means second_task chosen
                            if ordering == "canonical":
                                chosen_id = task_a_id if choice == "a" else task_b_id
                            else:
                                chosen_id = task_b_id if choice == "a" else task_a_id
                            chose_high_mu = chosen_id == high_mu_id
                        else:
                            chose_high_mu = None  # parse_fail

                        record = {
                            "pair_id": pair_idx,
                            "task_a_id": task_a_id,
                            "task_b_id": task_b_id,
                            "stratum": stratum,
                            "mu_a": mu_a,
                            "mu_b": mu_b,
                            "high_mu_id": high_mu_id,
                            "ordering": ordering,
                            "multiplier": mult,
                            "coefficient": coef,
                            "resample_idx": resample_idx,
                            "response": response,
                            "choice": choice,
                            "chose_high_mu": chose_high_mu,
                        }
                        checkpoint_file.write(json.dumps(record) + "\n")
                        checkpoint_file.flush()
                        pbar.update(1)

    checkpoint_file.close()
    print(f"\nDone. Results saved to {CHECKPOINT_PATH}")

    # Print summary stats
    n_results = 0
    n_parse_fail = 0
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            rec = json.loads(line)
            n_results += 1
            if rec["choice"] == "parse_fail":
                n_parse_fail += 1
    print(f"Total results: {n_results}")
    print(f"Parse failures: {n_parse_fail} ({100 * n_parse_fail / max(n_results, 1):.1f}%)")


if __name__ == "__main__":
    main()
