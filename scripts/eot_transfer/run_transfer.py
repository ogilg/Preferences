"""EOT Transfer Patching — Main experiment script.

Tests whether the EOT token encodes a positional choice ("pick slot A/B")
or task identity ("do this specific task") by patching EOT activations
into prompts with different tasks.

4 conditions:
1. Control (same prompt, opposite ordering — replication of scaled experiment)
2. Swap both tasks (replace both with new tasks C, D)
3. Swap target task (replace only the task the donor was pushing toward)
4. Swap headers (same tasks, but change labels from Task A/B to Task 1/2)

Checkpoints to JSONL. Supports --resume.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import LayerHook
from src.measurement.elicitation.response_format import CompletionChoiceFormat

EXPERIMENT_DIR = Path("experiments/patching/eot_transfer")
ORDERINGS_PATH = EXPERIMENT_DIR / "selected_orderings.json"
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoint.jsonl"

TEMPLATE = (
    "You will be given two tasks. Choose one and complete it.\n"
    "{format_instruction}\n\n"
    "{label_a}:\n{task_a}\n\n"
    "{label_b}:\n{task_b}"
)

N_TRIALS = 5
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 64
MODEL_NAME = "gemma-3-27b"


def format_prompt(
    task_a_prompt: str, task_b_prompt: str,
    label_a: str = "Task A", label_b: str = "Task B",
) -> str:
    fmt = CompletionChoiceFormat(task_a_label=label_a, task_b_label=label_b)
    return TEMPLATE.format(
        format_instruction=fmt.format_instruction(),
        label_a=label_a,
        label_b=label_b,
        task_a=task_a_prompt,
        task_b=task_b_prompt,
    )


def parse_choices(
    completions: list[str],
    label_a: str = "Task A", label_b: str = "Task B",
) -> list[str]:
    fmt = CompletionChoiceFormat(task_a_label=label_a, task_b_label=label_b)
    return [fmt.parse_sync(c) for c in completions]


def get_prompt_length(model: HuggingFaceModel, messages: list[dict]) -> int:
    formatted = model.format_messages(messages, add_generation_prompt=True)
    return len(model.tokenizer(formatted, return_tensors="pt").input_ids[0])


def cache_residuals_hook(cache: dict, positions: list[int]) -> LayerHook:
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:
            for pos in positions:
                cache[pos] = resid[:, pos, :].clone()
        return resid
    return hook


def inject_residuals_hook(cache: dict, positions: list[int]) -> LayerHook:
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:
            for pos in positions:
                resid[:, pos, :] = cache[pos]
        return resid
    return hook


def run_donor_pass(
    model: HuggingFaceModel, messages: list[dict], positions: list[int],
) -> dict[int, dict[int, torch.Tensor]]:
    layer_caches = {}
    hooks = []
    for layer in range(model.n_layers):
        cache = {}
        layer_caches[layer] = cache
        hooks.append((layer, cache_residuals_hook(cache, positions)))
    model.generate_with_hooks_n(
        messages, layer_hooks=hooks, n=1, temperature=0.0, max_new_tokens=1,
    )
    return layer_caches


def run_patched_generation(
    model: HuggingFaceModel,
    recipient_messages: list[dict],
    donor_cache: dict[int, dict[int, torch.Tensor]],
    recipient_eot: list[int],
    donor_eot: list[int],
) -> list[str]:
    inject_hooks = []
    for layer in range(model.n_layers):
        mapped = {rp: donor_cache[layer][dp] for rp, dp in zip(recipient_eot, donor_eot)}
        inject_hooks.append((layer, inject_residuals_hook(mapped, recipient_eot)))
    return model.generate_with_hooks_n(
        recipient_messages, layer_hooks=inject_hooks, n=N_TRIALS,
        temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
    )


def load_completed(checkpoint_path: Path) -> set[str]:
    completed = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                rec = json.loads(line)
                completed.add(rec["trial_key"])
    return completed


def make_trial_key(ordering_idx: int, condition: str, direction: str = "") -> str:
    if direction:
        return f"{ordering_idx}_{condition}_{direction}"
    return f"{ordering_idx}_{condition}"


def run_condition(
    model: HuggingFaceModel,
    donor_cache: dict[int, dict[int, torch.Tensor]],
    donor_eot: list[int],
    recipient_task_a: str,
    recipient_task_b: str,
    label_a: str = "Task A",
    label_b: str = "Task B",
) -> dict:
    content = format_prompt(recipient_task_a, recipient_task_b, label_a, label_b)
    messages = [{"role": "user", "content": content}]
    prompt_len = get_prompt_length(model, messages)
    recipient_eot = [prompt_len - 5, prompt_len - 4]

    # Baseline (unpatched)
    baseline_completions = model.generate_n(
        messages, n=N_TRIALS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
    )
    baseline_choices = parse_choices(baseline_completions, label_a, label_b)

    # Patched
    patched_completions = run_patched_generation(
        model, messages, donor_cache, recipient_eot, donor_eot,
    )
    patched_choices = parse_choices(patched_completions, label_a, label_b)

    return {
        "prompt_len": prompt_len,
        "baseline_completions": baseline_completions,
        "baseline_choices": baseline_choices,
        "patched_completions": patched_completions,
        "patched_choices": patched_choices,
        "label_a": label_a,
        "label_b": label_b,
        "recipient_task_a": recipient_task_a[:200],
        "recipient_task_b": recipient_task_b[:200],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit orderings (0=all)")
    args = parser.parse_args()

    with open(ORDERINGS_PATH) as f:
        orderings = json.load(f)
    print(f"Loaded {len(orderings)} selected orderings")

    if args.limit > 0:
        orderings = orderings[:args.limit]
        print(f"Limited to {len(orderings)} orderings")

    completed = set()
    if args.resume:
        completed = load_completed(CHECKPOINT_PATH)
        print(f"Resuming: {len(completed)} trials already done")

    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded: {model.n_layers} layers")

    t0 = time.time()
    total_trials = 0
    skipped = 0

    for idx, ordering in enumerate(orderings):
        tid_a = ordering["task_a_id"]
        tid_b = ordering["task_b_id"]
        task_a_prompt = ordering["task_a_prompt"]
        task_b_prompt = ordering["task_b_prompt"]
        donor_slot = ordering["donor_slot"]

        # --- Donor pass (opposite ordering) ---
        # The donor prompt is the OPPOSITE ordering of the source prompt
        donor_content = format_prompt(task_b_prompt, task_a_prompt)
        donor_messages = [{"role": "user", "content": donor_content}]
        donor_len = get_prompt_length(model, donor_messages)
        donor_eot = [donor_len - 5, donor_len - 4]
        donor_cache = run_donor_pass(model, donor_messages, donor_eot)

        # === Condition 1: Control (same prompt, replication) ===
        key = make_trial_key(idx, "control")
        if key not in completed:
            result = run_condition(
                model, donor_cache, donor_eot,
                task_a_prompt, task_b_prompt,
            )
            record = {
                "trial_key": key,
                "ordering_idx": idx,
                "condition": "control",
                "source_task_a_id": tid_a,
                "source_task_b_id": tid_b,
                "donor_slot": donor_slot,
                **result,
            }
            with open(CHECKPOINT_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
            total_trials += 1
        else:
            skipped += 1

        # === Condition 2: Swap both tasks ===
        swap_c = ordering["swap_both_c_prompt"]
        swap_d = ordering["swap_both_d_prompt"]
        for dir_label, ra, rb in [("cd", swap_c, swap_d), ("dc", swap_d, swap_c)]:
            key = make_trial_key(idx, "swap_both", dir_label)
            if key not in completed:
                result = run_condition(
                    model, donor_cache, donor_eot, ra, rb,
                )
                record = {
                    "trial_key": key,
                    "ordering_idx": idx,
                    "condition": "swap_both",
                    "swap_direction": dir_label,
                    "source_task_a_id": tid_a,
                    "source_task_b_id": tid_b,
                    "donor_slot": donor_slot,
                    "recipient_c_id": ordering["swap_both_c_id"],
                    "recipient_d_id": ordering["swap_both_d_id"],
                    **result,
                }
                with open(CHECKPOINT_PATH, "a") as f:
                    f.write(json.dumps(record) + "\n")
                total_trials += 1
            else:
                skipped += 1

        # === Condition 3: Swap target task ===
        # Replace only the task in the donor's target slot
        for swap_type, swap_id_key, swap_prompt_key in [
            ("same_topic", "swap_target_same_id", "swap_target_same_prompt"),
            ("cross_topic", "swap_target_cross_id", "swap_target_cross_prompt"),
        ]:
            replacement_prompt = ordering[swap_prompt_key]
            replacement_id = ordering[swap_id_key]

            # donor_slot tells us which slot the donor was pushing toward
            # Replace the task in that slot
            if donor_slot == "b":
                ra, rb = task_a_prompt, replacement_prompt
            else:
                ra, rb = replacement_prompt, task_b_prompt

            for dir_label, final_a, final_b in [
                ("orig", ra, rb),
                ("swap", rb, ra),
            ]:
                key = make_trial_key(idx, f"swap_target_{swap_type}", dir_label)
                if key not in completed:
                    result = run_condition(
                        model, donor_cache, donor_eot, final_a, final_b,
                    )
                    record = {
                        "trial_key": key,
                        "ordering_idx": idx,
                        "condition": "swap_target",
                        "swap_type": swap_type,
                        "swap_direction": dir_label,
                        "source_task_a_id": tid_a,
                        "source_task_b_id": tid_b,
                        "donor_slot": donor_slot,
                        "replacement_task_id": replacement_id,
                        **result,
                    }
                    with open(CHECKPOINT_PATH, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    total_trials += 1
                else:
                    skipped += 1

        # === Condition 4: Swap headers ===
        key = make_trial_key(idx, "swap_headers")
        if key not in completed:
            result = run_condition(
                model, donor_cache, donor_eot,
                task_a_prompt, task_b_prompt,
                label_a="Task 1", label_b="Task 2",
            )
            record = {
                "trial_key": key,
                "ordering_idx": idx,
                "condition": "swap_headers",
                "source_task_a_id": tid_a,
                "source_task_b_id": tid_b,
                "donor_slot": donor_slot,
                **result,
            }
            with open(CHECKPOINT_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
            total_trials += 1
        else:
            skipped += 1

        elapsed = time.time() - t0
        rate = (total_trials + skipped) / elapsed if elapsed > 0 else 0
        trials_per_ordering = 8  # 1 control + 2 swap_both + 4 swap_target + 1 swap_headers
        remaining_orderings = len(orderings) - idx - 1
        eta = remaining_orderings * trials_per_ordering / rate if rate > 0 else 0

        if (idx + 1) % 10 == 0 or idx == 0:
            print(
                f"[{idx+1}/{len(orderings)}] "
                f"trials={total_trials} skipped={skipped} "
                f"({elapsed:.0f}s, {rate:.1f} trials/s, ETA {eta/3600:.1f}h)"
            )

    print(f"\nDone. Total trials: {total_trials}, skipped: {skipped}")
    print(f"Total time: {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
