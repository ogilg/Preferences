"""EOT Label Swap experiment.

Tests whether the EOT token's choice signal follows position/label or content
when task positions are swapped in the recipient prompt.

Donor = SOURCE ordering (model's natural preference arrangement).
Recipient = REVERSED ordering (tasks in opposite positions).

If position-following: patching pushes toward donor's slot -> flip from baseline
If content-following: patching pushes toward preferred task -> no flip

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

EXPERIMENT_DIR = Path("experiments/patching/eot_transfer/label_swap")
ORDERINGS_PATH = Path("experiments/patching/eot_transfer/selected_orderings.json")
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
        baseline_dominant = ordering["baseline_dominant"]

        trial_key = f"{idx}_label_swap"

        if trial_key in completed:
            skipped += 1
            if (idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"[{idx+1}/{len(orderings)}] skipped={skipped} ({elapsed:.0f}s)")
            continue

        # --- Donor pass: SOURCE ordering ---
        # The model naturally picks baseline_dominant slot in this arrangement
        donor_content = format_prompt(task_a_prompt, task_b_prompt)
        donor_messages = [{"role": "user", "content": donor_content}]
        donor_len = get_prompt_length(model, donor_messages)
        donor_eot = [donor_len - 5, donor_len - 4]
        donor_cache = run_donor_pass(model, donor_messages, donor_eot)

        # --- Label swap recipient: REVERSED ordering ---
        # Tasks in opposite positions from source
        recipient_content = format_prompt(task_b_prompt, task_a_prompt)
        recipient_messages = [{"role": "user", "content": recipient_content}]
        recipient_len = get_prompt_length(model, recipient_messages)
        recipient_eot = [recipient_len - 5, recipient_len - 4]

        # Baseline (unpatched) on label swap recipient
        baseline_completions = model.generate_n(
            recipient_messages, n=N_TRIALS,
            temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
        )
        baseline_choices = parse_choices(baseline_completions)

        # Patched
        patched_completions = run_patched_generation(
            model, recipient_messages, donor_cache,
            recipient_eot, donor_eot,
        )
        patched_choices = parse_choices(patched_completions)

        record = {
            "trial_key": trial_key,
            "ordering_idx": idx,
            "condition": "label_swap",
            "source_task_a_id": tid_a,
            "source_task_b_id": tid_b,
            "baseline_dominant": baseline_dominant,
            # In source ordering: task_a in A, task_b in B
            # In label_swap recipient: task_b in A, task_a in B
            "recipient_task_a": task_b_prompt[:200],
            "recipient_task_b": task_a_prompt[:200],
            "donor_prompt_len": donor_len,
            "recipient_prompt_len": recipient_len,
            "baseline_completions": baseline_completions,
            "baseline_choices": baseline_choices,
            "patched_completions": patched_completions,
            "patched_choices": patched_choices,
        }
        with open(CHECKPOINT_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        total_trials += 1

        elapsed = time.time() - t0
        rate = (total_trials + skipped) / elapsed if elapsed > 0 else 0
        remaining = len(orderings) - idx - 1
        eta = remaining / rate if rate > 0 else 0

        if (idx + 1) % 10 == 0 or idx == 0:
            # Quick stats on this trial
            b_valid = [c for c in baseline_choices if c in ("a", "b")]
            p_valid = [c for c in patched_choices if c in ("a", "b")]
            b_summary = f"baseline={''.join(b_valid)}" if b_valid else "baseline=none"
            p_summary = f"patched={''.join(p_valid)}" if p_valid else "patched=none"
            print(
                f"[{idx+1}/{len(orderings)}] "
                f"trials={total_trials} skipped={skipped} "
                f"({elapsed:.0f}s, ETA {eta/60:.0f}m) "
                f"{b_summary} {p_summary}"
            )

    print(f"\nDone. Total trials: {total_trials}, skipped: {skipped}")
    print(f"Total time: {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
