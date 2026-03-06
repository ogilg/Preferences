"""Phase 1: Baseline + all-layer EOT patching for all 4950 pairs x 2 orderings.

For each ordering:
1. Baseline — 10 trials at temperature 1.0, max_new_tokens=16
2. All-layer EOT patch — patch <end_of_turn> + \n residuals (2 tokens) from
   the opposite ordering's donor pass at all 62 layers, 10 trials at temperature 1.0

Checkpoints to JSONL (one line per ordering). Supports --resume.
"""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import LayerHook
from src.measurement.elicitation.response_format import CompletionChoiceFormat

EXPERIMENT_DIR = Path("experiments/patching/eot_scaled")
TASKS_PATH = EXPERIMENT_DIR / "selected_tasks.json"
CHECKPOINT_PATH = EXPERIMENT_DIR / "phase1_checkpoint.jsonl"

TEMPLATE = (
    "You will be given two tasks. Choose one and complete it.\n"
    "{format_instruction}\n\n"
    "Task A:\n{task_a}\n\n"
    "Task B:\n{task_b}"
)

N_TRIALS = 10
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 16
MODEL_NAME = "gemma-3-27b"


def format_prompt(task_a_prompt: str, task_b_prompt: str) -> str:
    fmt = CompletionChoiceFormat()
    return TEMPLATE.format(
        format_instruction=fmt.format_instruction(),
        task_a=task_a_prompt,
        task_b=task_b_prompt,
    )


def parse_choices(completions: list[str], task_a_prompt: str, task_b_prompt: str) -> list[str]:
    fmt = CompletionChoiceFormat(task_a_prompt=task_a_prompt, task_b_prompt=task_b_prompt)
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


def run_donor_pass(model: HuggingFaceModel, messages: list[dict], positions: list[int]) -> dict[int, dict[int, torch.Tensor]]:
    layer_caches = {}
    hooks = []
    for layer in range(model.n_layers):
        cache = {}
        layer_caches[layer] = cache
        hooks.append((layer, cache_residuals_hook(cache, positions)))
    model.generate_with_hooks_n(messages, layer_hooks=hooks, n=1, temperature=0.0, max_new_tokens=1)
    return layer_caches


def generate_all_orderings(tasks: list[dict]) -> list[dict]:
    """Generate all orderings: for each pair, AB and BA."""
    orderings = []
    for task_a, task_b in combinations(tasks, 2):
        orderings.append({
            "task_a_id": task_a["task_id"],
            "task_b_id": task_b["task_id"],
            "direction": "ab",
            "pos_a_prompt": task_a["prompt"],
            "pos_b_prompt": task_b["prompt"],
        })
        orderings.append({
            "task_a_id": task_a["task_id"],
            "task_b_id": task_b["task_id"],
            "direction": "ba",
            "pos_a_prompt": task_b["prompt"],
            "pos_b_prompt": task_a["prompt"],
        })
    return orderings


def load_completed(checkpoint_path: Path) -> set[str]:
    completed = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                rec = json.loads(line)
                key = f"{rec['task_a_id']}_{rec['task_b_id']}_{rec['direction']}"
                completed.add(key)
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(TASKS_PATH) as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks")

    orderings = generate_all_orderings(tasks)
    print(f"Total orderings: {len(orderings)}")

    completed = set()
    if args.resume:
        completed = load_completed(CHECKPOINT_PATH)
        print(f"Resuming: {len(completed)} already done")

    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded: {model.n_layers} layers")

    remaining = [
        o for o in orderings
        if f"{o['task_a_id']}_{o['task_b_id']}_{o['direction']}" not in completed
    ]
    print(f"Remaining: {len(remaining)}")

    t0 = time.time()
    for idx, ordering in enumerate(remaining):
        pos_a_prompt = ordering["pos_a_prompt"]
        pos_b_prompt = ordering["pos_b_prompt"]

        # Recipient (this ordering)
        content = format_prompt(pos_a_prompt, pos_b_prompt)
        messages = [{"role": "user", "content": content}]
        prompt_len = get_prompt_length(model, messages)

        # Donor (opposite ordering)
        donor_content = format_prompt(pos_b_prompt, pos_a_prompt)
        donor_messages = [{"role": "user", "content": donor_content}]
        donor_len = get_prompt_length(model, donor_messages)

        # EOT positions: <end_of_turn> and \n (2 tokens, at -5 and -4 from end)
        recipient_eot = [prompt_len - 5, prompt_len - 4]
        donor_eot = [donor_len - 5, donor_len - 4]

        # 1. Baseline
        baseline_completions = model.generate_n(
            messages, n=N_TRIALS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS
        )
        baseline_choices = parse_choices(baseline_completions, pos_a_prompt, pos_b_prompt)

        # 2. Donor pass
        donor_cache = run_donor_pass(model, donor_messages, donor_eot)

        # 3. Patched generation
        inject_hooks = []
        for layer in range(model.n_layers):
            mapped = {rp: donor_cache[layer][dp] for rp, dp in zip(recipient_eot, donor_eot)}
            inject_hooks.append((layer, inject_residuals_hook(mapped, recipient_eot)))

        patched_completions = model.generate_with_hooks_n(
            messages, layer_hooks=inject_hooks, n=N_TRIALS,
            temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
        )
        patched_choices = parse_choices(patched_completions, pos_a_prompt, pos_b_prompt)

        # Record
        record = {
            "task_a_id": ordering["task_a_id"],
            "task_b_id": ordering["task_b_id"],
            "direction": ordering["direction"],
            "prompt_len": prompt_len,
            "baseline_choices": baseline_choices,
            "patched_choices": patched_choices,
        }

        with open(CHECKPOINT_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (len(remaining) - idx - 1) / rate if rate > 0 else 0

        if (idx + 1) % 50 == 0 or idx == 0:
            n_base_a = baseline_choices.count("a")
            n_patch_a = patched_choices.count("a")
            print(
                f"[{idx+1}/{len(remaining)}] "
                f"{ordering['task_a_id'][:15]}v{ordering['task_b_id'][:15]} "
                f"({ordering['direction']}) "
                f"base={n_base_a}/{N_TRIALS}A  patch={n_patch_a}/{N_TRIALS}A  "
                f"({elapsed:.0f}s, {rate:.1f}/s, ETA {eta/3600:.1f}h)"
            )

    print(f"\nPhase 1 complete. Total time: {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
