"""Activation patching pilot: swap task-position activations to test causal role."""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import torch

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import noop_steering, swap_positions, swap_spans
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.steering.tokenization import find_pairwise_task_spans

EXPERIMENT_DIR = Path("experiments/patching/pilot")
TASKS_PATH = EXPERIMENT_DIR / "selected_tasks.json"
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoint.jsonl"
RESULTS_PATH = EXPERIMENT_DIR / "results.json"

TEMPLATE = (
    "You will be given two tasks. Choose one and complete it.\n"
    "{format_instruction}\n\n"
    "Task A:\n{task_a}\n\n"
    "Task B:\n{task_b}"
)

N_TRIALS = 5
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


def compute_bos_offset(model: HuggingFaceModel) -> int:
    """Check if _tokenize adds a BOS token that find_pairwise_task_spans doesn't account for."""
    test_text = "Hello world"
    # How _tokenize does it (with special tokens by default)
    ids_with_special = model.tokenizer(test_text, return_tensors="pt").input_ids
    # How find_pairwise_task_spans does it
    ids_without_special = model.tokenizer(
        test_text, return_offsets_mapping=True, add_special_tokens=False
    )["input_ids"]
    return len(ids_with_special[0]) - len(ids_without_special)


def find_spans_with_offset(
    model: HuggingFaceModel,
    formatted_chat_prompt: str,
    task_a_text: str,
    task_b_text: str,
    bos_offset: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find task spans in the formatted prompt and add BOS offset if needed."""
    a_span, b_span = find_pairwise_task_spans(
        model.tokenizer, formatted_chat_prompt, task_a_text, task_b_text
    )
    return (
        (a_span[0] + bos_offset, a_span[1] + bos_offset),
        (b_span[0] + bos_offset, b_span[1] + bos_offset),
    )


def verify_spans(
    model: HuggingFaceModel,
    formatted_chat_prompt: str,
    a_span: tuple[int, int],
    b_span: tuple[int, int],
    task_a_text: str,
    task_b_text: str,
) -> None:
    """Verify that token spans actually correspond to the task text."""
    input_ids = model.tokenizer(formatted_chat_prompt, return_tensors="pt").input_ids[0]
    a_tokens = model.tokenizer.decode(input_ids[a_span[0]:a_span[1]])
    b_tokens = model.tokenizer.decode(input_ids[b_span[0]:b_span[1]])
    # Check that decoded tokens contain the task text (may have minor whitespace diffs)
    if task_a_text[:30] not in a_tokens[:50]:
        raise ValueError(
            f"Span A verification failed.\nExpected start: {task_a_text[:30]!r}\nGot: {a_tokens[:50]!r}"
        )
    if task_b_text[:30] not in b_tokens[:50]:
        raise ValueError(
            f"Span B verification failed.\nExpected start: {task_b_text[:30]!r}\nGot: {b_tokens[:50]!r}"
        )


def parse_choices(completions: list[str], task_a_prompt: str, task_b_prompt: str) -> list[str]:
    """Parse choices from completions. Returns list of 'a', 'b', or 'parse_fail'."""
    fmt = CompletionChoiceFormat(
        task_a_prompt=task_a_prompt, task_b_prompt=task_b_prompt
    )
    return [fmt.parse_sync(c) for c in completions]


def make_swap_hooks(
    n_layers: int,
    a_span: tuple[int, int],
    b_span: tuple[int, int],
    condition: str,
) -> list[tuple[int, "LayerHook"]]:
    """Create hooks for all layers based on condition."""
    hooks = []
    for layer in range(n_layers):
        if condition == "last_token_swap":
            hook = swap_positions(a_span[1] - 1, b_span[1] - 1)
        elif condition == "span_swap":
            hook = swap_spans(a_span[0], a_span[1], b_span[0], b_span[1])
        else:
            raise ValueError(f"Unknown condition: {condition}")
        hooks.append((layer, hook))
    return hooks


def load_checkpoint() -> set[str]:
    """Load completed pair keys from checkpoint."""
    completed = set()
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                record = json.loads(line)
                completed.add(record["key"])
    return completed


def append_checkpoint(key: str, result: dict) -> None:
    """Append a result to the checkpoint file."""
    record = {"key": key, **result}
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_pair(
    model: HuggingFaceModel,
    task_a: dict,
    task_b: dict,
    ordering: str,
    bos_offset: int,
    verify: bool = False,
) -> dict:
    """Run all conditions for one pair in one ordering."""
    if ordering == "AB":
        prompt_a, prompt_b = task_a["prompt"], task_b["prompt"]
        id_a, id_b = task_a["task_id"], task_b["task_id"]
    else:
        prompt_a, prompt_b = task_b["prompt"], task_a["prompt"]
        id_a, id_b = task_b["task_id"], task_a["task_id"]

    user_content = format_prompt(prompt_a, prompt_b)
    messages = [{"role": "user", "content": user_content}]

    # Get formatted chat prompt for tokenization
    formatted_chat = model.format_messages(messages, add_generation_prompt=True)

    # Find task spans
    a_span, b_span = find_spans_with_offset(
        model, formatted_chat, prompt_a, prompt_b, bos_offset
    )

    if verify:
        verify_spans(model, formatted_chat, a_span, b_span, prompt_a, prompt_b)
        print(f"  Span A: tokens {a_span[0]}-{a_span[1]} ({a_span[1]-a_span[0]} tokens)")
        print(f"  Span B: tokens {b_span[0]}-{b_span[1]} ({b_span[1]-b_span[0]} tokens)")

    result = {
        "task_a_id": id_a,
        "task_b_id": id_b,
        "ordering": ordering,
        "a_span": list(a_span),
        "b_span": list(b_span),
        "conditions": {},
    }

    # Baseline (no hooks)
    t0 = time.time()
    baseline_completions = model.generate_n(
        messages, n=N_TRIALS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS
    )
    baseline_choices = parse_choices(baseline_completions, prompt_a, prompt_b)
    result["conditions"]["baseline"] = {
        "completions": baseline_completions,
        "choices": baseline_choices,
        "time_s": round(time.time() - t0, 2),
    }

    # Last-token swap
    t0 = time.time()
    hooks = make_swap_hooks(model.n_layers, a_span, b_span, "last_token_swap")
    lt_completions = model.generate_with_hooks_n(
        messages, layer_hooks=hooks, n=N_TRIALS, temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    lt_choices = parse_choices(lt_completions, prompt_a, prompt_b)
    result["conditions"]["last_token_swap"] = {
        "completions": lt_completions,
        "choices": lt_choices,
        "time_s": round(time.time() - t0, 2),
    }

    # Span swap
    t0 = time.time()
    hooks = make_swap_hooks(model.n_layers, a_span, b_span, "span_swap")
    span_completions = model.generate_with_hooks_n(
        messages, layer_hooks=hooks, n=N_TRIALS, temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    span_choices = parse_choices(span_completions, prompt_a, prompt_b)
    result["conditions"]["span_swap"] = {
        "completions": span_completions,
        "choices": span_choices,
        "time_s": round(time.time() - t0, 2),
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", type=int, default=0, help="Run only N pairs for validation")
    parser.add_argument("--verify", action="store_true", help="Verify token spans on first pair")
    args = parser.parse_args()

    # Load tasks
    with open(TASKS_PATH) as f:
        tasks = json.load(f)
    task_by_id = {t["task_id"]: t for t in tasks}

    # Generate all pairs
    pairs = list(combinations(tasks, 2))
    print(f"Total pairs: {len(pairs)}, orderings: {len(pairs) * 2}")

    if args.pilot:
        pairs = pairs[:args.pilot]
        print(f"Pilot mode: running {args.pilot} pairs ({args.pilot * 2} orderings)")

    # Load checkpoint
    completed = load_checkpoint() if args.resume else set()
    if completed:
        print(f"Resuming: {len(completed)} pair-orderings already done")

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded. Layers: {model.n_layers}, hidden dim: {model.hidden_dim}")

    # Compute BOS offset
    bos_offset = compute_bos_offset(model)
    print(f"BOS token offset: {bos_offset}")

    all_results = []
    total = len(pairs) * 2
    done = 0

    for pair_idx, (task_a, task_b) in enumerate(pairs):
        for ordering in ["AB", "BA"]:
            key = f"{task_a['task_id']}_{task_b['task_id']}_{ordering}"
            if key in completed:
                done += 1
                continue

            print(f"\n[{done+1}/{total}] {task_a['task_id']} vs {task_b['task_id']} ({ordering})")

            verify = args.verify and done == 0
            result = run_pair(model, task_a, task_b, ordering, bos_offset, verify=verify)
            all_results.append(result)

            # Print summary
            for cond_name, cond_data in result["conditions"].items():
                choices = cond_data["choices"]
                n_a = choices.count("a")
                n_b = choices.count("b")
                n_fail = choices.count("parse_fail")
                print(f"  {cond_name}: A={n_a} B={n_b} fail={n_fail} ({cond_data['time_s']}s)")

            append_checkpoint(key, result)
            done += 1

    # Save full results
    # Combine with any existing checkpoint data
    if args.resume and CHECKPOINT_PATH.exists():
        all_from_checkpoint = []
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                record = json.loads(line)
                record.pop("key", None)
                all_from_checkpoint.append(record)
        all_results = all_from_checkpoint

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH} ({len(all_results)} pair-orderings)")


if __name__ == "__main__":
    main()
