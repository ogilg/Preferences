"""Full-block swap: swap labels + content to test if label tokens carry choice info."""

import json
import time
from itertools import combinations
from pathlib import Path

import torch

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import swap_spans
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.steering.tokenization import find_text_span

EXPERIMENT_DIR = Path("experiments/patching/pilot")
TASKS_PATH = EXPERIMENT_DIR / "selected_tasks.json"
RESULTS_PATH = EXPERIMENT_DIR / "full_block_results.json"
CHECKPOINT_PATH = EXPERIMENT_DIR / "full_block_checkpoint.jsonl"

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
    test_text = "Hello world"
    ids_with_special = model.tokenizer(test_text, return_tensors="pt").input_ids
    ids_without_special = model.tokenizer(
        test_text, return_offsets_mapping=True, add_special_tokens=False
    )["input_ids"]
    return len(ids_with_special[0]) - len(ids_without_special)


def find_full_block_spans(
    model: HuggingFaceModel,
    formatted_chat: str,
    task_a_text: str,
    task_b_text: str,
    bos_offset: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find spans including labels: 'Task A:\\n<content>' and 'Task B:\\n<content>'."""
    # Block A: from "Task A:\n" to end of task_a_text
    block_a_text = f"Task A:\n{task_a_text}"
    a_span = find_text_span(model.tokenizer, formatted_chat, block_a_text)
    a_span = (a_span[0] + bos_offset, a_span[1] + bos_offset)

    # Block B: from "Task B:\n" to end of task_b_text
    block_b_text = f"Task B:\n{task_b_text}"
    b_span = find_text_span(model.tokenizer, formatted_chat, block_b_text)
    b_span = (b_span[0] + bos_offset, b_span[1] + bos_offset)

    return a_span, b_span


def parse_choices(completions: list[str], task_a_prompt: str, task_b_prompt: str) -> list[str]:
    fmt = CompletionChoiceFormat(
        task_a_prompt=task_a_prompt, task_b_prompt=task_b_prompt
    )
    return [fmt.parse_sync(c) for c in completions]


def load_checkpoint() -> set[str]:
    completed = set()
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                record = json.loads(line)
                completed.add(record["key"])
    return completed


def main():
    with open(TASKS_PATH) as f:
        tasks = json.load(f)

    pairs = list(combinations(tasks, 2))
    print(f"Total pairs: {len(pairs)}, orderings: {len(pairs) * 2}")

    completed = load_checkpoint()
    if completed:
        print(f"Resuming: {len(completed)} already done")

    print(f"Loading model: {MODEL_NAME}")
    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Layers: {model.n_layers}")

    bos_offset = compute_bos_offset(model)
    print(f"BOS offset: {bos_offset}")

    all_results = []
    total = len(pairs) * 2
    done = 0

    for task_a, task_b in pairs:
        for ordering in ["AB", "BA"]:
            key = f"{task_a['task_id']}_{task_b['task_id']}_{ordering}"
            if key in completed:
                done += 1
                continue

            if ordering == "AB":
                prompt_a, prompt_b = task_a["prompt"], task_b["prompt"]
                id_a, id_b = task_a["task_id"], task_b["task_id"]
            else:
                prompt_a, prompt_b = task_b["prompt"], task_a["prompt"]
                id_a, id_b = task_b["task_id"], task_a["task_id"]

            user_content = format_prompt(prompt_a, prompt_b)
            messages = [{"role": "user", "content": user_content}]
            formatted_chat = model.format_messages(messages, add_generation_prompt=True)

            a_span, b_span = find_full_block_spans(
                model, formatted_chat, prompt_a, prompt_b, bos_offset
            )

            if done == 0:
                input_ids = model.tokenizer(formatted_chat, return_tensors="pt").input_ids[0]
                print(f"  Block A [{a_span[0]}:{a_span[1]}]: {model.tokenizer.decode(input_ids[a_span[0]:a_span[0]+10])!r}...")
                print(f"  Block B [{b_span[0]}:{b_span[1]}]: {model.tokenizer.decode(input_ids[b_span[0]:b_span[0]+10])!r}...")

            print(f"[{done+1}/{total}] {id_a} vs {id_b} ({ordering})")

            # Baseline
            t0 = time.time()
            baseline_completions = model.generate_n(
                messages, n=N_TRIALS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS
            )
            baseline_choices = parse_choices(baseline_completions, prompt_a, prompt_b)
            baseline_time = round(time.time() - t0, 2)

            # Full block swap
            t0 = time.time()
            hooks = [
                (layer, swap_spans(a_span[0], a_span[1], b_span[0], b_span[1]))
                for layer in range(model.n_layers)
            ]
            swap_completions = model.generate_with_hooks_n(
                messages, layer_hooks=hooks, n=N_TRIALS, temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            swap_choices = parse_choices(swap_completions, prompt_a, prompt_b)
            swap_time = round(time.time() - t0, 2)

            result = {
                "task_a_id": id_a,
                "task_b_id": id_b,
                "ordering": ordering,
                "a_span": list(a_span),
                "b_span": list(b_span),
                "conditions": {
                    "baseline": {
                        "completions": baseline_completions,
                        "choices": baseline_choices,
                        "time_s": baseline_time,
                    },
                    "full_block_swap": {
                        "completions": swap_completions,
                        "choices": swap_choices,
                        "time_s": swap_time,
                    },
                },
            }
            all_results.append(result)

            n_a_base = baseline_choices.count("a")
            n_a_swap = swap_choices.count("a")
            print(f"  baseline: A={n_a_base} B={5-n_a_base} ({baseline_time}s)")
            print(f"  full_block_swap: A={n_a_swap} B={5-n_a_swap} ({swap_time}s)")

            record = {"key": key, **result}
            with open(CHECKPOINT_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
            done += 1

    if CHECKPOINT_PATH.exists():
        all_results = []
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                record = json.loads(line)
                record.pop("key", None)
                all_results.append(record)

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
