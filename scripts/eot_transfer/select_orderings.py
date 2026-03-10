"""Select 200 deterministic flipping orderings from Phase 1 for the transfer experiment.

For each selected ordering, also pick replacement tasks for swap conditions.
"""

import json
import random
from collections import Counter
from pathlib import Path

PHASE1_PATH = Path("experiments/patching/eot_scaled/phase1_checkpoint.jsonl")
TASKS_PATH = Path("experiments/patching/eot_scaled/selected_tasks.json")
TOPICS_PATH = Path("data/topics/topics.json")
OUTPUT_PATH = Path("experiments/patching/eot_transfer/selected_orderings.json")

N_SELECT = 200
SEED = 42


def load_phase1() -> list[dict]:
    records = []
    with open(PHASE1_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def is_deterministic(choices: list[str]) -> bool:
    """Baseline is deterministic if max P >= 0.9 (9+ out of 10 same choice)."""
    counts = Counter(c for c in choices if c in ("a", "b"))
    if not counts:
        return False
    return max(counts.values()) / sum(counts.values()) >= 0.9


def get_dominant_choice(choices: list[str]) -> str:
    counts = Counter(c for c in choices if c in ("a", "b"))
    return counts.most_common(1)[0][0]


def did_flip(baseline_choices: list[str], patched_choices: list[str]) -> bool:
    """Patching flipped the choice if patched dominant != baseline dominant."""
    baseline_valid = [c for c in baseline_choices if c in ("a", "b")]
    patched_valid = [c for c in patched_choices if c in ("a", "b")]
    if len(baseline_valid) < 5 or len(patched_valid) < 5:
        return False
    baseline_dom = Counter(baseline_valid).most_common(1)[0][0]
    patched_dom = Counter(patched_valid).most_common(1)[0][0]
    return baseline_dom != patched_dom


def get_topic(task_id: str, topics: dict) -> str:
    if task_id not in topics:
        return "unknown"
    entry = topics[task_id]
    # Get primary topic from first available model
    for model_data in entry.values():
        return model_data["primary"]
    return "unknown"


def main():
    records = load_phase1()
    print(f"Loaded {len(records)} Phase 1 records")

    with open(TASKS_PATH) as f:
        tasks = json.load(f)
    task_map = {t["task_id"]: t for t in tasks}
    print(f"Loaded {len(tasks)} tasks")

    with open(TOPICS_PATH) as f:
        topics = json.load(f)

    # Find deterministic flipping orderings
    flipping = []
    for rec in records:
        if not is_deterministic(rec["baseline_choices"]):
            continue
        if not did_flip(rec["baseline_choices"], rec["patched_choices"]):
            continue
        flipping.append(rec)

    print(f"Deterministic flipping orderings: {len(flipping)}")

    # Sample 200
    random.seed(SEED)
    selected = random.sample(flipping, min(N_SELECT, len(flipping)))
    print(f"Selected: {len(selected)}")

    # For each selected ordering, pick replacement tasks
    task_ids = [t["task_id"] for t in tasks]
    task_topics = {tid: get_topic(tid, topics) for tid in task_ids}

    output = []
    for rec in selected:
        tid_a = rec["task_a_id"]
        tid_b = rec["task_b_id"]
        baseline_dom = get_dominant_choice(rec["baseline_choices"])

        # Donor was opposite ordering, which pushed toward the non-dominant slot
        # If baseline picks "a", donor (opposite) pushes toward "b"
        donor_slot = "b" if baseline_dom == "a" else "a"

        # The donor was trying to make the model pick a different slot
        # After patching, model picks donor_slot
        # The task in donor_slot in the recipient prompt is:
        donor_task_id = tid_b if donor_slot == "b" else tid_a

        topic_a = task_topics.get(tid_a, "unknown")
        topic_b = task_topics.get(tid_b, "unknown")
        donor_topic = task_topics.get(donor_task_id, "unknown")

        # Pick replacement tasks
        available = [t for t in task_ids if t != tid_a and t != tid_b]

        # For condition 2 (swap both): pick C and D
        # Try to get a same-topic and cross-topic pair
        same_topic_pool = [t for t in available if task_topics.get(t) == donor_topic]
        cross_topic_pool = [t for t in available if task_topics.get(t) != donor_topic]

        # Pick C (replacement for slot A task) and D (replacement for slot B task)
        random.shuffle(available)
        task_c = available[0]
        task_d = available[1]

        # For condition 3 (swap target): replace only the donor's target task
        # Pick a same-topic replacement and a cross-topic replacement
        swap_target_same = same_topic_pool[0] if same_topic_pool else available[2]
        swap_target_cross = cross_topic_pool[0] if cross_topic_pool else available[3]

        entry = {
            "task_a_id": tid_a,
            "task_b_id": tid_b,
            "direction": rec["direction"],
            "baseline_dominant": baseline_dom,
            "donor_slot": donor_slot,
            "donor_task_id": donor_task_id,
            "task_a_prompt": task_map[tid_a]["prompt"],
            "task_b_prompt": task_map[tid_b]["prompt"],
            "task_a_topic": topic_a,
            "task_b_topic": topic_b,
            "task_a_mu": task_map[tid_a]["mu"],
            "task_b_mu": task_map[tid_b]["mu"],
            # Replacement tasks
            "swap_both_c_id": task_c,
            "swap_both_c_prompt": task_map[task_c]["prompt"],
            "swap_both_d_id": task_d,
            "swap_both_d_prompt": task_map[task_d]["prompt"],
            "swap_target_same_id": swap_target_same,
            "swap_target_same_prompt": task_map[swap_target_same]["prompt"],
            "swap_target_cross_id": swap_target_cross,
            "swap_target_cross_prompt": task_map[swap_target_cross]["prompt"],
        }
        output.append(entry)

    # Stats
    donor_slots = Counter(e["donor_slot"] for e in output)
    print(f"Donor slot distribution: {dict(donor_slots)}")

    has_harmful = sum(1 for e in output if any(
        e[k].startswith("bailbench") or e[k].startswith("stresstest")
        for k in ["task_a_id", "task_b_id"]
    ))
    print(f"Orderings with harmful tasks: {has_harmful}/{len(output)}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
