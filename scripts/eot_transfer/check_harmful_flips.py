"""Find examples where patching pushed the model toward harmful tasks."""

import json
from pathlib import Path

FLIP_PATH = Path("experiments/patching/eot_scaled/flip_classification_v2.json")

with open(FLIP_PATH) as f:
    flips = json.load(f)

harmful_prefixes = ("bailbench", "stresstest")


def is_harmful(task_id: str) -> bool:
    return any(task_id.startswith(p) for p in harmful_prefixes)


toward_harm = []
away_from_harm = []
refusals = []

for flip in flips:
    a_id = flip["task_a_id"]
    b_id = flip["task_b_id"]
    a_harmful = is_harmful(a_id)
    b_harmful = is_harmful(b_id)

    if not (a_harmful or b_harmful):
        continue  # neither harmful
    if a_harmful and b_harmful:
        continue  # both harmful, skip

    # Which task did model execute after patching?
    patched_exec = flip["patched_executed_task"]  # "a" or "b"
    baseline_exec = flip["baseline_executed_task"]
    is_refusal = flip["patched_is_refusal"]

    if is_refusal:
        refusals.append(flip)
        continue

    # Did patching push toward or away from the harmful task?
    harmful_slot = "a" if a_harmful else "b"
    if patched_exec == harmful_slot and baseline_exec != harmful_slot:
        toward_harm.append(flip)
    elif patched_exec != harmful_slot and baseline_exec == harmful_slot:
        away_from_harm.append(flip)

print(f"Toward harm (genuine execution): {len(toward_harm)}")
print(f"Away from harm: {len(away_from_harm)}")
print(f"Refusals: {len(refusals)}")

print("\n=== TOWARD HARM: model executes harmful task after patching ===\n")
for flip in toward_harm:
    a_id = flip["task_a_id"]
    b_id = flip["task_b_id"]
    harmful_slot = "a" if is_harmful(a_id) else "b"
    harmful_id = a_id if harmful_slot == "a" else b_id
    safe_id = b_id if harmful_slot == "a" else a_id

    print(f"  Harmful: {harmful_id}")
    print(f"  Safe: {safe_id}")
    print(f"  Baseline executed: {flip['baseline_executed_task']} | Patched executed: {flip['patched_executed_task']}")
    print(f"  Patched reasoning: {flip['patched_reasoning'][:250]}")
    print()

print("\n=== REFUSALS ===\n")
for flip in refusals:
    a_id = flip["task_a_id"]
    b_id = flip["task_b_id"]
    print(f"  {a_id} vs {b_id}")
    print(f"  Patched reasoning: {flip['patched_reasoning'][:250]}")
    print()
