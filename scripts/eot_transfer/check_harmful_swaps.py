"""Check: when patching swaps in a harmful task, does the model execute it?"""

import json
from collections import Counter, defaultdict
from pathlib import Path

CHECKPOINT_PATH = Path("experiments/patching/eot_transfer/checkpoint.jsonl")
JUDGE_PATH = Path("experiments/patching/eot_transfer/judge_results.json")

records = []
with open(CHECKPOINT_PATH) as f:
    for line in f:
        records.append(json.loads(line))

judge_results = {}
with open(JUDGE_PATH) as f:
    for r in json.load(f):
        key = f"{r['trial_key']}_{r['phase']}_{r['trial_idx']}"
        judge_results[key] = r

harmful_prefixes = ("bailbench", "stresstest")


def is_harmful(task_id: str) -> bool:
    return any(task_id.startswith(p) for p in harmful_prefixes)


def majority_choice(choices: list[str]) -> str | None:
    counts = Counter(c for c in choices if c in ("a", "b"))
    if not counts:
        return None
    (top, top_n), *rest = counts.most_common()
    if rest and rest[0][1] == top_n:
        return None
    return top


# For swap_both: check if recipient C or D is harmful
# For swap_target: check if replacement task is harmful
print("=== SWAP_BOTH: cases where recipient has harmful task ===\n")

swap_both_harmful = []
for rec in records:
    if rec["condition"] != "swap_both":
        continue
    c_id = rec.get("recipient_c_id", "")
    d_id = rec.get("recipient_d_id", "")
    if not (is_harmful(c_id) or is_harmful(d_id)):
        continue

    baseline_maj = majority_choice(rec["baseline_choices"])
    patched_maj = majority_choice(rec["patched_choices"])
    donor_slot = rec["donor_slot"]
    flipped = baseline_maj != patched_maj if baseline_maj and patched_maj else None

    # Which slot has the harmful task?
    # recipient_task_a is slot a, recipient_task_b is slot b
    harmful_in_a = is_harmful(c_id) if rec.get("swap_direction") in ("cd", None) else is_harmful(d_id)

    swap_both_harmful.append({
        "trial_key": rec["trial_key"],
        "c_id": c_id,
        "d_id": d_id,
        "donor_slot": donor_slot,
        "baseline": baseline_maj,
        "patched": patched_maj,
        "flipped": flipped,
    })

print(f"Total swap_both orderings with harmful recipient: {len(swap_both_harmful)}")
flipped_harmful = [r for r in swap_both_harmful if r["flipped"]]
print(f"Of those, flipped: {len(flipped_harmful)}")

# Show some examples of flipped orderings with harmful tasks
print("\nExamples of flipped orderings (harmful in recipient):")
for r in flipped_harmful[:5]:
    print(f"  {r['trial_key']}: c={r['c_id']}, d={r['d_id']}, "
          f"baseline={r['baseline']}, patched={r['patched']}")

print("\n\n=== SWAP_TARGET: cases where replacement is harmful ===\n")

for swap_type in ["same_topic", "cross_topic"]:
    condition_filter = f"swap_target_{swap_type}"
    harmful_replacement = []

    for rec in records:
        cond = rec["condition"]
        st = rec.get("swap_type", "")
        if f"{cond}_{st}" != condition_filter:
            continue
        repl_id = rec.get("replacement_task_id", "")
        if not is_harmful(repl_id):
            continue

        baseline_maj = majority_choice(rec["baseline_choices"])
        patched_maj = majority_choice(rec["patched_choices"])
        flipped = baseline_maj != patched_maj if baseline_maj and patched_maj else None

        harmful_replacement.append({
            "trial_key": rec["trial_key"],
            "source_a": rec["source_task_a_id"],
            "source_b": rec["source_task_b_id"],
            "replacement": repl_id,
            "donor_slot": rec["donor_slot"],
            "baseline": baseline_maj,
            "patched": patched_maj,
            "flipped": flipped,
        })

    print(f"{condition_filter}: {len(harmful_replacement)} orderings with harmful replacement")
    flipped_h = [r for r in harmful_replacement if r["flipped"]]
    print(f"  Flipped: {len(flipped_h)}/{len(harmful_replacement)}")

    # Check patched completions for actual harmful content
    if flipped_h:
        print(f"\n  Examples of flipped (harmful replacement):")
        for r in flipped_h[:3]:
            print(f"    {r['trial_key']}: replacement={r['replacement']}")


# Now the real question: in flipped cases, did the model EXECUTE the harmful task?
# Look at patched completions
print("\n\n=== Detailed: did the model execute harmful content after patching? ===\n")

count = 0
for rec in records:
    cond = rec["condition"]
    if cond not in ("swap_both", "swap_target"):
        continue

    # Find if any recipient task is harmful
    c_id = rec.get("recipient_c_id", "")
    d_id = rec.get("recipient_d_id", "")
    repl_id = rec.get("replacement_task_id", "")

    any_harmful = is_harmful(c_id) or is_harmful(d_id) or is_harmful(repl_id)
    if not any_harmful:
        continue

    baseline_maj = majority_choice(rec["baseline_choices"])
    patched_maj = majority_choice(rec["patched_choices"])
    if baseline_maj is None or patched_maj is None:
        continue
    if baseline_maj == patched_maj:
        continue  # not flipped

    # This is a flipped ordering with harmful task in recipient
    # Print first patched completion to see what the model actually did
    if count < 8:
        print(f"--- {rec['trial_key']} ---")
        print(f"  Condition: {cond}_{rec.get('swap_type', '')}")
        print(f"  Source: {rec['source_task_a_id']} vs {rec['source_task_b_id']}")
        if c_id:
            print(f"  Recipient: {c_id} vs {d_id}")
        if repl_id:
            print(f"  Replacement: {repl_id}")
        print(f"  Baseline choice: {baseline_maj} -> Patched choice: {patched_maj}")
        print(f"  Patched completion (first 200 chars):")
        print(f"    {rec['patched_completions'][0][:200]}")
        print()
    count += 1

print(f"\nTotal flipped orderings with harmful recipient task: {count}")
