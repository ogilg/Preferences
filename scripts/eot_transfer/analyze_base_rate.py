"""Check whether swap_both flip rates are diluted by aligned baselines.

For each swap_both ordering:
- Compute baseline majority choice (which slot the model picks without patching)
- Check if donor pushes in the same direction (aligned) or opposite (conflicting)
- Compute flip rates separately for aligned vs conflicting orderings
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

CHECKPOINT_PATH = Path("experiments/patching/eot_transfer/checkpoint.jsonl")


def majority_choice(choices: list[str]) -> str | None:
    """Return majority choice, or None if tied."""
    counts = Counter(c for c in choices if c in ("a", "b"))
    if not counts:
        return None
    (top, top_n), *rest = counts.most_common()
    if rest and rest[0][1] == top_n:
        return None  # tie
    return top


def main():
    records = []
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    # Analyze each condition
    for condition_filter in ["swap_both", "swap_target_same_topic", "swap_target_cross_topic", "control"]:
        aligned = {"flipped": 0, "not_flipped": 0, "tie": 0}
        conflicting = {"flipped": 0, "not_flipped": 0, "tie": 0}

        for rec in records:
            cond = rec["condition"]
            swap_type = rec.get("swap_type", "")
            cond_key = f"{cond}_{swap_type}" if swap_type else cond
            if cond_key != condition_filter:
                continue

            donor_slot = rec["donor_slot"]
            baseline_maj = majority_choice(rec["baseline_choices"])
            patched_maj = majority_choice(rec["patched_choices"])

            if baseline_maj is None or patched_maj is None:
                aligned["tie"] += 1
                continue

            # Is donor push aligned with baseline?
            # Donor slot = the slot the donor picked. Patching pushes recipient toward that slot.
            is_aligned = baseline_maj == donor_slot
            flipped = patched_maj != baseline_maj

            bucket = aligned if is_aligned else conflicting
            if flipped:
                bucket["flipped"] += 1
            else:
                bucket["not_flipped"] += 1

        total_aligned = aligned["flipped"] + aligned["not_flipped"]
        total_conflicting = conflicting["flipped"] + conflicting["not_flipped"]
        total_all = total_aligned + total_conflicting

        print(f"\n=== {condition_filter} ===")
        print(f"Total orderings (excl ties): {total_all}")
        print(f"  Aligned (donor push = baseline): {total_aligned} ({100*total_aligned/total_all:.0f}%)")
        if total_aligned > 0:
            print(f"    Flipped: {aligned['flipped']}/{total_aligned} = {100*aligned['flipped']/total_aligned:.1f}%")
        print(f"  Conflicting (donor push != baseline): {total_conflicting} ({100*total_conflicting/total_all:.0f}%)")
        if total_conflicting > 0:
            print(f"    Flipped: {conflicting['flipped']}/{total_conflicting} = {100*conflicting['flipped']/total_conflicting:.1f}%")
        print(f"  Overall flip rate: {(aligned['flipped']+conflicting['flipped'])}/{total_all} = {100*(aligned['flipped']+conflicting['flipped'])/total_all:.1f}%")
        print(f"  Ties excluded: {aligned['tie']}")


if __name__ == "__main__":
    main()
