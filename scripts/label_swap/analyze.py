"""Analyze label swap experiment results.

Key question: does the donor EOT follow position/label or content?

Setup:
- Donor = source ordering (model naturally picks baseline_dominant slot)
- Recipient = reversed ordering (tasks swapped)
- In recipient: preferred task moved to opposite slot from donor

If position-following: patched picks donor's slot -> wrong task -> flip from baseline
If content-following: patched picks wherever preferred task is -> same as baseline -> no flip

The completion judge classifies stated label vs executed content for dissociation analysis.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path("experiments/patching/eot_transfer/label_swap")
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoint.jsonl"
JUDGE_PATH = EXPERIMENT_DIR / "judge_results.json"


def load_data():
    records = []
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    judge_results = {}
    if JUDGE_PATH.exists():
        with open(JUDGE_PATH) as f:
            for r in json.load(f):
                key = f"{r['trial_key']}_{r['phase']}_{r['trial_idx']}"
                judge_results[key] = r

    return records, judge_results


def majority_choice(choices: list[str]) -> str | None:
    valid = [c for c in choices if c in ("a", "b")]
    if len(valid) < 3:
        return None
    counts = Counter(valid)
    return counts.most_common(1)[0][0]


def bootstrap_ci(data: list[float], n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    arr = np.array(data)
    boot_means = np.array([
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1 - alpha) * 100))


def main():
    records, judge_results = load_data()
    print(f"Loaded {len(records)} records, {len(judge_results)} judge results")

    # === Flip rate (parse-based) ===
    flips = []
    excluded = 0
    for rec in records:
        baseline_dom = majority_choice(rec["baseline_choices"])
        patched_dom = majority_choice(rec["patched_choices"])
        if baseline_dom is None or patched_dom is None:
            excluded += 1
            continue
        flips.append(1 if baseline_dom != patched_dom else 0)

    flip_rate = np.mean(flips) if flips else 0
    ci_lo, ci_hi = bootstrap_ci(flips) if len(flips) > 1 else (0, 0)
    print(f"\n=== Flip Rate (parse-based) ===")
    print(f"Flip rate: {flip_rate:.1%} [{ci_lo:.1%}, {ci_hi:.1%}]")
    print(f"n={len(flips)}, excluded={excluded}")

    # === Per-trial analysis ===
    # For each patched trial, check what the model stated and executed
    stated_counts = Counter()
    executed_counts = Counter()
    stated_executed_matrix = Counter()

    # Track position-following vs content-following
    position_following = 0
    content_following = 0
    other = 0
    total_valid = 0

    for rec in records:
        baseline_dominant = rec["baseline_dominant"]
        # In source: model picks baseline_dominant slot
        # In recipient (reversed): preferred task is in OPPOSITE slot
        # donor_source_slot = baseline_dominant (what donor pushes toward)
        # If position-following: patched picks donor_source_slot in recipient

        n_trials = len(rec["patched_completions"])
        for i in range(n_trials):
            judge_key = f"{rec['trial_key']}_patched_{i}"
            judge = judge_results.get(judge_key)
            if not judge or "error" in judge:
                continue

            stated = judge.get("stated_label")
            executed = judge.get("executed_task")
            is_refusal = judge.get("is_refusal", False)

            stated_counts[stated] += 1
            executed_counts[executed] += 1
            stated_executed_matrix[(stated, executed)] += 1

            if is_refusal or stated not in ("a", "b") or executed not in ("a", "b"):
                other += 1
                continue

            total_valid += 1

            # In the recipient (reversed ordering):
            # - Slot A has the OTHER task (not the one donor preferred)
            # - Slot B has the OTHER task
            # Wait, let's be precise:
            # Source: task_a in A, task_b in B. Model picks baseline_dominant.
            # Recipient: task_b in A, task_a in B.
            # Donor pushes toward baseline_dominant slot.
            # If baseline_dominant="b", donor pushes toward slot B.
            # In recipient, slot B has task_a (the NON-preferred task).
            # Position-following: patched picks slot B = task_a
            # Content-following: patched picks wherever preferred task is
            #   Preferred task was in baseline_dominant of source.
            #   In recipient, preferred task is in OPPOSITE slot.
            #   So content-following picks opposite of baseline_dominant.

            donor_slot = baseline_dominant
            opposite_slot = "b" if baseline_dominant == "a" else "a"

            if stated == donor_slot:
                position_following += 1
            elif stated == opposite_slot:
                content_following += 1

    print(f"\n=== Patched Trial Classification (judge) ===")
    print(f"Position/label-following: {position_following} ({position_following/total_valid:.1%})" if total_valid else "")
    print(f"Content-following: {content_following} ({content_following/total_valid:.1%})" if total_valid else "")
    print(f"Other (refusal/unclear): {other}")
    print(f"Total valid: {total_valid}")

    print(f"\n=== Stated Label Distribution (patched) ===")
    for label, count in sorted(stated_counts.items()):
        print(f"  {label}: {count}")

    print(f"\n=== Executed Task Distribution (patched) ===")
    for task, count in sorted(executed_counts.items()):
        print(f"  {task}: {count}")

    print(f"\n=== Stated vs Executed Matrix (patched) ===")
    for (stated, executed), count in sorted(stated_executed_matrix.items()):
        print(f"  stated={stated}, executed={executed}: {count}")

    # === Baseline analysis ===
    print(f"\n=== Baseline Stats ===")
    baseline_stated = Counter()
    for rec in records:
        n_trials = len(rec["baseline_completions"])
        for i in range(n_trials):
            judge_key = f"{rec['trial_key']}_baseline_{i}"
            judge = judge_results.get(judge_key)
            if judge and "error" not in judge:
                baseline_stated[judge.get("stated_label")] += 1

    for label, count in sorted(baseline_stated.items()):
        print(f"  {label}: {count}")

    # === Dissociation analysis ===
    # Cases where stated label != executed task
    print(f"\n=== Dissociation (stated != executed) ===")
    dissociation = 0
    aligned = 0
    for (stated, executed), count in stated_executed_matrix.items():
        if stated in ("a", "b") and executed in ("a", "b"):
            if stated != executed:
                dissociation += count
            else:
                aligned += count
    total_clear = dissociation + aligned
    if total_clear > 0:
        print(f"Dissociation rate: {dissociation/total_clear:.1%} ({dissociation}/{total_clear})")
    else:
        print("No clear stated+executed pairs")

    # === Per-ordering breakdown ===
    print(f"\n=== Per-ordering flip analysis ===")
    baseline_dominant_dist = Counter()
    for rec in records:
        baseline_dominant_dist[rec["baseline_dominant"]] += 1
    print(f"Baseline dominant in source: {dict(baseline_dominant_dist)}")

    # Check baseline determinism on recipient
    baseline_det = 0
    baseline_nondet = 0
    for rec in records:
        dom = majority_choice(rec["baseline_choices"])
        if dom is None:
            continue
        valid = [c for c in rec["baseline_choices"] if c in ("a", "b")]
        counts = Counter(valid)
        max_frac = max(counts.values()) / len(valid)
        if max_frac >= 0.8:
            baseline_det += 1
        else:
            baseline_nondet += 1
    print(f"Baseline on recipient: deterministic={baseline_det}, non-deterministic={baseline_nondet}")


if __name__ == "__main__":
    main()
