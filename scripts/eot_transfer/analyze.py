"""Analyze EOT transfer experiment results.

Computes slot following rates, task following rates, and confusion rates
across all conditions. Produces summary tables and plots.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

EXPERIMENT_DIR = Path("experiments/patching/eot_transfer")
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoint.jsonl"
JUDGE_PATH = EXPERIMENT_DIR / "judge_results.json"
RESULTS_PATH = EXPERIMENT_DIR / "results.json"


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


def analyze_record(rec: dict, judge_results: dict) -> dict:
    """Analyze a single record. Returns trial-level results."""
    donor_slot = rec["donor_slot"]
    condition = rec["condition"]
    n_trials = len(rec["patched_completions"])

    trials = []
    for i in range(n_trials):
        # Stated choice from parse_sync
        stated_baseline = rec["baseline_choices"][i]
        stated_patched = rec["patched_choices"][i]

        # Judge results for patched completion
        judge_key = f"{rec['trial_key']}_patched_{i}"
        judge = judge_results.get(judge_key, {})

        # Slot following: did model pick the donor's slot?
        followed_slot = stated_patched == donor_slot

        # For swap_headers, map Task 1 -> a, Task 2 -> b
        if condition == "swap_headers":
            followed_slot = stated_patched == donor_slot

        trial = {
            "stated_baseline": stated_baseline,
            "stated_patched": stated_patched,
            "followed_donor_slot": followed_slot,
            "judge_stated": judge.get("stated_label"),
            "judge_executed": judge.get("executed_task"),
            "judge_refusal": judge.get("is_refusal"),
        }
        trials.append(trial)

    return {
        "trial_key": rec["trial_key"],
        "condition": condition,
        "donor_slot": donor_slot,
        "ordering_idx": rec["ordering_idx"],
        "swap_type": rec.get("swap_type"),
        "swap_direction": rec.get("swap_direction"),
        "trials": trials,
    }


def compute_metrics(analyzed: list[dict]) -> dict:
    """Compute aggregate metrics by condition."""
    by_condition = defaultdict(list)
    for rec in analyzed:
        cond = rec["condition"]
        if rec.get("swap_type"):
            cond = f"{cond}_{rec['swap_type']}"
        by_condition[cond].extend(rec["trials"])

    metrics = {}
    for cond, trials in sorted(by_condition.items()):
        n = len(trials)
        valid = [t for t in trials if t["stated_patched"] in ("a", "b")]
        n_valid = len(valid)

        slot_follow = sum(1 for t in valid if t["followed_donor_slot"])
        slot_rate = slot_follow / n_valid if n_valid > 0 else 0

        # Parse fail rate
        parse_fails = sum(1 for t in trials if t["stated_patched"] == "parse_fail")

        # Judge-based metrics (if available)
        judge_valid = [t for t in trials if t.get("judge_stated") in ("a", "b")]
        judge_slot_follow = sum(
            1 for t in judge_valid
            if t["judge_stated"] == trials[0].get("donor_slot", "")
        )

        # Refusal rate from judge
        refusals = sum(1 for t in trials if t.get("judge_refusal"))

        metrics[cond] = {
            "n_total": n,
            "n_valid": n_valid,
            "slot_following_rate": slot_rate,
            "slot_following_count": slot_follow,
            "parse_fail_rate": parse_fails / n if n > 0 else 0,
            "refusal_count": refusals,
        }

    return metrics


def compute_slot_following_with_judge(records: list[dict], judge_results: dict) -> dict:
    """Compute slot following using judge's stated_label (handles parse_fails)."""
    by_condition = defaultdict(lambda: {"follow": 0, "not_follow": 0, "unclear": 0, "total": 0})

    for rec in records:
        donor_slot = rec["donor_slot"]
        condition = rec["condition"]
        swap_type = rec.get("swap_type", "")
        cond_key = f"{condition}_{swap_type}" if swap_type else condition

        n_trials = len(rec["patched_completions"])
        for i in range(n_trials):
            judge_key = f"{rec['trial_key']}_patched_{i}"
            judge = judge_results.get(judge_key, {})
            stated = judge.get("stated_label")

            by_condition[cond_key]["total"] += 1
            if stated == donor_slot:
                by_condition[cond_key]["follow"] += 1
            elif stated in ("a", "b"):
                by_condition[cond_key]["not_follow"] += 1
            else:
                by_condition[cond_key]["unclear"] += 1

    result = {}
    for cond, counts in sorted(by_condition.items()):
        valid = counts["follow"] + counts["not_follow"]
        rate = counts["follow"] / valid if valid > 0 else 0
        result[cond] = {
            **counts,
            "valid": valid,
            "slot_following_rate": rate,
        }
    return result


def compute_direction_breakdown(records: list[dict], judge_results: dict) -> dict:
    """Slot following by recipient ordering direction (position bias check)."""
    by_cond_dir = defaultdict(lambda: {"follow": 0, "not_follow": 0, "total": 0})

    for rec in records:
        donor_slot = rec["donor_slot"]
        condition = rec["condition"]
        swap_dir = rec.get("swap_direction", "none")
        swap_type = rec.get("swap_type", "")
        cond_key = f"{condition}_{swap_type}" if swap_type else condition
        group_key = f"{cond_key}_{swap_dir}"

        n_trials = len(rec["patched_completions"])
        for i in range(n_trials):
            judge_key = f"{rec['trial_key']}_patched_{i}"
            judge = judge_results.get(judge_key, {})
            stated = judge.get("stated_label")

            by_cond_dir[group_key]["total"] += 1
            if stated == donor_slot:
                by_cond_dir[group_key]["follow"] += 1
            elif stated in ("a", "b"):
                by_cond_dir[group_key]["not_follow"] += 1

    result = {}
    for key, counts in sorted(by_cond_dir.items()):
        valid = counts["follow"] + counts["not_follow"]
        rate = counts["follow"] / valid if valid > 0 else 0
        result[key] = {"valid": valid, "slot_following_rate": rate, **counts}
    return result


def compute_stated_vs_executed(records: list[dict], judge_results: dict) -> dict:
    """Matrix of stated label vs executed task from judge."""
    by_condition = defaultdict(lambda: Counter())

    for rec in records:
        condition = rec["condition"]
        swap_type = rec.get("swap_type", "")
        cond_key = f"{condition}_{swap_type}" if swap_type else condition

        n_trials = len(rec["patched_completions"])
        for i in range(n_trials):
            judge_key = f"{rec['trial_key']}_patched_{i}"
            judge = judge_results.get(judge_key, {})
            stated = judge.get("stated_label", "unknown")
            executed = judge.get("executed_task", "unknown")
            by_condition[cond_key][(stated, executed)] += 1

    return {
        cond: dict(counts)
        for cond, counts in sorted(by_condition.items())
    }


def main():
    records, judge_results = load_data()
    print(f"Loaded {len(records)} records, {len(judge_results)} judge results")

    # Parse-based metrics
    analyzed = [analyze_record(rec, judge_results) for rec in records]
    metrics = compute_metrics(analyzed)

    print("\n=== Slot Following (parse_sync) ===")
    print(f"{'Condition':<30} {'Rate':>8} {'n_valid':>8} {'parse_fail%':>12}")
    for cond, m in metrics.items():
        print(f"{cond:<30} {m['slot_following_rate']:>8.1%} {m['n_valid']:>8} {m['parse_fail_rate']:>12.1%}")

    # Judge-based metrics
    if judge_results:
        judge_metrics = compute_slot_following_with_judge(records, judge_results)
        print("\n=== Slot Following (judge) ===")
        print(f"{'Condition':<30} {'Rate':>8} {'valid':>8} {'unclear':>8}")
        for cond, m in judge_metrics.items():
            print(f"{cond:<30} {m['slot_following_rate']:>8.1%} {m['valid']:>8} {m['unclear']:>8}")

        dir_breakdown = compute_direction_breakdown(records, judge_results)
        print("\n=== Direction Breakdown ===")
        print(f"{'Condition_Direction':<40} {'Rate':>8} {'valid':>8}")
        for key, m in dir_breakdown.items():
            print(f"{key:<40} {m['slot_following_rate']:>8.1%} {m['valid']:>8}")

        stated_vs_exec = compute_stated_vs_executed(records, judge_results)
        print("\n=== Stated vs Executed ===")
        for cond, counts in stated_vs_exec.items():
            print(f"\n{cond}:")
            for (stated, executed), count in sorted(counts.items()):
                print(f"  stated={stated}, executed={executed}: {count}")

    # Save results
    results = {
        "n_records": len(records),
        "n_judge_results": len(judge_results),
        "parse_metrics": metrics,
    }
    if judge_results:
        results["judge_metrics"] = judge_metrics
        results["direction_breakdown"] = dir_breakdown

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
