"""Analyze EOT scaled experiment results across all three phases.

Computes:
1. Overall flip rate (Phase 1)
2. Per-layer flip rates (Phase 2)
3. Layer combination additivity (Phase 3)
4. Flip rate vs |Δμ| analysis
5. Task-specific effects
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path("experiments/patching/eot_scaled")
TASKS_PATH = EXPERIMENT_DIR / "selected_tasks.json"


def load_tasks():
    with open(TASKS_PATH) as f:
        tasks = json.load(f)
    return {t["task_id"]: t for t in tasks}


def majority_choice(choices: list[str]) -> str | None:
    """Return majority choice (a or b), or None if tied or all parse_fail."""
    a = choices.count("a")
    b = choices.count("b")
    if a > b:
        return "a"
    if b > a:
        return "b"
    return None


def analyze_phase1():
    """Analyze Phase 1: overall flip rate, direction, task effects."""
    tasks = load_tasks()
    records = []
    with open(EXPERIMENT_DIR / "phase1_checkpoint.jsonl") as f:
        for line in f:
            records.append(json.loads(line))

    total = 0
    flipped = 0
    correct_direction = 0
    ambiguous_baseline = 0
    parse_fail_dominant = 0

    # Per-task flip counts
    task_flip_counts = Counter()
    task_total_counts = Counter()

    # Delta mu vs flip
    delta_mu_flips = []  # (|Δμ|, flipped_bool)

    for rec in records:
        base = rec["baseline_choices"]
        patch = rec["patched_choices"]

        # Count parse failures
        base_pf = base.count("parse_fail")
        patch_pf = patch.count("parse_fail")
        if base_pf > 7 and patch_pf > 7:
            parse_fail_dominant += 1
            continue

        base_choice = majority_choice(base)
        patch_choice = majority_choice(patch)

        if base_choice is None:
            ambiguous_baseline += 1
            continue

        total += 1
        ta_id = rec["task_a_id"]
        tb_id = rec["task_b_id"]
        task_total_counts[ta_id] += 1
        task_total_counts[tb_id] += 1

        # Compute delta mu
        ta = tasks[ta_id]
        tb = tasks[tb_id]
        delta_mu = abs(ta["mu"] - tb["mu"])

        if patch_choice is not None and base_choice != patch_choice:
            flipped += 1
            task_flip_counts[ta_id] += 1
            task_flip_counts[tb_id] += 1
            delta_mu_flips.append((delta_mu, True))
        else:
            delta_mu_flips.append((delta_mu, False))

    print("=" * 60)
    print("PHASE 1 ANALYSIS")
    print("=" * 60)
    print(f"Total records: {len(records)}")
    print(f"Parse-fail dominant (excluded): {parse_fail_dominant}")
    print(f"Ambiguous baseline (excluded): {ambiguous_baseline}")
    print(f"Analyzed orderings: {total}")
    print(f"Flipped: {flipped}/{total} = {flipped/total:.1%}" if total > 0 else "No data")

    # Position bias
    position_a_count = 0
    position_total = 0
    for rec in records:
        base = rec["baseline_choices"]
        for c in base:
            if c in ("a", "b"):
                position_total += 1
                if c == "a":
                    position_a_count += 1
    if position_total > 0:
        print(f"\nPosition bias: P(choose A) = {position_a_count/position_total:.3f} ({position_a_count}/{position_total})")

    # Task-specific effects
    print(f"\nTop 10 tasks by flip count:")
    for task_id, count in task_flip_counts.most_common(10):
        total_for_task = task_total_counts[task_id]
        mu = tasks[task_id]["mu"]
        print(f"  {task_id:35s}  flips={count}/{total_for_task}  mu={mu:+.2f}")

    # Delta mu bins
    if delta_mu_flips:
        bins = np.linspace(0, 20, 11)
        bin_flips = defaultdict(lambda: [0, 0])  # [flipped, total]
        for dm, flip in delta_mu_flips:
            bin_idx = min(int(dm / 2), 9)
            bin_flips[bin_idx][1] += 1
            if flip:
                bin_flips[bin_idx][0] += 1

        print(f"\nFlip rate by |Δμ| bin:")
        for i in range(10):
            f, t = bin_flips[i]
            rate = f / t if t > 0 else 0
            lo, hi = i * 2, (i + 1) * 2
            print(f"  |Δμ| [{lo:2d}-{hi:2d}]: {f:4d}/{t:4d} = {rate:.1%}")

    return {"total": total, "flipped": flipped, "flip_rate": flipped / total if total > 0 else 0}


def analyze_phase2():
    """Analyze Phase 2: per-layer flip rates."""
    records = []
    p2_path = EXPERIMENT_DIR / "phase2_checkpoint.jsonl"
    if not p2_path.exists():
        print("\nPhase 2 checkpoint not found, skipping.")
        return {}

    with open(p2_path) as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        print("\nPhase 2: no records.")
        return {}

    # Per-layer flip counts
    layer_flips = defaultdict(int)
    layer_totals = defaultdict(int)
    total = len(records)

    for rec in records:
        baseline_chose_a = rec["baseline_chose_a"]
        for layer_str, choices in rec["layer_choices"].items():
            layer = int(layer_str)
            layer_totals[layer] += 1

            patch_choice = majority_choice(choices)
            if patch_choice is None:
                continue

            patched_chose_a = (patch_choice == "a")
            if baseline_chose_a != patched_chose_a:
                layer_flips[layer] += 1

    print("\n" + "=" * 60)
    print("PHASE 2 ANALYSIS — Per-Layer Flip Rates")
    print("=" * 60)
    print(f"Orderings tested: {total}")

    layers_sorted = sorted(layer_totals.keys())
    print(f"\nLayer | Flips | Rate")
    print("-" * 35)
    for layer in layers_sorted:
        f = layer_flips[layer]
        t = layer_totals[layer]
        rate = f / t if t > 0 else 0
        bar = "#" * int(rate * 40)
        print(f"  L{layer:2d}  | {f:4d}/{t:4d} | {rate:.1%} {bar}")

    # Top layers
    top_layers = sorted(layer_totals.keys(), key=lambda l: layer_flips.get(l, 0) / max(layer_totals[l], 1), reverse=True)
    print(f"\nTop 10 layers:")
    for l in top_layers[:10]:
        rate = layer_flips[l] / layer_totals[l] if layer_totals[l] > 0 else 0
        print(f"  L{l}: {rate:.1%}")

    return {
        "per_layer": {l: {"flips": layer_flips[l], "total": layer_totals[l], "rate": layer_flips[l] / layer_totals[l] if layer_totals[l] > 0 else 0} for l in layers_sorted}
    }


def analyze_phase3():
    """Analyze Phase 3: layer combination results."""
    p3_path = EXPERIMENT_DIR / "phase3_checkpoint.jsonl"
    if not p3_path.exists():
        print("\nPhase 3 checkpoint not found, skipping.")
        return {}

    records = []
    with open(p3_path) as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        print("\nPhase 3: no records.")
        return {}

    # Per-combination flip counts
    combo_flips = defaultdict(int)
    combo_totals = defaultdict(int)
    total = len(records)

    for rec in records:
        baseline_chose_a = rec["baseline_chose_a"]
        for combo_name, choices in rec["combo_choices"].items():
            combo_totals[combo_name] += 1

            patch_choice = majority_choice(choices)
            if patch_choice is None:
                continue

            patched_chose_a = (patch_choice == "a")
            if baseline_chose_a != patched_chose_a:
                combo_flips[combo_name] += 1

    print("\n" + "=" * 60)
    print("PHASE 3 ANALYSIS — Layer Combinations")
    print("=" * 60)
    print(f"Orderings tested: {total}")

    # Sort by flip rate
    combos_sorted = sorted(combo_totals.keys(), key=lambda c: combo_flips[c] / max(combo_totals[c], 1), reverse=True)
    print(f"\nCombination | Flips | Rate")
    print("-" * 60)
    for combo in combos_sorted:
        f = combo_flips[combo]
        t = combo_totals[combo]
        rate = f / t if t > 0 else 0
        print(f"  {combo:40s} | {f:4d}/{t:4d} | {rate:.1%}")

    return {
        "per_combo": {c: {"flips": combo_flips[c], "total": combo_totals[c], "rate": combo_flips[c] / combo_totals[c] if combo_totals[c] > 0 else 0} for c in combos_sorted}
    }


def main():
    p1 = analyze_phase1()
    p2 = analyze_phase2()
    p3 = analyze_phase3()

    # Save analysis summary
    summary = {"phase1": p1, "phase2": p2, "phase3": p3}
    with open(EXPERIMENT_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved analysis summary to analysis_summary.json")


if __name__ == "__main__":
    main()
