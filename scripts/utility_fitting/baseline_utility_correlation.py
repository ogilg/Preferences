"""Compute baseline utility correlation (Pearson r) for OOD experiments 1b, 1c, 1d.

For each experiment and condition:
1. Load baseline utilities and condition utilities
2. Compute Pearson r between them
3. Repeat excluding harmful tasks

Report mean ± SE across conditions for each experiment.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from dotenv import load_dotenv

load_dotenv()

BASE = Path(".")


def load_thurstonian_latest(run_dir: Path) -> dict[str, float]:
    """Load utilities from the most recent thurstonian CSV."""
    csvs = sorted(run_dir.glob("thurstonian_*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No thurstonian CSVs in {run_dir}")
    utilities = {}
    with open(csvs[-1]) as f:
        next(f)
        for line in f:
            task_id, mu, _ = line.strip().split(",")
            utilities[task_id] = float(mu)
    return utilities


def main():
    run_prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"

    experiments = {
        "exp1b": {
            "results_dir": BASE / "results/experiments/ood_exp1b/pre_task_active_learning",
        },
        "exp1c": {
            "results_dir": BASE / "results/experiments/ood_exp1c/pre_task_active_learning",
        },
        "exp1d": {
            "results_dir": BASE / "results/experiments/ood_exp1d/pre_task_active_learning",
        },
    }

    for exp_name, exp_info in experiments.items():
        print(f"\n{'='*70}")
        print(f"  {exp_name.upper()}")
        print(f"{'='*70}")

        results_dir = exp_info["results_dir"]

        # Load baseline utilities
        baseline_dir = results_dir / run_prefix
        if not baseline_dir.exists():
            print(f"  No baseline results at {baseline_dir}")
            continue

        baseline_utils = load_thurstonian_latest(baseline_dir)
        baseline_task_ids = set(baseline_utils.keys())

        # Identify harmful tasks from baseline
        harmful_ids = {t for t in baseline_task_ids if "_harmful" in t}
        non_harmful_ids = baseline_task_ids - harmful_ids

        print(f"Baseline: {len(baseline_task_ids)} tasks ({len(non_harmful_ids)} non-harmful, {len(harmful_ids)} harmful)")

        # Find all condition directories
        condition_dirs = [
            d
            for d in results_dir.iterdir()
            if d.is_dir() and d.name.startswith(run_prefix)
        ]
        condition_dirs = sorted([d for d in condition_dirs if d.name != run_prefix])

        print(f"Found {len(condition_dirs)} conditions")

        # Analyze each condition
        results = []
        for cond_dir in condition_dirs:
            cond_name = cond_dir.name[len(run_prefix) + 1 :]  # remove prefix and underscore

            try:
                cond_utils = load_thurstonian_latest(cond_dir)
            except FileNotFoundError:
                continue

            # Find shared tasks
            shared_tasks = baseline_task_ids & set(cond_utils.keys())
            if len(shared_tasks) < 5:
                continue

            # Align utilities
            baseline_vals = np.array([baseline_utils[t] for t in shared_tasks])
            cond_vals = np.array([cond_utils[t] for t in shared_tasks])

            # All tasks
            r_all, _ = stats.pearsonr(baseline_vals, cond_vals)

            # Excluding harmful
            non_harmful_shared = [t for t in shared_tasks if t not in harmful_ids]
            if len(non_harmful_shared) >= 5:
                baseline_nh = np.array([baseline_utils[t] for t in non_harmful_shared])
                cond_nh = np.array([cond_utils[t] for t in non_harmful_shared])
                r_no_harmful, _ = stats.pearsonr(baseline_nh, cond_nh)
            else:
                r_no_harmful = float("nan")

            results.append({
                "condition": cond_name,
                "r_all": r_all,
                "r_no_harmful": r_no_harmful,
                "n_shared": len(shared_tasks),
                "n_no_harmful": len(non_harmful_shared),
            })

        if not results:
            print("  No conditions analyzed.")
            continue

        # Print per-condition results
        print(f"\n{'Condition':<45} {'N':>4} {'N-NH':>4} {'r(all)':>8} {'r(no-H)':>8}")
        print("-" * 75)
        for r in results:
            print(
                f"  {r['condition']:<43} {r['n_shared']:>4} {r['n_no_harmful']:>4} "
                f"{r['r_all']:>8.3f} {r['r_no_harmful']:>8.3f}"
            )

        # Compute means
        rs_all = np.array([r["r_all"] for r in results])
        rs_no_harmful = np.array([r["r_no_harmful"] for r in results])

        # Filter out NaNs for stats
        rs_no_harmful_valid = rs_no_harmful[~np.isnan(rs_no_harmful)]

        n = len(results)
        n_valid_nh = len(rs_no_harmful_valid)

        print(f"\nMEAN across {n} conditions:")
        mean_all = np.mean(rs_all)
        se_all = np.std(rs_all) / np.sqrt(n)
        print(f"  All tasks:       r = {mean_all:.3f} ± {se_all:.3f}")

        if n_valid_nh > 0:
            mean_nh = np.mean(rs_no_harmful_valid)
            se_nh = np.std(rs_no_harmful_valid) / np.sqrt(n_valid_nh)
            print(f"  Excl harmful:    r = {mean_nh:.3f} ± {se_nh:.3f}")
        else:
            print(f"  Excl harmful:    (no valid data)")


if __name__ == "__main__":
    main()
