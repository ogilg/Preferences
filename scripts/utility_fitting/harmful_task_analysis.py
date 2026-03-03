"""Investigate harmful tasks in OOD utility fitting experiments (1b, 1c, 1d).

For each experiment:
1. List all 48 task IDs, identify harmful ones, print their text
2. Compute probe r and acc with and without harmful tasks
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from dotenv import load_dotenv

load_dotenv()

BASE = Path(".")


def load_thurstonian_latest(run_dir: Path) -> dict[str, float]:
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


def load_tasks_by_id() -> dict[str, dict]:
    """Load task definitions from the OOD task JSON files."""
    tasks = {}
    for json_path in (BASE / "configs/ood/tasks").glob("*.json"):
        data = json.loads(json_path.read_text())
        if not isinstance(data, list):
            continue
        for task in data:
            if not isinstance(task, dict) or "task_id" not in task:
                continue
            tasks[task["task_id"]] = task
    return tasks


def pairwise_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    n = len(predicted)
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if actual[i] == actual[j]:
                continue
            total += 1
            if (predicted[i] - predicted[j]) * (actual[i] - actual[j]) > 0:
                correct += 1
    return correct / total if total > 0 else float("nan")


def score_with_probe(probe_weights: np.ndarray, activations: np.ndarray) -> np.ndarray:
    return activations @ probe_weights[:-1] + probe_weights[-1]


def compute_metrics(
    probe_scores: np.ndarray, utilities: np.ndarray
) -> dict[str, float]:
    r, _ = stats.pearsonr(probe_scores, utilities)
    acc = pairwise_accuracy(probe_scores, utilities)
    return {"r": r, "acc": acc}


def analyze_condition(
    cond_name: str,
    result_dir: Path,
    act_dir: Path,
    probe_weights: np.ndarray,
    layer: int,
    harmful_ids: set[str],
) -> dict | None:
    """Analyze a single condition, returning metrics with/without harmful tasks."""
    utils = load_thurstonian_latest(result_dir)
    task_ids = list(utils.keys())

    act_path = act_dir / "activations_prompt_last.npz"
    if not act_path.exists():
        return None
    d = np.load(act_path)
    act_task_ids = list(d["task_ids"])
    acts = d[f"layer_{layer}"]
    act_idx = {t: i for i, t in enumerate(act_task_ids)}

    # Align
    shared = [t for t in task_ids if t in act_idx]
    if len(shared) < 5:
        return None

    aligned_acts = np.array([acts[act_idx[t]] for t in shared])
    aligned_utils = np.array([utils[t] for t in shared])
    probe_scores = score_with_probe(probe_weights, aligned_acts)

    # All tasks
    all_metrics = compute_metrics(probe_scores, aligned_utils)

    # Without harmful
    non_harmful_mask = [i for i, t in enumerate(shared) if t not in harmful_ids]
    if len(non_harmful_mask) >= 5:
        nh_metrics = compute_metrics(
            probe_scores[non_harmful_mask], aligned_utils[non_harmful_mask]
        )
    else:
        nh_metrics = {"r": float("nan"), "acc": float("nan")}

    # Only harmful
    harmful_mask = [i for i, t in enumerate(shared) if t in harmful_ids]
    n_harmful = len(harmful_mask)

    return {
        "condition": cond_name,
        "n_total": len(shared),
        "n_harmful": n_harmful,
        "all_r": all_metrics["r"],
        "all_acc": all_metrics["acc"],
        "no_harmful_r": nh_metrics["r"],
        "no_harmful_acc": nh_metrics["acc"],
    }


def map_configs_to_results(
    config_dir: Path, results_dir: Path, run_prefix: str
) -> dict[str, Path]:
    import hashlib
    import yaml

    mapping = {}
    for f in sorted(config_dir.glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text())
        sp = cfg.get("measurement_system_prompt", "")
        if sp:
            h = hashlib.sha256(sp.encode()).hexdigest()[:8]
            result_name = f"{run_prefix}_sys{h}"
        else:
            result_name = run_prefix
        result_path = results_dir / result_name
        if result_path.exists():
            mapping[f.stem] = result_path
    return mapping


def main():
    # Load tasks
    all_tasks = load_tasks_by_id()

    # Load probe
    probe_dir = BASE / "results/probes/gemma3_10k_heldout_std_raw"
    layer = 31
    probe_weights = np.load(probe_dir / "probes" / f"probe_ridge_L{layer:02d}.npy")
    print(f"Loaded probe ridge_L{layer:02d}, shape {probe_weights.shape}")

    run_prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"

    experiments = {
        "exp1b": {
            "config_dir": BASE / "configs/measurement/active_learning/ood_exp1b",
            "results_dir": BASE / "results/experiments/ood_exp1b/pre_task_active_learning",
            "task_prefix": "hidden_",
        },
        "exp1c": {
            "config_dir": BASE / "configs/measurement/active_learning/ood_exp1c",
            "results_dir": BASE / "results/experiments/ood_exp1c/pre_task_active_learning",
            "task_prefix": "crossed_",
        },
        "exp1d": {
            "config_dir": BASE / "configs/measurement/active_learning/ood_exp1d",
            "results_dir": BASE / "results/experiments/ood_exp1d/pre_task_active_learning",
            "task_prefix": "crossed_",
        },
    }

    act_dir_base = BASE / "activations/ood/exp1_prompts"

    for exp_name, exp_info in experiments.items():
        print(f"\n{'='*70}")
        print(f"  {exp_name.upper()}")
        print(f"{'='*70}")

        # Load baseline to get all task IDs
        baseline_dir = exp_info["results_dir"] / run_prefix
        if not baseline_dir.exists():
            print(f"  No baseline results at {baseline_dir}")
            continue

        baseline_utils = load_thurstonian_latest(baseline_dir)
        task_ids = sorted(baseline_utils.keys())

        # Identify harmful tasks
        harmful_ids = {t for t in task_ids if "_harmful" in t}
        non_harmful_ids = {t for t in task_ids if "_harmful" not in t}

        print(f"\nAll {len(task_ids)} task IDs:")
        for t in task_ids:
            marker = " [HARMFUL]" if t in harmful_ids else ""
            print(f"  {t}{marker}")

        print(f"\n  Harmful: {len(harmful_ids)}, Non-harmful: {len(non_harmful_ids)}")

        if harmful_ids:
            print(f"\nHarmful task texts:")
            for t in sorted(harmful_ids):
                task_info = all_tasks.get(t, {})
                prompt = task_info.get("prompt", "<not found>")
                shell = task_info.get("category_shell", "?")
                print(f"  {t} (shell={shell}):")
                print(f"    {prompt[:200]}")
                print()

        # Map conditions to result dirs
        condition_results = map_configs_to_results(
            exp_info["config_dir"], exp_info["results_dir"], run_prefix
        )
        print(f"Found {len(condition_results)} conditions (excl baseline)")

        # Analyze each condition
        all_cond_results = []
        for cond_name, result_dir in sorted(condition_results.items()):
            if cond_name == "baseline":
                continue
            act_dir = act_dir_base / cond_name
            if not act_dir.exists():
                continue
            result = analyze_condition(
                cond_name, result_dir, act_dir, probe_weights, layer, harmful_ids
            )
            if result is not None:
                all_cond_results.append(result)

        if not all_cond_results:
            print("  No conditions analyzed.")
            continue

        # Print per-condition table
        print(f"\n{'Condition':<45} {'N':>3} {'Nharm':>5}  "
              f"{'All r':>7} {'All acc':>8}  {'NoHarm r':>8} {'NoHarm acc':>10}")
        print("-" * 100)
        for r in all_cond_results:
            print(
                f"  {r['condition']:<43} {r['n_total']:>3} {r['n_harmful']:>5}  "
                f"{r['all_r']:>7.3f} {r['all_acc']:>8.3f}  "
                f"{r['no_harmful_r']:>8.3f} {r['no_harmful_acc']:>10.3f}"
            )

        # Compute means
        all_rs = [r["all_r"] for r in all_cond_results]
        all_accs = [r["all_acc"] for r in all_cond_results]
        nh_rs = [r["no_harmful_r"] for r in all_cond_results]
        nh_accs = [r["no_harmful_acc"] for r in all_cond_results]

        n = len(all_cond_results)
        print(f"\n  MEAN (n={n} conditions):")
        print(f"    All tasks:          r = {np.mean(all_rs):.3f} +/- {np.std(all_rs)/np.sqrt(n):.3f},  "
              f"acc = {np.mean(all_accs):.3f} +/- {np.std(all_accs)/np.sqrt(n):.3f}")
        print(f"    Excl harmful:       r = {np.mean(nh_rs):.3f} +/- {np.std(nh_rs)/np.sqrt(n):.3f},  "
              f"acc = {np.mean(nh_accs):.3f} +/- {np.std(nh_accs)/np.sqrt(n):.3f}")
        print(f"    Delta (no-harm - all): r = {np.mean(nh_rs) - np.mean(all_rs):+.3f}, "
              f"acc = {np.mean(nh_accs) - np.mean(all_accs):+.3f}")


if __name__ == "__main__":
    main()
