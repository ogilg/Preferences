"""Analyze probe predictions vs condition-specific utilities for OOD experiments."""

import hashlib
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from dotenv import load_dotenv

load_dotenv()


def load_thurstonian_latest(run_dir: Path) -> dict[str, float]:
    """Load utilities from the latest Thurstonian CSV (by mtime)."""
    csvs = sorted(run_dir.glob("thurstonian_*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No thurstonian CSVs in {run_dir}")
    csv_path = csvs[-1]
    utilities = {}
    with open(csv_path) as f:
        next(f)  # skip header
        for line in f:
            task_id, mu, _ = line.strip().split(",")
            utilities[task_id] = float(mu)
    return utilities


def map_configs_to_results(
    config_dir: Path, results_dir: Path, run_prefix: str
) -> dict[str, Path]:
    """Map condition names to result directories via system prompt hashing."""
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
        else:
            print(f"  WARNING: no results for {f.stem} -> {result_name}")
    return mapping


def load_activations_for_condition(
    act_dir: Path, layer: int, task_filter: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Load activations from npz, optionally filtering to specific tasks."""
    d = np.load(act_dir / "activations_prompt_last.npz")
    task_ids = list(d["task_ids"])
    acts = d[f"layer_{layer}"]

    if task_filter is not None:
        filter_set = set(task_filter)
        mask = [i for i, t in enumerate(task_ids) if t in filter_set]
        task_ids = [task_ids[i] for i in mask]
        acts = acts[mask]

    return acts, task_ids


def score_with_probe(
    probe_weights: np.ndarray, activations: np.ndarray
) -> np.ndarray:
    return activations @ probe_weights[:-1] + probe_weights[-1]


def pairwise_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of pairs where predicted ranking agrees with actual."""
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


def evaluate_prediction(
    predicted: np.ndarray, actual: np.ndarray
) -> dict[str, float]:
    """Compute Pearson r, R², and pairwise accuracy."""
    r, p = stats.pearsonr(predicted, actual)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pw_acc = pairwise_accuracy(predicted, actual)
    return {"pearson_r": r, "pearson_p": p, "r2": r2, "pairwise_acc": pw_acc}


def align_data(
    utilities: dict[str, float],
    task_ids_act: list[str],
    activations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Inner-join activations and utilities on shared task IDs."""
    shared = [t for t in task_ids_act if t in utilities]
    act_idx = {t: i for i, t in enumerate(task_ids_act)}
    acts = np.array([activations[act_idx[t]] for t in shared])
    utils = np.array([utilities[t] for t in shared])
    return acts, utils, shared


def analyze_experiment(
    exp_name: str,
    config_dir: Path,
    results_dir: Path,
    act_dir_base: Path,
    run_prefix: str,
    task_prefix: str,
    probe_weights: np.ndarray,
    layer: int,
) -> list[dict]:
    """Run full analysis for one experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")

    # Map conditions to result dirs
    condition_results = map_configs_to_results(config_dir, results_dir, run_prefix)
    print(f"Found {len(condition_results)} conditions with results")

    if "baseline" not in condition_results:
        print("  No baseline results found, skipping experiment")
        return []

    # Load baseline utilities and activations
    baseline_utils = load_thurstonian_latest(condition_results["baseline"])
    task_filter = [t for t in baseline_utils if t.startswith(task_prefix)]
    if not task_filter:
        task_filter = list(baseline_utils.keys())

    baseline_acts, baseline_task_ids = load_activations_for_condition(
        act_dir_base / "baseline", layer, task_filter
    )
    baseline_acts_aligned, baseline_utils_aligned, shared_baseline = align_data(
        baseline_utils, baseline_task_ids, baseline_acts
    )
    baseline_probe_scores = score_with_probe(probe_weights, baseline_acts_aligned)

    print(
        f"Baseline: {len(shared_baseline)} tasks, "
        f"probe r={stats.pearsonr(baseline_probe_scores, baseline_utils_aligned)[0]:.3f}"
    )

    results = []
    for cond_name, result_dir in sorted(condition_results.items()):
        if cond_name == "baseline":
            continue

        # Load condition utilities
        cond_utils = load_thurstonian_latest(result_dir)

        # Determine activation directory
        # For exp1_prompts conditions, the activation dirs match config names
        act_dir = act_dir_base / cond_name
        if not act_dir.exists():
            print(f"  SKIP {cond_name}: no activations at {act_dir}")
            continue

        # Load condition activations
        cond_acts, cond_task_ids = load_activations_for_condition(
            act_dir, layer, task_filter
        )

        # Align condition activations with condition utilities
        cond_acts_aligned, cond_utils_aligned, shared_tasks = align_data(
            cond_utils, cond_task_ids, cond_acts
        )

        if len(shared_tasks) < 5:
            print(f"  SKIP {cond_name}: only {len(shared_tasks)} shared tasks")
            continue

        # 1. Condition probe scores (probe applied to condition activations)
        cond_probe_scores = score_with_probe(probe_weights, cond_acts_aligned)
        metrics_cond = evaluate_prediction(cond_probe_scores, cond_utils_aligned)

        # 2. Baseline: baseline probe scores -> condition utilities
        # Need to align baseline probe scores to the same task set
        baseline_scores_for_cond = {}
        for t, score in zip(shared_baseline, baseline_probe_scores):
            baseline_scores_for_cond[t] = score
        shared_both = [t for t in shared_tasks if t in baseline_scores_for_cond]
        if len(shared_both) >= 5:
            bl_scores = np.array([baseline_scores_for_cond[t] for t in shared_both])
            cu = np.array([cond_utils[t] for t in shared_both])
            metrics_baseline_probe = evaluate_prediction(bl_scores, cu)
        else:
            metrics_baseline_probe = {
                "pearson_r": float("nan"),
                "r2": float("nan"),
                "pairwise_acc": float("nan"),
            }

        # 3. Baseline: baseline utilities -> condition utilities
        shared_utils = [t for t in shared_tasks if t in baseline_utils]
        if len(shared_utils) >= 5:
            bu = np.array([baseline_utils[t] for t in shared_utils])
            cu2 = np.array([cond_utils[t] for t in shared_utils])
            metrics_baseline_utils = evaluate_prediction(bu, cu2)
        else:
            metrics_baseline_utils = {
                "pearson_r": float("nan"),
                "r2": float("nan"),
                "pairwise_acc": float("nan"),
            }

        result = {
            "experiment": exp_name,
            "condition": cond_name,
            "n_tasks": len(shared_tasks),
            "cond_probe_r": metrics_cond["pearson_r"],
            "cond_probe_r2": metrics_cond["r2"],
            "cond_probe_acc": metrics_cond["pairwise_acc"],
            "bl_probe_r": metrics_baseline_probe["pearson_r"],
            "bl_probe_r2": metrics_baseline_probe["r2"],
            "bl_probe_acc": metrics_baseline_probe["pairwise_acc"],
            "bl_utils_r": metrics_baseline_utils["pearson_r"],
            "bl_utils_r2": metrics_baseline_utils["r2"],
            "bl_utils_acc": metrics_baseline_utils["pairwise_acc"],
        }
        results.append(result)

        print(
            f"  {cond_name}: n={len(shared_tasks)} | "
            f"cond_probe r={metrics_cond['pearson_r']:.3f} acc={metrics_cond['pairwise_acc']:.3f} | "
            f"bl_probe r={metrics_baseline_probe['pearson_r']:.3f} | "
            f"bl_utils r={metrics_baseline_utils['pearson_r']:.3f}"
        )

    return results


def analyze_mra(
    probe_weights: np.ndarray,
    layer: int,
) -> list[dict]:
    """Analyze MRA experiment 2 (role-induced preferences)."""
    print(f"\n{'='*60}")
    print("Experiment: MRA Exp2 (role-induced preferences)")
    print(f"{'='*60}")

    # Persona → (activation dir, system prompt hash for result lookup)
    # Hashes from configs/measurement/active_learning/mra_exp2/
    persona_info = {
        "no_prompt": {"act_dir": "gemma_3_27b", "hash": None},
        "villain": {"act_dir": "gemma_3_27b_villain", "hash": "syse8f24ac6"},
        "midwest": {"act_dir": "gemma_3_27b_midwest", "hash": "sys5d504504"},
        "aesthete": {"act_dir": "gemma_3_27b_aesthete", "hash": "sys021d8ca1"},
    }

    mra_results_dir = Path("results/experiments/mra_exp2/pre_task_active_learning")
    mra_villain_dir = Path("results/experiments/mra_villain/pre_task_active_learning")

    # Map each persona to its result dirs
    persona_result_dirs = {}
    for persona, info in persona_info.items():
        h = info["hash"]
        matching = []
        for d in sorted(mra_results_dir.iterdir()):
            if h is None and "_sys" not in d.name:
                matching.append(d)
            elif h is not None and f"_{h}_" in d.name:
                matching.append(d)
        persona_result_dirs[persona] = matching
        print(f"  {persona}: {len(matching)} result dirs")

    # Also add mra_villain results for villain
    if mra_villain_dir.exists():
        for d in sorted(mra_villain_dir.iterdir()):
            persona_result_dirs.setdefault("villain", []).append(d)
        print(f"  villain (mra_villain): +{len(list(mra_villain_dir.iterdir()))} dirs")

    # Load baseline (no_prompt) activations and utilities for baselines
    baseline_act_path = Path("activations/gemma_3_27b/activations_prompt_last.npz")
    bl_d = np.load(baseline_act_path)
    bl_task_ids = list(bl_d["task_ids"])
    bl_acts = bl_d[f"layer_{layer}"]
    bl_act_idx = {t: i for i, t in enumerate(bl_task_ids)}

    # Load all baseline utilities (merge across no_prompt result dirs)
    baseline_utils = {}
    for d in persona_result_dirs.get("no_prompt", []):
        baseline_utils.update(load_thurstonian_latest(d))
    print(f"  Baseline utilities: {len(baseline_utils)} tasks")

    results = []

    for persona, info in persona_info.items():
        if persona == "no_prompt":
            continue

        act_path = Path(f"activations/{info['act_dir']}/activations_prompt_last.npz")
        if not act_path.exists():
            print(f"  SKIP {persona}: no activations")
            continue

        d = np.load(act_path)
        task_ids = list(d["task_ids"])
        acts = d[f"layer_{layer}"]
        act_idx = {t: i for i, t in enumerate(task_ids)}
        print(f"\n  {persona}: {len(task_ids)} tasks in activations")

        for result_dir in persona_result_dirs.get(persona, []):
            utils = load_thurstonian_latest(result_dir)
            shared = [t for t in task_ids if t in utils]
            if len(shared) < 10:
                continue

            # Condition probe scores
            aligned_acts = np.array([acts[act_idx[t]] for t in shared])
            aligned_utils = np.array([utils[t] for t in shared])
            probe_scores = score_with_probe(probe_weights, aligned_acts)
            metrics_cond = evaluate_prediction(probe_scores, aligned_utils)

            # Baseline probe scores → condition utilities
            shared_bl = [t for t in shared if t in bl_act_idx]
            if len(shared_bl) >= 10:
                bl_aligned = np.array([bl_acts[bl_act_idx[t]] for t in shared_bl])
                bl_scores = score_with_probe(probe_weights, bl_aligned)
                bl_cu = np.array([utils[t] for t in shared_bl])
                metrics_bl_probe = evaluate_prediction(bl_scores, bl_cu)
            else:
                metrics_bl_probe = {"pearson_r": float("nan"), "r2": float("nan"), "pairwise_acc": float("nan")}

            # Baseline utilities → condition utilities
            shared_bu = [t for t in shared if t in baseline_utils]
            if len(shared_bu) >= 10:
                bu = np.array([baseline_utils[t] for t in shared_bu])
                cu = np.array([utils[t] for t in shared_bu])
                metrics_bl_utils = evaluate_prediction(bu, cu)
            else:
                metrics_bl_utils = {"pearson_r": float("nan"), "r2": float("nan"), "pairwise_acc": float("nan")}

            dir_label = result_dir.parent.parent.parent.name  # experiment name
            result = {
                "experiment": f"mra_{dir_label}",
                "condition": persona,
                "n_tasks": len(shared),
                "cond_probe_r": metrics_cond["pearson_r"],
                "cond_probe_r2": metrics_cond["r2"],
                "cond_probe_acc": metrics_cond["pairwise_acc"],
                "bl_probe_r": metrics_bl_probe["pearson_r"],
                "bl_probe_r2": metrics_bl_probe["r2"],
                "bl_probe_acc": metrics_bl_probe["pairwise_acc"],
                "bl_utils_r": metrics_bl_utils["pearson_r"],
                "bl_utils_r2": metrics_bl_utils["r2"],
                "bl_utils_acc": metrics_bl_utils["pairwise_acc"],
            }
            results.append(result)

            print(
                f"    {persona} vs {dir_label}: n={len(shared)} | "
                f"cond r={metrics_cond['pearson_r']:.3f} acc={metrics_cond['pairwise_acc']:.3f} | "
                f"bl_probe r={metrics_bl_probe['pearson_r']:.3f} | "
                f"bl_utils r={metrics_bl_utils['pearson_r']:.3f}"
            )

    return results


def main():
    # Load the baseline probe
    probe_dir = Path("results/probes/gemma3_10k_heldout_std_raw")
    manifest_path = probe_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    layer = 31  # Start with best-performing layer
    probe_id = f"ridge_L{layer:02d}"
    probe_weights = np.load(probe_dir / "probes" / f"probe_{probe_id}.npy")
    print(f"Loaded probe {probe_id} from {probe_dir}")
    print(f"Probe shape: {probe_weights.shape}")

    # Check probe performance from manifest
    for p in manifest["probes"]:
        if p["id"] == probe_id:
            print(f"Manifest metrics: r={p.get('final_r', 'N/A')}, acc={p.get('final_acc', 'N/A')}")
            break

    all_results = []

    # Run prefix for OOD experiments
    run_prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"

    # Exp 1b: Hidden preference
    results_1b = analyze_experiment(
        exp_name="exp1b_hidden",
        config_dir=Path("configs/measurement/active_learning/ood_exp1b"),
        results_dir=Path("results/experiments/ood_exp1b/pre_task_active_learning"),
        act_dir_base=Path("activations/ood/exp1_prompts"),
        run_prefix=run_prefix,
        task_prefix="hidden_",
        probe_weights=probe_weights,
        layer=layer,
    )
    all_results.extend(results_1b)

    # Exp 1c: Crossed preference (same configs as 1b, different tasks)
    results_1c = analyze_experiment(
        exp_name="exp1c_crossed",
        config_dir=Path("configs/measurement/active_learning/ood_exp1c"),
        results_dir=Path("results/experiments/ood_exp1c/pre_task_active_learning"),
        act_dir_base=Path("activations/ood/exp1_prompts"),
        run_prefix=run_prefix,
        task_prefix="crossed_",
        probe_weights=probe_weights,
        layer=layer,
    )
    all_results.extend(results_1c)

    # Exp 1d: Competing preference
    # Activations for competing conditions are in exp1_prompts/compete_*
    results_1d = analyze_experiment(
        exp_name="exp1d_competing",
        config_dir=Path("configs/measurement/active_learning/ood_exp1d"),
        results_dir=Path("results/experiments/ood_exp1d/pre_task_active_learning"),
        act_dir_base=Path("activations/ood/exp1_prompts"),
        run_prefix=run_prefix,
        task_prefix="crossed_",
        probe_weights=probe_weights,
        layer=layer,
    )
    all_results.extend(results_1d)

    # Exp 1a: Category preference
    results_1a = analyze_experiment(
        exp_name="exp1a_category",
        config_dir=Path("configs/measurement/active_learning/ood_exp1a"),
        results_dir=Path("results/experiments/ood_exp1a/pre_task_active_learning"),
        act_dir_base=Path("activations/ood/exp1_category"),
        run_prefix=run_prefix,
        task_prefix="",  # mixed task types
        probe_weights=probe_weights,
        layer=layer,
    )
    all_results.extend(results_1a)

    # MRA Exp2
    mra_results = analyze_mra(probe_weights, layer)
    all_results.extend(mra_results)

    # Save results
    output_path = Path("experiments/ood_system_prompts/utility_fitting/analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results to {output_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(
        f"{'Experiment':<20} {'Condition':<35} {'N':>4} "
        f"{'Cond r':>8} {'Cond acc':>9} "
        f"{'BL probe r':>10} {'BL utils r':>10}"
    )
    print("-" * 100)
    for r in all_results:
        print(
            f"{r['experiment']:<20} {r['condition']:<35} {r['n_tasks']:>4} "
            f"{r['cond_probe_r']:>8.3f} {r['cond_probe_acc']:>9.3f} "
            f"{r.get('bl_probe_r', float('nan')):>10.3f} "
            f"{r.get('bl_utils_r', float('nan')):>10.3f}"
        )


if __name__ == "__main__":
    main()
