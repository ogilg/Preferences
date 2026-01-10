"""Analysis of Thurstonian model sensitivity to optimization bounds and initial conditions.

Uses only real measurement data from results/measurements/, aggregated across all prompt templates.

Run: python -m thurstonian_analysis.bounds_sensitivity
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import spearmanr

from src.preferences.ranking.thurstonian import (
    PairwiseData,
    fit_thurstonian,
    DEFAULT_MU_BOUNDS,
    DEFAULT_LOG_SIGMA_BOUNDS,
)
from src.task_data import Task, OriginDataset


from src.experiments.thurstonian_analysis.config import N_TASKS, RESULTS_DIR
from src.experiments.thurstonian_analysis.utils import split_wins, eval_held_out_nll, eval_held_out_accuracy

OUTPUT_DIR = Path(__file__).parent / "plots" / "bounds_sensitivity"


@dataclass
class MuBoundsConfig:
    name: str
    short_name: str
    mu_bounds: tuple[float, float]


@dataclass
class LogSigmaBoundsConfig:
    name: str
    short_name: str
    log_sigma_bounds: tuple[float, float]


MU_BOUNDS_CONFIGS = [
    MuBoundsConfig("Tight", "μ±5", (-5.0, 5.0)),
    MuBoundsConfig("Default", "μ±10", (-10.0, 10.0)),
    MuBoundsConfig("Wide", "μ±20", (-20.0, 20.0)),
    MuBoundsConfig("Very Wide", "μ±50", (-50.0, 50.0)),
    MuBoundsConfig("Unbounded", "μ±1000", (-1000.0, 1000.0)),
]

LOG_SIGMA_BOUNDS_CONFIGS = [
    LogSigmaBoundsConfig("Very Tight", "logσ±1", (-1.0, 1.0)),
    LogSigmaBoundsConfig("Tight", "logσ±2", (-2.0, 2.0)),
    LogSigmaBoundsConfig("Default", "logσ±3", (-3.0, 3.0)),
    LogSigmaBoundsConfig("Wide", "logσ±5", (-5.0, 5.0)),
    LogSigmaBoundsConfig("Very Wide", "logσ±10", (-10.0, 10.0)),
]

SIGMA_INIT_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]


def make_task(id: str) -> Task:
    return Task(prompt=f"Task {id}", origin=OriginDataset.ALPACA, id=id, metadata={})


def fit_with_mu_bounds(data: PairwiseData, config: MuBoundsConfig):
    result = fit_thurstonian(
        data,
        mu_bounds=config.mu_bounds,
        log_sigma_bounds=(-3.0, 3.0),
        max_iter=1000,
    )
    if not result.converged:
        print(f"[!] {config.name}: {result.termination_message}", end=" ", flush=True)
    return {
        "config": config.name,
        "converged": result.converged,
        "iterations": result.n_iterations,
        "function_evals": result.n_function_evals,
        "nll": result.neg_log_likelihood,
        "mu": result.mu,
        "sigma": result.sigma,
    }


def fit_with_log_sigma_bounds(data: PairwiseData, config: LogSigmaBoundsConfig):
    result = fit_thurstonian(
        data,
        mu_bounds=DEFAULT_MU_BOUNDS,
        log_sigma_bounds=config.log_sigma_bounds,
        max_iter=1000,
    )
    if not result.converged:
        print(f"[!] {config.name}: {result.termination_message}", end=" ", flush=True)
    return {
        "config": config.name,
        "converged": result.converged,
        "iterations": result.n_iterations,
        "function_evals": result.n_function_evals,
        "nll": result.neg_log_likelihood,
        "mu": result.mu,
        "sigma": result.sigma,
    }


def fit_with_sigma_init(data: PairwiseData, sigma_init: float, log_sigma_bounds: tuple[float, float]):
    result = fit_thurstonian(
        data,
        sigma_init=sigma_init,
        mu_bounds=DEFAULT_MU_BOUNDS,
        log_sigma_bounds=log_sigma_bounds,
        max_iter=1000,
    )
    if not result.converged:
        print(f"[!] sigma_init={sigma_init}: {result.termination_message}", end=" ", flush=True)
    return {
        "sigma_init": sigma_init,
        "converged": result.converged,
        "iterations": result.n_iterations,
        "function_evals": result.n_function_evals,
        "nll": result.neg_log_likelihood,
        "mu": result.mu,
        "sigma": result.sigma,
    }


def load_all_datasets() -> list[tuple[str, PairwiseData]]:
    datasets = []
    if not RESULTS_DIR.exists():
        return datasets

    dirs = [d for d in sorted(RESULTS_DIR.iterdir()) if d.is_dir() and not d.name.startswith("rating_")]
    print(f"Loading {len(dirs)} datasets...")

    # First pass: find common tasks across all datasets
    print("  Finding common tasks across all datasets...", end=" ", flush=True)
    all_task_sets = []
    dir_measurements = {}
    for result_dir in dirs:
        measurements_path = result_dir / "measurements.yaml"
        if not measurements_path.exists():
            continue
        with open(measurements_path) as f:
            measurements = yaml.load(f, Loader=yaml.CSafeLoader)
        if not measurements:
            continue
        dir_measurements[result_dir.name] = measurements
        task_ids = set()
        for m in measurements:
            task_ids.add(m["task_a"])
            task_ids.add(m["task_b"])
        all_task_sets.append(task_ids)

    common_tasks = set.intersection(*all_task_sets) if all_task_sets else set()
    # Take first N_TASKS (sorted for consistency)
    selected_tasks = set(sorted(common_tasks)[:N_TASKS])
    print(f"{len(common_tasks)} common, using {len(selected_tasks)}")

    # Second pass: build datasets with only selected tasks
    for idx, (name, measurements) in enumerate(dir_measurements.items(), 1):
        print(f"  [{idx}/{len(dir_measurements)}] Building {name}...", end=" ", flush=True)

        # Filter to only comparisons between selected tasks
        filtered = [m for m in measurements
                    if m["task_a"] in selected_tasks and m["task_b"] in selected_tasks]
        print(f"{len(filtered)}/{len(measurements)} comparisons")

        tasks = [make_task(tid) for tid in sorted(selected_tasks)]
        id_to_idx = {t.id: i for i, t in enumerate(tasks)}
        n = len(tasks)
        wins = np.zeros((n, n), dtype=np.int32)

        for m in filtered:
            i, j = id_to_idx[m["task_a"]], id_to_idx[m["task_b"]]
            if m["choice"] == "a":
                wins[i, j] += 1
            else:
                wins[j, i] += 1

        datasets.append((name, PairwiseData(tasks=tasks, wins=wins)))

    print(f"Loaded {len(datasets)} datasets with {N_TASKS} tasks each\n")
    return datasets


def analyze_mu_bounds(datasets: list[tuple[str, PairwiseData]], test_frac: float = 0.2, seed: int = 42):
    configs = MU_BOUNDS_CONFIGS
    n_configs = len(configs)
    all_rank_corrs = []
    all_iterations = {c.name: [] for c in configs}
    all_func_evals = {c.name: [] for c in configs}
    all_held_out_nll = {c.name: [] for c in configs}
    all_held_out_acc = {c.name: [] for c in configs}

    print(f"Analyzing mu_bounds on {len(datasets)} datasets ({n_configs} configs each, {test_frac:.0%} held out)...")

    for idx, (name, data) in enumerate(datasets, 1):
        print(f"  [{idx}/{len(datasets)}] {name}: fitting {n_configs} configs...", end=" ", flush=True)

        # Split into train/test
        rng = np.random.default_rng(seed + idx)
        train_wins, test_wins = split_wins(data.wins, test_frac, rng)
        train_data = PairwiseData(tasks=data.tasks, wins=train_wins)

        results = [fit_with_mu_bounds(train_data, c) for c in configs]
        print("done")

        for r, c in zip(results, configs):
            all_iterations[c.name].append(r["iterations"])
            all_func_evals[c.name].append(r["function_evals"])
            all_held_out_nll[c.name].append(eval_held_out_nll(r["mu"], r["sigma"], test_wins))
            all_held_out_acc[c.name].append(eval_held_out_accuracy(r["mu"], r["sigma"], test_wins))

        rank_corr = np.zeros((n_configs, n_configs))
        for i in range(n_configs):
            for j in range(n_configs):
                rank_corr[i, j] = spearmanr(results[i]["mu"], results[j]["mu"]).correlation
        all_rank_corrs.append(rank_corr)

    print()
    return {
        "rank_corrs": all_rank_corrs,
        "iterations": all_iterations,
        "func_evals": all_func_evals,
        "held_out_nll": all_held_out_nll,
        "held_out_acc": all_held_out_acc,
        "n_datasets": len(all_rank_corrs),
        "configs": configs,
    }


def analyze_log_sigma_bounds(datasets: list[tuple[str, PairwiseData]], test_frac: float = 0.2, seed: int = 42):
    configs = LOG_SIGMA_BOUNDS_CONFIGS
    n_configs = len(configs)
    all_rank_corrs = []
    all_iterations = {c.name: [] for c in configs}
    all_func_evals = {c.name: [] for c in configs}
    all_held_out_nll = {c.name: [] for c in configs}
    all_held_out_acc = {c.name: [] for c in configs}

    print(f"Analyzing log_sigma_bounds on {len(datasets)} datasets ({n_configs} configs each, {test_frac:.0%} held out)...")

    for idx, (name, data) in enumerate(datasets, 1):
        print(f"  [{idx}/{len(datasets)}] {name}: fitting {n_configs} configs...", end=" ", flush=True)

        rng = np.random.default_rng(seed + idx)
        train_wins, test_wins = split_wins(data.wins, test_frac, rng)
        train_data = PairwiseData(tasks=data.tasks, wins=train_wins)

        results = [fit_with_log_sigma_bounds(train_data, c) for c in configs]
        print("done")

        for r, c in zip(results, configs):
            all_iterations[c.name].append(r["iterations"])
            all_func_evals[c.name].append(r["function_evals"])
            all_held_out_nll[c.name].append(eval_held_out_nll(r["mu"], r["sigma"], test_wins))
            all_held_out_acc[c.name].append(eval_held_out_accuracy(r["mu"], r["sigma"], test_wins))

        rank_corr = np.zeros((n_configs, n_configs))
        for i in range(n_configs):
            for j in range(n_configs):
                rank_corr[i, j] = spearmanr(results[i]["mu"], results[j]["mu"]).correlation
        all_rank_corrs.append(rank_corr)

    print()
    return {
        "rank_corrs": all_rank_corrs,
        "iterations": all_iterations,
        "func_evals": all_func_evals,
        "held_out_nll": all_held_out_nll,
        "held_out_acc": all_held_out_acc,
        "n_datasets": len(all_rank_corrs),
        "configs": configs,
    }


def analyze_sigma_init(datasets: list[tuple[str, PairwiseData]], log_sigma_bounds: tuple[float, float], test_frac: float = 0.2, seed: int = 42):
    n_sigma = len(SIGMA_INIT_VALUES)
    all_rank_corrs = []
    all_iterations = {s: [] for s in SIGMA_INIT_VALUES}
    all_held_out_nll = {s: [] for s in SIGMA_INIT_VALUES}
    all_held_out_acc = {s: [] for s in SIGMA_INIT_VALUES}

    print(f"Analyzing sigma_init on {len(datasets)} datasets ({n_sigma} values each, {test_frac:.0%} held out)...")

    for idx, (name, data) in enumerate(datasets, 1):
        print(f"  [{idx}/{len(datasets)}] {name}: fitting {n_sigma} configs...", end=" ", flush=True)

        rng = np.random.default_rng(seed + idx)
        train_wins, test_wins = split_wins(data.wins, test_frac, rng)
        train_data = PairwiseData(tasks=data.tasks, wins=train_wins)

        results = [fit_with_sigma_init(train_data, s, log_sigma_bounds) for s in SIGMA_INIT_VALUES]
        print("done")

        for r, s in zip(results, SIGMA_INIT_VALUES):
            all_iterations[s].append(r["iterations"])
            all_held_out_nll[s].append(eval_held_out_nll(r["mu"], r["sigma"], test_wins))
            all_held_out_acc[s].append(eval_held_out_accuracy(r["mu"], r["sigma"], test_wins))

        rank_corr = np.zeros((n_sigma, n_sigma))
        for i in range(n_sigma):
            for j in range(n_sigma):
                rank_corr[i, j] = spearmanr(results[i]["mu"], results[j]["mu"]).correlation
        all_rank_corrs.append(rank_corr)

    print()
    return {
        "rank_corrs": all_rank_corrs,
        "iterations": all_iterations,
        "held_out_nll": all_held_out_nll,
        "held_out_acc": all_held_out_acc,
        "n_datasets": len(all_rank_corrs),
    }


def plot_bounds_analysis(results: dict, param_name: str, output_dir: Path):
    configs = results["configs"]
    n_configs = len(configs)
    labels = [c.short_name for c in configs]
    n_datasets = results["n_datasets"]

    mean_corr = np.mean(results["rank_corrs"], axis=0)
    min_corr = np.min(results["rank_corrs"], axis=0)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))

    # Row 1: Rank correlations
    im0 = axes[0, 0].imshow(mean_corr, cmap='RdYlGn', vmin=0.85, vmax=1.0)
    axes[0, 0].set_xticks(range(n_configs))
    axes[0, 0].set_yticks(range(n_configs))
    axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 0].set_yticklabels(labels)
    for i in range(n_configs):
        for j in range(n_configs):
            color = 'white' if mean_corr[i, j] < 0.92 else 'black'
            axes[0, 0].text(j, i, f'{mean_corr[i, j]:.2f}', ha='center', va='center', fontsize=10, color=color)
    axes[0, 0].set_title("Mean Rank Correlation")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(min_corr, cmap='RdYlGn', vmin=0.85, vmax=1.0)
    axes[0, 1].set_xticks(range(n_configs))
    axes[0, 1].set_yticks(range(n_configs))
    axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 1].set_yticklabels(labels)
    for i in range(n_configs):
        for j in range(n_configs):
            color = 'white' if min_corr[i, j] < 0.92 else 'black'
            axes[0, 1].text(j, i, f'{min_corr[i, j]:.2f}', ha='center', va='center', fontsize=10, color=color)
    axes[0, 1].set_title("Minimum Rank Correlation (worst case)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Row 2: Convergence
    iter_data = [results["iterations"][c.name] for c in configs]
    bp = axes[1, 0].boxplot(iter_data, tick_labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    axes[1, 0].set_ylabel("Iterations")
    axes[1, 0].set_title("Convergence Speed")

    evals_per_iter = []
    for c in configs:
        iters = results["iterations"][c.name]
        fevals = results["func_evals"][c.name]
        ratios = [f / i if i > 0 else 0 for f, i in zip(fevals, iters)]
        evals_per_iter.append(ratios)

    bp2 = axes[1, 1].boxplot(evals_per_iter, tick_labels=labels, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.7)
    axes[1, 1].set_ylabel("Function Evals / Iteration")
    axes[1, 1].set_title("Line Search Efficiency")
    axes[1, 1].axhline(y=2, color='green', linestyle='--', alpha=0.7, label='Ideal (~2)')
    axes[1, 1].legend()

    # Row 3: Held-out validation
    nll_data = [results["held_out_nll"][c.name] for c in configs]
    bp3 = axes[2, 0].boxplot(nll_data, tick_labels=labels, patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('seagreen')
        patch.set_alpha(0.7)
    axes[2, 0].set_ylabel("NLL per comparison")
    axes[2, 0].set_title("Held-out NLL (lower is better)")

    acc_data = [results["held_out_acc"][c.name] for c in configs]
    bp4 = axes[2, 1].boxplot(acc_data, tick_labels=labels, patch_artist=True)
    for patch in bp4['boxes']:
        patch.set_facecolor('purple')
        patch.set_alpha(0.7)
    axes[2, 1].set_ylabel("Accuracy")
    axes[2, 1].set_title("Held-out Accuracy (higher is better)")
    axes[2, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random')
    axes[2, 1].legend()

    plt.suptitle(f"{param_name} Sensitivity (n={n_datasets} templates, N={N_TASKS} tasks, 20% held out)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = f"{param_name.lower().replace(' ', '_')}_sensitivity.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sigma_init_analysis(results: dict, output_dir: Path):
    n_sigma = len(SIGMA_INIT_VALUES)
    labels = [f'{s}' for s in SIGMA_INIT_VALUES]
    n_datasets = results["n_datasets"]

    mean_corr = np.mean(results["rank_corrs"], axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1: Rank correlation and convergence
    im0 = axes[0, 0].imshow(mean_corr, cmap='RdYlGn', vmin=0.95, vmax=1.0)
    axes[0, 0].set_xticks(range(n_sigma))
    axes[0, 0].set_yticks(range(n_sigma))
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_yticklabels(labels)
    for i in range(n_sigma):
        for j in range(n_sigma):
            color = 'white' if mean_corr[i, j] < 0.98 else 'black'
            axes[0, 0].text(j, i, f'{mean_corr[i, j]:.3f}', ha='center', va='center', fontsize=9, color=color)
    axes[0, 0].set_xlabel("σ_init")
    axes[0, 0].set_ylabel("σ_init")
    axes[0, 0].set_title("Mean Rank Correlation")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    iter_data = [results["iterations"][s] for s in SIGMA_INIT_VALUES]
    bp = axes[0, 1].boxplot(iter_data, tick_labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('seagreen')
        patch.set_alpha(0.7)
    axes[0, 1].set_xlabel("σ_init")
    axes[0, 1].set_ylabel("Iterations")
    axes[0, 1].set_title("Convergence Speed")

    off_diag_per_dataset = []
    for rc in results["rank_corrs"]:
        off_diag = rc[np.triu_indices(n_sigma, k=1)]
        off_diag_per_dataset.append(off_diag.min())

    axes[0, 2].hist(off_diag_per_dataset, bins=15, edgecolor='black', alpha=0.7, color='purple')
    axes[0, 2].axvline(np.mean(off_diag_per_dataset), color='red', linestyle='--',
                    label=f'Mean: {np.mean(off_diag_per_dataset):.3f}')
    axes[0, 2].set_xlabel("Min Rank Correlation (per dataset)")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].set_title("Worst-case Stability Distribution")
    axes[0, 2].legend()

    # Row 2: Held-out validation
    nll_data = [results["held_out_nll"][s] for s in SIGMA_INIT_VALUES]
    bp2 = axes[1, 0].boxplot(nll_data, tick_labels=labels, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    axes[1, 0].set_xlabel("σ_init")
    axes[1, 0].set_ylabel("NLL per comparison")
    axes[1, 0].set_title("Held-out NLL (lower is better)")

    acc_data = [results["held_out_acc"][s] for s in SIGMA_INIT_VALUES]
    bp3 = axes[1, 1].boxplot(acc_data, tick_labels=labels, patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.7)
    axes[1, 1].set_xlabel("σ_init")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_title("Held-out Accuracy (higher is better)")
    axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random')
    axes[1, 1].legend()

    # Empty third panel in row 2
    axes[1, 2].axis('off')

    plt.suptitle(f"σ_init Sensitivity (n={n_datasets} templates, N={N_TASKS} tasks, 20% held out)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "sigma_init_sensitivity.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets()
    if len(datasets) < 2:
        print("Not enough datasets found!")
        return

    print("=" * 60)
    print("1/3: MU BOUNDS ANALYSIS")
    print("=" * 60)
    mu_results = analyze_mu_bounds(datasets)
    plot_bounds_analysis(mu_results, "mu_bounds", OUTPUT_DIR)
    print(f"Saved: {OUTPUT_DIR}/mu_bounds_sensitivity.png\n")

    print("=" * 60)
    print("2/3: LOG SIGMA BOUNDS ANALYSIS")
    print("=" * 60)
    log_sigma_results = analyze_log_sigma_bounds(datasets)
    plot_bounds_analysis(log_sigma_results, "log_sigma_bounds", OUTPUT_DIR)
    print(f"Saved: {OUTPUT_DIR}/log_sigma_bounds_sensitivity.png\n")

    print("=" * 60)
    print("3/3: SIGMA INIT ANALYSIS")
    print("=" * 60)
    sigma_init_results = analyze_sigma_init(datasets, log_sigma_bounds=DEFAULT_LOG_SIGMA_BOUNDS)
    plot_sigma_init_analysis(sigma_init_results, OUTPUT_DIR)
    print(f"Saved: {OUTPUT_DIR}/sigma_init_sensitivity.png\n")

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
