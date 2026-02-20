"""Analyze temperature sweep results: consistency, ranking stability, Thurstonian fit, probe R², entropy."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.fitting.thurstonian_fitting.thurstonian import PairwiseData, fit_thurstonian
from src.measurement.storage.loading import load_yaml
from src.probes.core.activations import load_activations
from src.probes.core.linear_probe import train_and_evaluate
from src.probes.data_loading import load_thurstonian_scores
from src.task_data import Task, OriginDataset

TEMPERATURES = [0.3, 0.5, 0.7, 1.0, 1.3]
TEMP_LABELS = ["0.3", "0.5", "0.7", "1.0", "1.3"]
TEMP_DIRS = ["temp_03", "temp_05", "temp_07", "temp_10", "temp_13"]

BASE_DIR = Path("results/experiments/temperature_sweep")
ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
EVAL_RUN_DIR = Path(
    "results/experiments/gemma3_4k_pre_task/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0"
)
ASSETS_DIR = Path("experiments/temperature_calibration/assets")
LAYER = 31
EVAL_SPLIT_SEED = 42


def find_run_dir(temp_dir: str) -> Path:
    """Find the single run directory under a temperature's AL results."""
    al_dir = BASE_DIR / temp_dir / "pre_task_active_learning"
    run_dirs = [d for d in al_dir.iterdir() if d.is_dir()]
    if len(run_dirs) != 1:
        raise ValueError(f"Expected 1 run dir in {al_dir}, found {len(run_dirs)}: {run_dirs}")
    return run_dirs[0]


def load_measurements(run_dir: Path) -> list[dict]:
    return load_yaml(run_dir / "measurements.yaml")


def compute_choice_consistency(measurements: list[dict]) -> float:
    """Per pair: fraction of samples agreeing with majority. Averaged across pairs."""
    pair_choices: dict[tuple[str, str], list[str]] = {}
    for m in measurements:
        key = (m["task_a"], m["task_b"])
        if key not in pair_choices:
            pair_choices[key] = []
        pair_choices[key].append(m["choice"])

    consistencies = []
    for choices in pair_choices.values():
        n = len(choices)
        if n == 0:
            continue
        majority_count = max(choices.count("a"), choices.count("b"))
        consistencies.append(majority_count / n)

    return float(np.mean(consistencies))


def compute_choice_entropy(measurements: list[dict]) -> float:
    """Mean binary entropy across pairs. H(p) = -p*log2(p) - (1-p)*log2(1-p)."""
    pair_choices: dict[tuple[str, str], list[str]] = {}
    for m in measurements:
        key = (m["task_a"], m["task_b"])
        if key not in pair_choices:
            pair_choices[key] = []
        pair_choices[key].append(m["choice"])

    entropies = []
    for choices in pair_choices.values():
        n = len(choices)
        if n == 0:
            continue
        p = choices.count("a") / n
        if p == 0.0 or p == 1.0:
            entropies.append(0.0)
        else:
            h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            entropies.append(h)

    return float(np.mean(entropies))


def fit_thurstonian_from_measurements(measurements: list[dict]) -> dict:
    """Fit Thurstonian model, return {task_id: mu} and fit stats."""
    # Collect unique tasks
    task_ids_seen: dict[str, str] = {}  # id -> origin
    for m in measurements:
        task_ids_seen[m["task_a"]] = m["origin_a"]
        task_ids_seen[m["task_b"]] = m["origin_b"]

    tasks = [
        Task(id=tid, prompt="", origin=OriginDataset[origin], metadata={})
        for tid, origin in sorted(task_ids_seen.items())
    ]

    from src.types import BinaryPreferenceMeasurement, PreferenceType

    comparisons = [
        BinaryPreferenceMeasurement(
            task_a=Task(id=m["task_a"], prompt="", origin=OriginDataset[m["origin_a"]], metadata={}),
            task_b=Task(id=m["task_b"], prompt="", origin=OriginDataset[m["origin_b"]], metadata={}),
            choice=m["choice"],
            preference_type=PreferenceType.PRE_TASK_REVEALED,
        )
        for m in measurements
    ]

    pairwise_data = PairwiseData.from_comparisons(comparisons, tasks)
    result = fit_thurstonian(pairwise_data)

    scores = {t.id: float(result.mu[i]) for i, t in enumerate(result.tasks)}
    stats = {
        "neg_log_likelihood": float(result.neg_log_likelihood),
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "sigma_mean": float(result.sigma.mean()),
        "sigma_std": float(result.sigma.std()),
        "sigma_max": float(result.sigma.max()),
    }
    return {"scores": scores, "stats": stats, "result": result}


def train_probe_heldout(
    train_scores: dict[str, float],
    eval_scores: dict[str, float],
) -> dict:
    """Train Ridge probe on train_scores, evaluate on eval_scores (split into sweep/final)."""
    task_ids, activations_by_layer = load_activations(
        ACTIVATIONS_PATH,
        task_id_filter=set(train_scores) | set(eval_scores),
        layers=[LAYER],
    )
    activations = activations_by_layer[LAYER]
    tid_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    # Build train arrays
    train_indices = [tid_to_idx[tid] for tid in train_scores if tid in tid_to_idx]
    train_labels = np.array([train_scores[task_ids[i]] for i in train_indices])
    X_train = activations[train_indices]

    # Split eval into sweep/final
    rng = np.random.default_rng(EVAL_SPLIT_SEED)
    eval_tids = sorted(tid for tid in eval_scores if tid in tid_to_idx)
    perm = rng.permutation(len(eval_tids))
    half = len(eval_tids) // 2
    sweep_tids = [eval_tids[i] for i in perm[:half]]
    final_tids = [eval_tids[i] for i in perm[half:]]

    sweep_indices = [tid_to_idx[tid] for tid in sweep_tids]
    sweep_labels = np.array([eval_scores[tid] for tid in sweep_tids])
    final_indices = [tid_to_idx[tid] for tid in final_tids]
    final_labels = np.array([eval_scores[tid] for tid in final_tids])

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_sweep_s = scaler.transform(activations[sweep_indices])
    X_final_s = scaler.transform(activations[final_indices])

    # Alpha sweep on sweep half
    alphas = np.logspace(-1, 5, 20)
    best_alpha = alphas[0]
    best_sweep_r = -1.0
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_s, train_labels)
        y_pred = ridge.predict(X_sweep_s)
        r, _ = pearsonr(sweep_labels, y_pred)
        if r > best_sweep_r:
            best_sweep_r = float(r)
            best_alpha = float(alpha)

    # Final eval at best alpha
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train_s, train_labels)
    y_pred_final = ridge.predict(X_final_s)
    final_r, _ = pearsonr(final_labels, y_pred_final)
    final_r2 = float(final_r) ** 2

    return {
        "best_alpha": best_alpha,
        "sweep_r": best_sweep_r,
        "final_r": float(final_r),
        "final_r2": final_r2,
        "n_train": len(train_indices),
        "n_sweep": len(sweep_tids),
        "n_final": len(final_tids),
    }


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Load eval scores (4K heldout set)
    eval_scores = load_thurstonian_scores(EVAL_RUN_DIR)
    print(f"Loaded {len(eval_scores)} eval scores from 4K heldout set")

    # Process each temperature
    results = {}
    for temp, temp_dir, label in zip(TEMPERATURES, TEMP_DIRS, TEMP_LABELS):
        print(f"\n{'='*60}")
        print(f"Temperature {label}")
        print(f"{'='*60}")

        run_dir = find_run_dir(temp_dir)
        measurements = load_measurements(run_dir)
        print(f"  Loaded {len(measurements)} measurements from {run_dir}")

        consistency = compute_choice_consistency(measurements)
        entropy = compute_choice_entropy(measurements)
        print(f"  Choice consistency: {consistency:.4f}")
        print(f"  Choice entropy: {entropy:.4f}")

        thurst = fit_thurstonian_from_measurements(measurements)
        stats = thurst["stats"]
        print(f"  Thurstonian NLL: {stats['neg_log_likelihood']:.2f}")
        print(f"  Converged: {stats['converged']}")
        print(f"  Sigma mean={stats['sigma_mean']:.4f}, std={stats['sigma_std']:.4f}, max={stats['sigma_max']:.4f}")

        probe = train_probe_heldout(thurst["scores"], eval_scores)
        print(f"  Probe R²: {probe['final_r2']:.4f} (r={probe['final_r']:.4f})")
        print(f"  Alpha: {probe['best_alpha']:.4g}, n_train={probe['n_train']}, n_eval={probe['n_final']}")

        results[label] = {
            "temperature": temp,
            "consistency": consistency,
            "entropy": entropy,
            "thurstonian": stats,
            "probe": probe,
            "scores": thurst["scores"],
        }

    # Ranking stability matrix (Spearman correlations between all pairs)
    print(f"\n{'='*60}")
    print("Ranking stability (Spearman correlation)")
    print(f"{'='*60}")

    stability_matrix = np.zeros((len(TEMPERATURES), len(TEMPERATURES)))
    for i, li in enumerate(TEMP_LABELS):
        for j, lj in enumerate(TEMP_LABELS):
            if i == j:
                stability_matrix[i, j] = 1.0
                continue
            scores_i = results[li]["scores"]
            scores_j = results[lj]["scores"]
            common_ids = sorted(set(scores_i) & set(scores_j))
            vals_i = [scores_i[tid] for tid in common_ids]
            vals_j = [scores_j[tid] for tid in common_ids]
            rho, _ = spearmanr(vals_i, vals_j)
            stability_matrix[i, j] = rho

    # Print matrix
    header = "      " + "  ".join(f"T={l:>4s}" for l in TEMP_LABELS)
    print(header)
    for i, li in enumerate(TEMP_LABELS):
        row = f"T={li:>4s}"
        for j in range(len(TEMP_LABELS)):
            row += f"  {stability_matrix[i, j]:.3f}"
        print(row)

    # Generate plots
    _plot_metrics_vs_temperature(results)
    _plot_stability_heatmap(stability_matrix)
    _plot_sigma_distributions(results)

    # Summary table
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Temp':>6s} {'Consist':>8s} {'Entropy':>8s} {'NLL':>10s} {'Probe R²':>9s} {'Probe r':>8s}")
    for label in TEMP_LABELS:
        r = results[label]
        print(
            f"{label:>6s} "
            f"{r['consistency']:>8.4f} "
            f"{r['entropy']:>8.4f} "
            f"{r['thurstonian']['neg_log_likelihood']:>10.2f} "
            f"{r['probe']['final_r2']:>9.4f} "
            f"{r['probe']['final_r']:>8.4f}"
        )


def _plot_metrics_vs_temperature(results: dict):
    """4-panel plot: consistency, entropy, NLL, probe R² vs temperature."""
    temps = TEMPERATURES
    consistency = [results[l]["consistency"] for l in TEMP_LABELS]
    entropy = [results[l]["entropy"] for l in TEMP_LABELS]
    nll = [results[l]["thurstonian"]["neg_log_likelihood"] for l in TEMP_LABELS]
    probe_r2 = [results[l]["probe"]["final_r2"] for l in TEMP_LABELS]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(temps, consistency, "o-", color="tab:blue")
    axes[0, 0].set_xlabel("Temperature")
    axes[0, 0].set_ylabel("Choice consistency")
    axes[0, 0].set_ylim(0.5, 1.0)
    axes[0, 0].set_title("Choice consistency")

    axes[0, 1].plot(temps, entropy, "o-", color="tab:orange")
    axes[0, 1].set_xlabel("Temperature")
    axes[0, 1].set_ylabel("Mean binary entropy")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_title("Choice entropy")

    axes[1, 0].plot(temps, nll, "o-", color="tab:green")
    axes[1, 0].set_xlabel("Temperature")
    axes[1, 0].set_ylabel("Negative log-likelihood")
    axes[1, 0].set_title("Thurstonian NLL")

    axes[1, 1].plot(temps, probe_r2, "o-", color="tab:red")
    axes[1, 1].set_xlabel("Temperature")
    axes[1, 1].set_ylabel("Heldout R²")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title("Probe R² (L31, heldout eval)")

    fig.suptitle("Temperature calibration — Gemma 3 27B", fontsize=14)
    fig.tight_layout()
    out = ASSETS_DIR / "plot_022026_metrics_vs_temperature.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def _plot_stability_heatmap(matrix: np.ndarray):
    """Heatmap of Spearman correlations between temperature conditions."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(TEMP_LABELS)))
    ax.set_xticklabels([f"T={l}" for l in TEMP_LABELS])
    ax.set_yticks(range(len(TEMP_LABELS)))
    ax.set_yticklabels([f"T={l}" for l in TEMP_LABELS])
    for i in range(len(TEMP_LABELS)):
        for j in range(len(TEMP_LABELS)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_title("Utility ranking stability across temperatures")
    fig.tight_layout()
    out = ASSETS_DIR / "plot_022026_ranking_stability_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def _plot_sigma_distributions(results: dict):
    """Box plot of Thurstonian sigma distributions across temperatures."""
    fig, ax = plt.subplots(figsize=(8, 5))
    temps = TEMPERATURES
    sigma_means = [results[l]["thurstonian"]["sigma_mean"] for l in TEMP_LABELS]
    sigma_stds = [results[l]["thurstonian"]["sigma_std"] for l in TEMP_LABELS]

    ax.errorbar(temps, sigma_means, yerr=sigma_stds, fmt="o-", capsize=5, color="tab:purple")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Sigma (mean +/- std)")
    ax.set_ylim(0, None)
    ax.set_title("Thurstonian sigma distribution vs temperature")
    fig.tight_layout()
    out = ASSETS_DIR / "plot_022026_sigma_vs_temperature.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
