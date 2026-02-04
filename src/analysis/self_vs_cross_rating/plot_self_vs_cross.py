"""Analyze self-rating vs cross-rating bias.

Compares:
- ICC (cross-seed consistency) between self and cross rating
- Rating distributions (KL from uniform, mean, std)
- Heatmaps of metrics across model pairs
"""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.measurement.storage.base import load_yaml

EXPERIMENT_DIR = Path("results/experiments/self_vs_cross_v1/post_task_stated")
OUTPUT_DIR = Path("src/analysis/self_vs_cross_rating")


def parse_run_name(run_name: str) -> dict:
    """Parse run name to extract rating model, completion model, and seed."""
    # Pattern: {template}_{rating_model}_regex[_comp_{completion_model}]_cseed{N}_rseed{N}
    # Model names like "gemma-2-27b", "llama-3.1-8b", "llama-3.3-70b"
    model_pattern = r"(?:gemma-\d+-\d+b|llama-\d+\.\d+-\d+b)"

    match = re.match(
        rf"(.+?)_({model_pattern})_regex(?:_comp_({model_pattern}))?_cseed(\d+)_rseed(\d+)",
        run_name
    )
    if not match:
        raise ValueError(f"Cannot parse run name: {run_name}")

    template, rating_model, completion_model, cseed, rseed = match.groups()
    # If no completion_model in name, it's self-rating
    if completion_model is None:
        completion_model = rating_model

    return {
        "template": template,
        "rating_model": rating_model,
        "completion_model": completion_model,
        "completion_seed": int(cseed),
        "rating_seed": int(rseed),
        "is_self": rating_model == completion_model,
    }


def load_experiment_data(experiment_dir: Path) -> pd.DataFrame:
    """Load all measurements into a DataFrame."""
    rows = []
    for run_dir in experiment_dir.iterdir():
        if not run_dir.is_dir():
            continue

        measurements_path = run_dir / "measurements.yaml"
        if not measurements_path.exists():
            continue

        run_info = parse_run_name(run_dir.name)
        measurements = load_yaml(measurements_path)

        for m in measurements:
            score = m["score"]
            # Skip non-numeric scores (e.g., "unclear")
            if not isinstance(score, (int, float)):
                continue
            rows.append({
                **run_info,
                "task_id": m["task_id"],
                "score": float(score),
                "origin": m["origin"],
            })

    return pd.DataFrame(rows)


def compute_icc(scores_by_seed: dict[int, dict[str, float]]) -> float:
    """Compute ICC(2,1) for consistency across seeds.

    Args:
        scores_by_seed: {seed: {task_id: score}}
    """
    seeds = sorted(scores_by_seed.keys())
    if len(seeds) < 2:
        return np.nan

    # Find tasks present in all seeds
    common_tasks = set.intersection(*[set(scores_by_seed[s].keys()) for s in seeds])
    if len(common_tasks) < 2:
        return np.nan

    task_ids = sorted(common_tasks)
    n_tasks = len(task_ids)
    n_seeds = len(seeds)

    # Build matrix: tasks × seeds
    matrix = np.zeros((n_tasks, n_seeds))
    for j, seed in enumerate(seeds):
        for i, task_id in enumerate(task_ids):
            matrix[i, j] = scores_by_seed[seed][task_id]

    # ICC(2,1) calculation
    grand_mean = np.mean(matrix)
    ss_total = np.sum((matrix - grand_mean) ** 2)

    row_means = np.mean(matrix, axis=1)
    ss_rows = n_seeds * np.sum((row_means - grand_mean) ** 2)

    col_means = np.mean(matrix, axis=0)
    ss_cols = n_tasks * np.sum((col_means - grand_mean) ** 2)

    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n_tasks - 1)
    ms_error = ss_error / ((n_tasks - 1) * (n_seeds - 1))
    ms_cols = ss_cols / (n_seeds - 1)

    icc = (ms_rows - ms_error) / (ms_rows + (n_seeds - 1) * ms_error + n_seeds * (ms_cols - ms_error) / n_tasks)
    return icc


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ICC and distribution metrics for each (rating_model, completion_model) pair."""
    results = []

    for (rating, completion), group in df.groupby(["rating_model", "completion_model"]):
        # Get scores by seed for ICC: {seed: {task_id: score}}
        scores_by_seed = {}
        for seed, seed_group in group.groupby("rating_seed"):
            scores_by_seed[seed] = dict(zip(seed_group["task_id"], seed_group["score"]))

        icc = compute_icc(scores_by_seed)

        # Distribution metrics
        all_scores = group["score"].values
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)

        # KL from uniform (discretize to bins)
        bins = np.arange(-0.5, 6, 1)  # 0-5 scale
        hist, _ = np.histogram(all_scores, bins=bins, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        uniform = np.ones_like(hist) / len(hist)
        kl_uniform = stats.entropy(hist, uniform)

        results.append({
            "rating_model": rating,
            "completion_model": completion,
            "is_self": rating == completion,
            "icc": icc,
            "mean": mean_score,
            "std": std_score,
            "kl_from_uniform": kl_uniform,
            "n_measurements": len(group),
        })

    return pd.DataFrame(results)


def plot_heatmaps(metrics_df: pd.DataFrame, output_dir: Path):
    """Plot heatmaps of ICC and KL across model pairs."""
    models = sorted(metrics_df["rating_model"].unique())

    # Create matrices
    icc_matrix = np.full((len(models), len(models)), np.nan)
    kl_matrix = np.full((len(models), len(models)), np.nan)
    mean_matrix = np.full((len(models), len(models)), np.nan)

    for _, row in metrics_df.iterrows():
        i = models.index(row["rating_model"])
        j = models.index(row["completion_model"])
        icc_matrix[i, j] = row["icc"]
        kl_matrix[i, j] = row["kl_from_uniform"]
        mean_matrix[i, j] = row["mean"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ICC heatmap
    sns.heatmap(icc_matrix, ax=axes[0], annot=True, fmt=".2f",
                xticklabels=models, yticklabels=models, cmap="RdYlGn", vmin=0, vmax=1)
    axes[0].set_title("ICC (Cross-Seed Consistency)")
    axes[0].set_xlabel("Completion Model")
    axes[0].set_ylabel("Rating Model")

    # KL heatmap
    sns.heatmap(kl_matrix, ax=axes[1], annot=True, fmt=".2f",
                xticklabels=models, yticklabels=models, cmap="RdYlGn_r")
    axes[1].set_title("KL from Uniform (Lower = More Uniform)")
    axes[1].set_xlabel("Completion Model")
    axes[1].set_ylabel("Rating Model")

    # Mean heatmap
    sns.heatmap(mean_matrix, ax=axes[2], annot=True, fmt=".2f",
                xticklabels=models, yticklabels=models, cmap="coolwarm")
    axes[2].set_title("Mean Rating")
    axes[2].set_xlabel("Completion Model")
    axes[2].set_ylabel("Rating Model")

    plt.tight_layout()

    from datetime import datetime
    date_str = datetime.now().strftime("%m%d%y")
    output_path = output_dir / f"plot_{date_str}_heatmaps.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_self_vs_cross_comparison(metrics_df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing self vs cross rating metrics."""
    self_df = metrics_df[metrics_df["is_self"]]
    cross_df = metrics_df[~metrics_df["is_self"]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # ICC comparison
    self_icc = self_df["icc"].values
    cross_icc = cross_df.groupby("rating_model")["icc"].mean().values

    x = np.arange(len(self_df))
    width = 0.35
    axes[0].bar(x - width/2, self_icc, width, label="Self-rating", color="steelblue")
    axes[0].bar(x + width/2, cross_icc, width, label="Cross-rating (avg)", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(self_df["rating_model"], rotation=45, ha="right")
    axes[0].set_ylabel("ICC")
    axes[0].set_title("Cross-Seed Consistency")
    axes[0].legend()

    # Mean comparison
    self_mean = self_df["mean"].values
    cross_mean = cross_df.groupby("rating_model")["mean"].mean().values
    axes[1].bar(x - width/2, self_mean, width, label="Self-rating", color="steelblue")
    axes[1].bar(x + width/2, cross_mean, width, label="Cross-rating (avg)", color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(self_df["rating_model"], rotation=45, ha="right")
    axes[1].set_ylabel("Mean Rating")
    axes[1].set_title("Mean Rating Score")
    axes[1].legend()

    # KL comparison
    self_kl = self_df["kl_from_uniform"].values
    cross_kl = cross_df.groupby("rating_model")["kl_from_uniform"].mean().values
    axes[2].bar(x - width/2, self_kl, width, label="Self-rating", color="steelblue")
    axes[2].bar(x + width/2, cross_kl, width, label="Cross-rating (avg)", color="coral")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(self_df["rating_model"], rotation=45, ha="right")
    axes[2].set_ylabel("KL from Uniform")
    axes[2].set_title("Scale Usage (Lower = More Uniform)")
    axes[2].legend()

    plt.tight_layout()

    from datetime import datetime
    date_str = datetime.now().strftime("%m%d%y")
    output_path = output_dir / f"plot_{date_str}_self_vs_cross.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def print_summary(metrics_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SELF VS CROSS RATING SUMMARY")
    print("="*60)

    self_df = metrics_df[metrics_df["is_self"]]
    cross_df = metrics_df[~metrics_df["is_self"]]

    print(f"\nSelf-rating (n={len(self_df)}):")
    print(f"  ICC:  {self_df['icc'].mean():.3f} ± {self_df['icc'].std():.3f}")
    print(f"  Mean: {self_df['mean'].mean():.3f} ± {self_df['mean'].std():.3f}")
    print(f"  KL:   {self_df['kl_from_uniform'].mean():.3f} ± {self_df['kl_from_uniform'].std():.3f}")

    print(f"\nCross-rating (n={len(cross_df)}):")
    print(f"  ICC:  {cross_df['icc'].mean():.3f} ± {cross_df['icc'].std():.3f}")
    print(f"  Mean: {cross_df['mean'].mean():.3f} ± {cross_df['mean'].std():.3f}")
    print(f"  KL:   {cross_df['kl_from_uniform'].mean():.3f} ± {cross_df['kl_from_uniform'].std():.3f}")

    # Statistical tests
    print("\nStatistical Tests (Self vs Cross):")
    for metric in ["icc", "mean", "kl_from_uniform"]:
        t_stat, p_val = stats.ttest_ind(self_df[metric], cross_df[metric])
        print(f"  {metric}: t={t_stat:.2f}, p={p_val:.3f}")

    print("\n" + "="*60)
    print("PER-MODEL BREAKDOWN")
    print("="*60)
    print(metrics_df.to_string(index=False))


def main():
    print("Loading experiment data...")
    df = load_experiment_data(EXPERIMENT_DIR)
    print(f"Loaded {len(df)} measurements")

    print("\nComputing metrics...")
    metrics_df = compute_metrics(df)

    print_summary(metrics_df)

    print("\nGenerating plots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(metrics_df, OUTPUT_DIR)
    plot_self_vs_cross_comparison(metrics_df, OUTPUT_DIR)

    # Save metrics CSV
    from datetime import datetime
    date_str = datetime.now().strftime("%m%d%y")
    csv_path = OUTPUT_DIR / f"metrics_{date_str}.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
