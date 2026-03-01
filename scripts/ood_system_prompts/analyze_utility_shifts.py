"""Analyze how system prompts shift Thurstonian utilities relative to baseline.

Loads per-condition utility fits, computes Δu = u_condition - u_baseline,
assigns ground truth labels, and generates specificity plots.

Usage: python -m scripts.ood_system_prompts.analyze_utility_shifts
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.measurement.storage.loading import load_run_utilities
from scripts.ood_system_prompts.analyze_ground_truth import (
    _ground_truth_exp1b,
    _ground_truth_exp1c,
    _ground_truth_exp1d,
    _load_condition_metadata,
)

REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS = REPO_ROOT / "results" / "experiments"
AL_CONFIGS = REPO_ROOT / "configs" / "measurement" / "active_learning"
OOD_CONFIGS = REPO_ROOT / "configs" / "ood"
ASSETS = REPO_ROOT / "experiments" / "ood_system_prompts" / "utility_fitting" / "assets"

EXPERIMENTS = {
    "exp1b": {
        "results_dir": RESULTS / "ood_exp1b" / "pre_task_active_learning",
        "configs_dir": AL_CONFIGS / "ood_exp1b",
        "prompt_config": OOD_CONFIGS / "prompts" / "targeted_preference.json",
        "gt_fn": _ground_truth_exp1b,
    },
    "exp1c": {
        "results_dir": RESULTS / "ood_exp1c" / "pre_task_active_learning",
        "configs_dir": AL_CONFIGS / "ood_exp1c",
        "prompt_config": OOD_CONFIGS / "prompts" / "targeted_preference.json",
        "gt_fn": _ground_truth_exp1c,
    },
    "exp1d": {
        "results_dir": RESULTS / "ood_exp1d" / "pre_task_active_learning",
        "configs_dir": AL_CONFIGS / "ood_exp1d",
        "prompt_config": OOD_CONFIGS / "prompts" / "competing_preference.json",
        "gt_fn": _ground_truth_exp1d,
    },
}


def _build_sysprompt_to_condition(configs_dir: Path) -> dict[str, str]:
    """Map system_prompt text → config filename stem (condition name)."""
    mapping = {}
    for cfg_path in configs_dir.glob("*.yaml"):
        if cfg_path.stem == "baseline":
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        sp = cfg.get("measurement_system_prompt", "")
        if sp:
            mapping[sp.strip()] = cfg_path.stem
    return mapping


def _identify_condition(result_dir: Path, sp_to_cond: dict[str, str]) -> str | None:
    """Identify condition name from a result directory."""
    # Baseline has no _sys suffix
    if "_sys" not in result_dir.name:
        return "baseline"
    al_path = result_dir / "active_learning.yaml"
    if not al_path.exists():
        return None
    with open(al_path) as f:
        al = yaml.safe_load(f)
    sp = al.get("system_prompt", "").strip()
    if not sp:
        return None
    # Try exact match first, then prefix match (YAML may truncate)
    if sp in sp_to_cond:
        return sp_to_cond[sp]
    for full_sp, cond_name in sp_to_cond.items():
        if full_sp.startswith(sp) or sp.startswith(full_sp):
            return cond_name
    return None


def load_experiment_utilities(exp_name: str) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Load utilities for all conditions in an experiment.

    Returns (DataFrame with task_id index and condition columns, dir mapping).
    """
    cfg = EXPERIMENTS[exp_name]
    sp_to_cond = _build_sysprompt_to_condition(cfg["configs_dir"])

    run_dirs: dict[str, Path] = {}
    for d in sorted(cfg["results_dir"].iterdir()):
        if not d.is_dir():
            continue
        cond = _identify_condition(d, sp_to_cond)
        if cond is None:
            print(f"  WARNING: could not identify condition for {d.name}")
            continue
        run_dirs[cond] = d

    if "baseline" not in run_dirs:
        raise FileNotFoundError(f"No baseline found for {exp_name}")

    # Load utilities
    series = {}
    for cond, d in run_dirs.items():
        mu, task_ids = load_run_utilities(d)
        series[cond] = pd.Series(mu, index=task_ids, name=cond)

    df = pd.DataFrame(series).dropna()

    # Demean each condition to remove arbitrary Thurstonian location constant
    for col in df.columns:
        df[col] -= df[col].mean()

    print(f"  {exp_name}: {len(df)} tasks × {len(df.columns) - 1} conditions + baseline")
    return df, run_dirs


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Δu = u_condition - u_baseline for each condition."""
    conditions = [c for c in df.columns if c != "baseline"]
    deltas = df[conditions].sub(df["baseline"], axis=0)
    return deltas


def assign_ground_truth(
    deltas: pd.DataFrame, exp_name: str
) -> pd.DataFrame:
    """Assign ground truth labels per (condition, task) pair.

    Returns DataFrame with same shape as deltas, values in {-1, 0, +1}.
    """
    gt_fn = EXPERIMENTS[exp_name]["gt_fn"]
    task_ids = list(deltas.index)
    conditions = list(deltas.columns)

    # Build flattened arrays matching gt_fn signature
    gt_df = pd.DataFrame(0.0, index=deltas.index, columns=deltas.columns)
    for cond in conditions:
        cond_labels = np.array([cond] * len(task_ids))
        gt_vals = gt_fn(cond_labels, task_ids)
        gt_df[cond] = gt_vals
    return gt_df


def compute_summary_stats(
    deltas: pd.DataFrame, gt_df: pd.DataFrame
) -> pd.DataFrame:
    """Per-condition summary: mean Δu for on-target (+1 and -1) and mean |Δu| for off-target."""
    rows = []
    for cond in deltas.columns:
        du = deltas[cond].values
        gt = gt_df[cond].values

        on_pos = gt == 1
        on_neg = gt == -1
        off = gt == 0

        rows.append({
            "condition": cond,
            "mean_du_pos": float(np.mean(du[on_pos])) if on_pos.any() else np.nan,
            "mean_du_neg": float(np.mean(du[on_neg])) if on_neg.any() else np.nan,
            "n_pos": int(on_pos.sum()),
            "n_neg": int(on_neg.sum()),
            "mean_abs_du_off": float(np.mean(np.abs(du[off]))) if off.any() else np.nan,
            "std_du_off": float(np.std(du[off])) if off.any() else np.nan,
            "n_off": int(off.sum()),
            "mean_du_on": float(np.mean(du[gt != 0])) if (gt != 0).any() else np.nan,
            "sem_du_on": (
                float(np.std(du[gt != 0]) / np.sqrt((gt != 0).sum()))
                if (gt != 0).sum() > 1 else np.nan
            ),
        })
    return pd.DataFrame(rows)


# ── Plot 1: Per-condition bar chart of mean on-target Δu ──


def plot_condition_bars(
    deltas: pd.DataFrame, gt_df: pd.DataFrame, stats: pd.DataFrame, exp_name: str
) -> Path:
    conditions = list(deltas.columns)

    # Compute mean Δu for on-target tasks per condition (signed, gt-weighted)
    means = []
    sems = []
    for cond in conditions:
        du = deltas[cond].values
        gt = gt_df[cond].values
        on_mask = gt != 0
        # Weight by gt sign so pos tasks contribute positively and neg contribute negatively
        weighted = du[on_mask] * gt[on_mask]
        means.append(float(np.mean(weighted)) if on_mask.any() else 0.0)
        sems.append(
            float(np.std(weighted) / np.sqrt(on_mask.sum()))
            if on_mask.sum() > 1 else 0.0
        )

    # Sort by mean
    order = np.argsort(means)[::-1]
    sorted_conds = [conditions[i] for i in order]
    sorted_means = [means[i] for i in order]
    sorted_sems = [sems[i] for i in order]
    colors = ["#4477AA" if m >= 0 else "#CC6677" for m in sorted_means]

    # Off-target noise floor
    off_stds = stats["std_du_off"].dropna().values
    noise_sd = float(np.mean(off_stds)) if len(off_stds) > 0 else 0

    fig, ax = plt.subplots(figsize=(max(10, len(conditions) * 0.6), 5))
    x = np.arange(len(sorted_conds))
    ax.bar(x, sorted_means, color=colors, edgecolor="none", width=0.7)
    ax.errorbar(x, sorted_means, yerr=sorted_sems, fmt="none", ecolor="black",
                capsize=3, linewidth=1)

    if noise_sd > 0:
        ax.axhspan(-noise_sd, noise_sd, color="gray", alpha=0.15, label="±1 SD off-target")
        ax.legend(loc="upper right", fontsize=9)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_persona", "").replace("compete_", "") for c in sorted_conds],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean gt-weighted Δu (on-target)")
    ax.set_title(f"{exp_name}: On-target utility shift by condition")
    ax.set_ylim(bottom=min(-0.5, min(sorted_means) - 0.3))
    fig.tight_layout()

    path = ASSETS / f"plot_022726_utility_shift_{exp_name}_conditions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")
    return path


# ── Plot 2: Task-level bar chart for representative conditions ──


def _pick_representative_conditions(stats: pd.DataFrame, exp_name: str) -> list[str]:
    """Pick 2-4 representative conditions: strongest pos, strongest neg, maybe one weak."""
    s = stats.copy()
    s["on_effect"] = s["mean_du_on"].fillna(0)

    # Strongest positive on-target shift (condition where mean Δu on-target is most positive)
    pos_rows = s[s["mean_du_pos"].notna() & (s["n_pos"] > 0)]
    neg_rows = s[s["mean_du_neg"].notna() & (s["n_neg"] > 0)]

    picks = []
    if not pos_rows.empty:
        best_pos = pos_rows.loc[pos_rows["mean_du_pos"].idxmax(), "condition"]
        picks.append(best_pos)
    if not neg_rows.empty:
        best_neg = neg_rows.loc[neg_rows["mean_du_neg"].idxmin(), "condition"]
        picks.append(best_neg)

    # Add a weak condition (smallest absolute on-target effect)
    remaining = s[~s["condition"].isin(picks)]
    if not remaining.empty:
        weakest = remaining.loc[remaining["on_effect"].abs().idxmin(), "condition"]
        picks.append(weakest)

    return picks[:4]


def plot_task_bars(
    deltas: pd.DataFrame, gt_df: pd.DataFrame, cond: str, exp_name: str
) -> Path:
    du = deltas[cond].values
    gt = gt_df[cond].values
    task_ids = list(deltas.index)

    order = np.argsort(du)[::-1]
    sorted_du = du[order]
    sorted_gt = gt[order]
    sorted_ids = [task_ids[i] for i in order]

    colors = []
    for g in sorted_gt:
        if g != 0:
            colors.append("#228833")  # on-target green
        else:
            colors.append("#BBBBBB")  # off-target gray

    fig, ax = plt.subplots(figsize=(max(10, len(sorted_du) * 0.25), 5))
    x = np.arange(len(sorted_du))
    ax.bar(x, sorted_du, color=colors, edgecolor="none", width=0.8)
    ax.axhline(0, color="black", linewidth=0.5)

    # Simplify task labels
    short_labels = [tid.replace("hidden_", "").replace("crossed_", "") for tid in sorted_ids]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=90, fontsize=6)
    ax.set_ylabel("Δu (condition − baseline)")

    cond_label = cond.replace("_persona", "").replace("compete_", "")
    ax.set_title(f"{exp_name} / {cond_label}: per-task utility shift")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#228833", label="On-target"),
        Patch(facecolor="#BBBBBB", label="Off-target"),
    ], loc="upper right", fontsize=9)

    fig.tight_layout()
    cond_short = cond.replace("_persona", "")
    path = ASSETS / f"plot_022726_utility_shift_{exp_name}_tasks_{cond_short}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")
    return path


# ── Plot 3 (exp1d only): Competing resolution ──


def plot_competing_resolution(deltas: pd.DataFrame, exp_name: str) -> Path | None:
    if exp_name != "exp1d":
        return None

    cond_meta = _load_condition_metadata(OOD_CONFIGS / "prompts" / "competing_preference.json")
    tasks = json.load(open(OOD_CONFIGS / "tasks" / "crossed_tasks.json"))
    task_lookup = {t["task_id"]: t for t in tasks}

    # For each condition, classify tasks into topic-only, shell-only, conflicted
    rows = []
    for cond in deltas.columns:
        meta = cond_meta[cond]
        subject = meta["subject"]
        task_type = meta["task_type"]
        direction = meta["direction"]

        topic_only_du, shell_only_du, conflicted_du = [], [], []

        for tid in deltas.index:
            task = task_lookup.get(tid)
            if task is None:
                continue
            du = deltas.loc[tid, cond]
            topic_match = task["topic"] == subject
            shell_match = task["category_shell"] == task_type

            if topic_match and shell_match:
                conflicted_du.append(du)
            elif topic_match:
                topic_only_du.append(du)
            elif shell_match:
                shell_only_du.append(du)

        if not conflicted_du:
            continue

        pair_label = f"{subject}/{task_type}"
        dir_label = "topic+" if direction == "love_subject" else "shell+"

        rows.append({
            "pair": pair_label,
            "direction": dir_label,
            "cond": cond,
            "mean_topic_only": float(np.mean(topic_only_du)) if topic_only_du else np.nan,
            "mean_shell_only": float(np.mean(shell_only_du)) if shell_only_du else np.nan,
            "mean_conflicted": float(np.mean(conflicted_du)),
            "n_conflicted": len(conflicted_du),
        })

    if not rows:
        return None

    rdf = pd.DataFrame(rows)

    # Group by pair, show topic+/shell+ side by side
    pairs = sorted(rdf["pair"].unique())
    n_pairs = len(pairs)

    fig, axes = plt.subplots(1, n_pairs, figsize=(n_pairs * 3.5, 5), sharey=True)
    if n_pairs == 1:
        axes = [axes]

    bar_colors = {"topic-only": "#4477AA", "shell-only": "#CC6677", "conflicted": "#DDCC77"}

    for ax, pair in zip(axes, pairs):
        pair_rows = rdf[rdf["pair"] == pair].sort_values("direction")
        x = np.arange(len(pair_rows))
        width = 0.25

        for offset, (col, label) in enumerate([
            ("mean_topic_only", "topic-only"),
            ("mean_shell_only", "shell-only"),
            ("mean_conflicted", "conflicted"),
        ]):
            vals = pair_rows[col].values
            ax.bar(x + (offset - 1) * width, vals, width=width,
                   color=bar_colors[label], label=label if ax == axes[0] else None)

        ax.set_xticks(x)
        ax.set_xticklabels(pair_rows["direction"].values, fontsize=9)
        ax.set_title(pair, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.5)

    axes[0].set_ylabel("Mean Δu")
    fig.legend(loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Exp 1d: Conflicted vs pure tasks", fontsize=12, y=1.06)
    fig.tight_layout()

    path = ASSETS / "plot_022726_utility_shift_exp1d_competing.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")
    return path


# ── Plot 4: Per-persona topic utility grid ──


def _task_id_to_topic(task_id: str, exp_name: str) -> str | None:
    """Extract topic from task_id based on experiment."""
    if exp_name == "exp1b":
        # hidden_{topic}_{n}
        parts = task_id.replace("hidden_", "").rsplit("_", 1)
        return parts[0] if len(parts) == 2 else None
    elif exp_name in ("exp1c", "exp1d"):
        # crossed_{topic}_{shell}
        parts = task_id.replace("crossed_", "").rsplit("_", 1)
        return parts[0] if len(parts) == 2 else None
    return None


def plot_topic_utilities_per_persona(
    df: pd.DataFrame, exp_name: str
) -> Path:
    """Grid of subplots: one per persona, bars = mean utility per topic, annotated with Δu."""
    conditions = sorted([c for c in df.columns if c != "baseline"])
    task_ids = list(df.index)

    # Build topic grouping
    topic_for_task = {}
    for tid in task_ids:
        t = _task_id_to_topic(tid, exp_name)
        if t is not None:
            topic_for_task[tid] = t
    topics = sorted(set(topic_for_task.values()))

    # Compute per-topic mean utility under baseline (already demeaned at load time)
    baseline_topic_means = {}
    for topic in topics:
        tids = [t for t in task_ids if topic_for_task.get(t) == topic]
        baseline_topic_means[topic] = float(df.loc[tids, "baseline"].mean())

    ncols = 4
    nrows = (len(conditions) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5), sharey=True)
    axes_flat = axes.flatten()

    for idx, cond in enumerate(conditions):
        ax = axes_flat[idx]

        # Per-topic mean utility under this condition
        cond_means = []
        delta_vals = []
        for topic in topics:
            tids = [t for t in task_ids if topic_for_task.get(t) == topic]
            mu = float(df.loc[tids, cond].mean())
            bl = baseline_topic_means[topic]
            cond_means.append(mu)
            delta_vals.append(mu - bl)

        x = np.arange(len(topics))
        # Color: highlight the on-target topic for this condition
        cond_stem = cond.replace("_pos_persona", "").replace("_neg_persona", "")
        cond_stem = cond_stem.replace("compete_", "").replace("_topicpos", "").replace("_shellpos", "")
        colors = []
        for topic in topics:
            if topic in cond_stem:
                colors.append("#228833")
            else:
                colors.append("#BBBBBB")

        ax.bar(x, cond_means, color=colors, edgecolor="none", width=0.7)

        # Annotate each bar with Δu
        for i, (val, dv) in enumerate(zip(cond_means, delta_vals)):
            sign = "+" if dv >= 0 else ""
            ax.text(i, val + 0.3, f"{sign}{dv:.1f}", ha="center", va="bottom",
                    fontsize=6, color="#333333")

        # Baseline reference line per topic
        for i, topic in enumerate(topics):
            ax.plot([i - 0.35, i + 0.35], [baseline_topic_means[topic]] * 2,
                    color="black", linewidth=0.8, linestyle="--")

        short_labels = [t.replace("_", "\n") for t in topics]
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=6)
        cond_label = cond.replace("_persona", "").replace("compete_", "")
        ax.set_title(cond_label, fontsize=8)

    # Hide unused subplots
    for idx in range(len(conditions), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared ylabel
    for row_idx in range(nrows):
        axes[row_idx, 0].set_ylabel("Mean utility", fontsize=8)

    fig.suptitle(f"{exp_name}: Mean topic utility per persona (zero-centered, dashed = baseline, numbers = Δu)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = ASSETS / f"plot_022826_topic_utility_per_persona_{exp_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")
    return path


# ── Main ──


def analyze_experiment(exp_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  {exp_name}")
    print(f"{'='*60}")

    df, _ = load_experiment_utilities(exp_name)
    deltas = compute_deltas(df)
    gt_df = assign_ground_truth(deltas, exp_name)
    stats = compute_summary_stats(deltas, gt_df)

    print(f"\n  Summary stats:")
    for _, row in stats.iterrows():
        c = row["condition"]
        print(f"    {c}: on+={row['mean_du_pos']:+.3f} (n={row['n_pos']}), "
              f"on-={row['mean_du_neg']:+.3f} (n={row['n_neg']}), "
              f"|off|={row['mean_abs_du_off']:.3f} (n={row['n_off']})")

    # Plot 1: condition bars
    plot_condition_bars(deltas, gt_df, stats, exp_name)

    # Plot 2: representative task-level bars
    picks = _pick_representative_conditions(stats, exp_name)
    print(f"\n  Representative conditions: {picks}")
    for cond in picks:
        plot_task_bars(deltas, gt_df, cond, exp_name)

    # Plot 3: competing resolution (exp1d only)
    plot_competing_resolution(deltas, exp_name)

    # Plot 4: per-persona topic utility grid
    plot_topic_utilities_per_persona(df, exp_name)

    return {
        "n_tasks": len(deltas),
        "n_conditions": len(deltas.columns),
        "stats": stats.to_dict(orient="records"),
    }


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for exp_name in ["exp1b", "exp1c", "exp1d"]:
        all_results[exp_name] = analyze_experiment(exp_name)

    # Save summary JSON
    out_path = ASSETS.parent / "utility_shift_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
