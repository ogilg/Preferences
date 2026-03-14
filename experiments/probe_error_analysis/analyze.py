"""Probe error analysis: overshoot/undershoot by task type.

Computes residuals (predicted - actual) across three contexts:
1. Character probes (Llama 3.1 8B, L12) — base probe on 10 fine-tuned personas
2. MRA (Gemma 3 27B, L31, 8 personas) — noprompt probe on system-prompt personas
3. OOD Exp 1 (Gemma 3 27B, L31) — 4 sub-experiments

Usage: python -m experiments.probe_error_analysis.analyze
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.ood.analysis import compute_p_choose_from_pairwise
from src.probes.core.activations import load_activations, load_probe_data
from src.probes.data_loading import load_thurstonian_scores
from src.task_data import load_tasks, OriginDataset

from experiments.probe_error_analysis.plots import (
    plot_scatter_grid,
    plot_residual_violins,
    plot_residual_heatmap,
    plot_waterfall,
    plot_cross_context_comparison,
)

ASSETS = Path("experiments/probe_error_analysis/assets")
TOPICS_PATH = Path("data/topics/topics.json")

# ── Character probes ──

CHARACTER_PERSONAS = [
    "goodness", "humor", "impulsiveness", "loving", "mathematical",
    "nonchalance", "poeticism", "remorse", "sarcasm", "sycophancy",
]
CHARACTER_ACTIVATIONS_DIR = Path("activations/character_probes")
CHARACTER_RESULTS_DIR = Path("results/experiments/character_probes")
CHARACTER_PROBE_PATH = Path(
    "results/probes/character_probes/llama8b_base_task_mean/probes/probe_ridge_L12.npy"
)
CHARACTER_LAYER = 12

# ── MRA ──

MRA_PERSONAS = [
    "noprompt", "villain", "aesthete", "midwest",
    "provocateur", "trickster", "autocrat", "sadist",
]

MRA_ACTIVATION_PATHS = {
    "noprompt": Path("activations/gemma_3_27b_pt/activations_prompt_last.npz"),
    "villain": Path("activations/gemma_3_27b_villain/activations_prompt_last.npz"),
    "midwest": Path("activations/gemma_3_27b_midwest/activations_prompt_last.npz"),
    "aesthete": Path("activations/gemma_3_27b_aesthete/activations_prompt_last.npz"),
    "provocateur": Path("activations/gemma_3_27b_provocateur/activations_prompt_last.npz"),
    "trickster": Path("activations/gemma_3_27b_trickster/activations_prompt_last.npz"),
    "autocrat": Path("activations/gemma_3_27b_autocrat/activations_prompt_last.npz"),
    "sadist": Path("activations/gemma_3_27b_sadist/activations_prompt_last.npz"),
}

MRA_PERSONA_RUNS = {
    "noprompt": (Path("results/experiments/mra_exp2/pre_task_active_learning"), ""),
    "villain": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "syse8f24ac6"),
    "aesthete": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys021d8ca1"),
    "midwest": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys5d504504"),
    "provocateur": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sysf4d93514"),
    "trickster": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys09a42edc"),
    "autocrat": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys1c18219a"),
    "sadist": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys39e01d59"),
}

MRA_SPLIT_TASK_ID_FILES = {
    "a": Path("configs/measurement/active_learning/mra_exp2_split_a_1000_task_ids.txt"),
    "b": Path("configs/measurement/active_learning/mra_exp2_split_b_500_task_ids.txt"),
    "c": Path("configs/measurement/active_learning/mra_exp2_split_c_1000_task_ids.txt"),
}

MRA_LAYER = 31
MRA_ALPHAS = np.logspace(0, 5, 10)

# ── OOD ──

OOD_ACTS_DIR = Path("activations/ood")
OOD_RESULTS_DIR = Path("results/ood")
OOD_PROBE_PATH = Path(
    "results/probes/gemma3_10k_heldout_std_raw/probes/probe_ridge_L31.npy"
)
OOD_LAYER = 31


# ── Topic loading ──

def load_topic_map() -> dict[str, str]:
    with open(TOPICS_PATH) as f:
        raw = json.load(f)
    topic_map = {}
    for tid, models in raw.items():
        for model_name, cats in models.items():
            topic_map[tid] = cats["primary"]
            break
    return topic_map


def load_prompt_lookup() -> dict[str, str]:
    all_tasks = load_tasks(
        n=100000,
        origins=[
            OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH,
            OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST,
        ],
    )
    lookup = {t.id: t.prompt for t in all_tasks}
    for tasks_file in [
        Path("configs/ood/tasks/target_tasks.json"),
        Path("configs/ood/tasks/crossed_tasks.json"),
    ]:
        if tasks_file.exists():
            custom = json.load(open(tasks_file))
            for t in custom:
                lookup[t["task_id"]] = t["prompt"]
    return lookup


# ── Character probes data loading ──

def predict_with_probe(weights: np.ndarray, activations: np.ndarray) -> np.ndarray:
    return activations @ weights[:-1] + weights[-1]


def load_character_data(topic_map: dict[str, str]) -> pd.DataFrame:
    print("Loading character probes data...")
    weights = np.load(CHARACTER_PROBE_PATH)
    records = []

    for persona in CHARACTER_PERSONAS:
        # Load scores from split_a
        persona_dir = CHARACTER_RESULTS_DIR / persona / "pre_task_active_learning"
        split_dirs = list(persona_dir.glob("*split_a_*"))
        assert len(split_dirs) == 1, f"Expected 1 split_a dir for {persona}, got {len(split_dirs)}"
        scores = load_thurstonian_scores(split_dirs[0])

        # Load activations
        act_path = CHARACTER_ACTIVATIONS_DIR / f"llama_3_1_8b_{persona}" / "activations_task_mean.npz"
        task_ids, acts = load_activations(act_path, task_id_filter=set(scores), layers=[CHARACTER_LAYER])
        id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
        common = [tid for tid in scores if tid in id_to_idx]

        X = acts[CHARACTER_LAYER][[id_to_idx[tid] for tid in common]]
        y = np.array([scores[tid] for tid in common])
        pred = predict_with_probe(weights, X)

        # Mean-adjust predictions
        pred_adj = pred - np.mean(pred) + np.mean(y)
        residuals = pred_adj - y

        for i, tid in enumerate(common):
            records.append({
                "task_id": tid,
                "actual": float(y[i]),
                "predicted": float(pred_adj[i]),
                "residual": float(residuals[i]),
                "topic": topic_map.get(tid, "unknown"),
                "condition": persona,
                "context": "character",
            })

        r = pearsonr(y, pred)[0]
        print(f"  {persona}: n={len(common)}, r={r:.3f}")

    return pd.DataFrame(records)


# ── MRA data loading ──

def _mra_load_split_task_ids(split: str) -> set[str]:
    with open(MRA_SPLIT_TASK_ID_FILES[split]) as f:
        return {line.strip() for line in f if line.strip()}


def _mra_get_run_dir(persona: str, split: str) -> Path:
    results_dir, sys_hash = MRA_PERSONA_RUNS[persona]
    n = {"a": 1000, "b": 500, "c": 1000}[split]
    prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"
    suffix = f"mra_exp2_split_{split}_{n}_task_ids"
    dirname = f"{prefix}_{sys_hash}_{suffix}" if sys_hash else f"{prefix}_{suffix}"
    return results_dir / dirname


def _mra_load_split_data(persona: str, split: str):
    run_dir = _mra_get_run_dir(persona, split)
    scores = load_thurstonian_scores(run_dir)
    task_ids = sorted(_mra_load_split_task_ids(split) & set(scores))
    return load_probe_data(MRA_ACTIVATION_PATHS[persona], scores, task_ids, MRA_LAYER)


def load_mra_data(topic_map: dict[str, str]) -> pd.DataFrame:
    print("Loading MRA data...")
    rng = np.random.RandomState(42)

    # Train noprompt probe on splits a+c
    X_a, y_a, ids_a = _mra_load_split_data("noprompt", "a")
    X_c, y_c, ids_c = _mra_load_split_data("noprompt", "c")
    X_train = np.concatenate([X_a, X_c])
    y_train = np.concatenate([y_a, y_c])

    # Split noprompt split_b into sweep/eval halves
    X_b, y_b, ids_b = _mra_load_split_data("noprompt", "b")
    n_b = len(y_b)
    idx_b = rng.permutation(n_b)
    half = n_b // 2
    X_sweep, y_sweep = X_b[idx_b[:half]], y_b[idx_b[:half]]

    # Train with alpha selection
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_sweep_s = scaler.transform(X_sweep)

    best_alpha, best_r2 = None, -np.inf
    for alpha in MRA_ALPHAS:
        probe = Ridge(alpha=alpha)
        probe.fit(X_train_s, y_train)
        y_pred = probe.predict(X_sweep_s)
        r2 = 1 - np.sum((y_sweep - y_pred) ** 2) / np.sum((y_sweep - np.mean(y_sweep)) ** 2)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    probe = Ridge(alpha=best_alpha)
    probe.fit(X_train_s, y_train)
    print(f"  Noprompt probe: alpha={best_alpha:.1f}, sweep R²={best_r2:.4f}")

    # Evaluate on all personas' eval half of split_b
    records = []
    for persona in MRA_PERSONAS:
        X_p_b, y_p_b, ids_p_b = _mra_load_split_data(persona, "b")
        X_eval = X_p_b[idx_b[half:]]
        y_eval = y_p_b[idx_b[half:]]
        eval_ids = [ids_p_b[i] for i in idx_b[half:]]

        pred = probe.predict(scaler.transform(X_eval))
        pred_adj = pred - np.mean(pred) + np.mean(y_eval)
        residuals = pred_adj - y_eval

        r = pearsonr(y_eval, pred)[0]
        print(f"  {persona}: n={len(y_eval)}, r={r:.3f}")

        for i, tid in enumerate(eval_ids):
            records.append({
                "task_id": tid,
                "actual": float(y_eval[i]),
                "predicted": float(pred_adj[i]),
                "residual": float(residuals[i]),
                "topic": topic_map.get(tid, "unknown"),
                "condition": persona,
                "context": "mra",
            })

    return pd.DataFrame(records)


# ── OOD data loading ──

def _ood_load_probe():
    probe = np.load(OOD_PROBE_PATH)
    return probe[:-1], float(probe[-1])


def _ood_score_activations(npz_path: Path, weights: np.ndarray, bias: float) -> dict[str, float]:
    data = np.load(npz_path, allow_pickle=True)
    acts = data[f"layer_{OOD_LAYER}"]
    scores = acts @ weights + bias
    task_ids = list(data["task_ids"])
    return dict(zip(task_ids, scores.tolist()))


def _ood_compute_records(
    rates: dict[str, dict[str, float]],
    acts_dir: Path,
    weights: np.ndarray,
    bias: float,
) -> list[dict]:
    baseline_rates = rates["baseline"]
    baseline_npz = acts_dir / "baseline" / "activations_prompt_last.npz"
    baseline_scores = _ood_score_activations(baseline_npz, weights, bias)

    records = []
    for cid in rates:
        if cid == "baseline":
            continue
        cond_npz = acts_dir / cid / "activations_prompt_last.npz"
        if not cond_npz.exists():
            continue
        cond_scores = _ood_score_activations(cond_npz, weights, bias)

        for tid, cond_rate in rates[cid].items():
            if tid not in baseline_rates or tid not in baseline_scores or tid not in cond_scores:
                continue
            beh_delta = cond_rate - baseline_rates[tid]
            probe_delta = cond_scores[tid] - baseline_scores[tid]
            records.append({
                "task_id": tid,
                "actual": beh_delta,
                "predicted": probe_delta,
                "residual": probe_delta - beh_delta,
                "condition": cid,
            })
    return records


def load_ood_data(topic_map: dict[str, str]) -> pd.DataFrame:
    print("Loading OOD Exp 1 data...")
    weights, bias = _ood_load_probe()
    all_records = []

    # Exp 1a: category preference
    print("  Exp 1a: category preference")
    pairwise = json.load(open(OOD_RESULTS_DIR / "category_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    records = _ood_compute_records(rates, OOD_ACTS_DIR / "exp1_category", weights, bias)
    for r in records:
        r["context"] = "ood_1a"
    print(f"    {len(records)} records")
    all_records.extend(records)

    # Exp 1b: hidden preference (targeted only, hidden_ tasks)
    print("  Exp 1b: hidden preference")
    pairwise = json.load(open(OOD_RESULTS_DIR / "hidden_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    targeted_rates = {
        k: {tid: v for tid, v in rd.items() if tid.startswith("hidden_")}
        for k, rd in rates.items()
        if not k.startswith("compete_")
    }
    records = _ood_compute_records(targeted_rates, OOD_ACTS_DIR / "exp1_prompts", weights, bias)
    for r in records:
        r["context"] = "ood_1b"
    print(f"    {len(records)} records")
    all_records.extend(records)

    # Exp 1c: crossed preference (crossed_ tasks, non-compete conditions)
    print("  Exp 1c: crossed preference")
    pairwise = json.load(open(OOD_RESULTS_DIR / "crossed_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    targeted_rates = {
        k: {tid: v for tid, v in rd.items() if tid.startswith("crossed_")}
        for k, rd in rates.items()
        if not k.startswith("compete_")
    }
    records = _ood_compute_records(targeted_rates, OOD_ACTS_DIR / "exp1_prompts", weights, bias)
    for r in records:
        r["context"] = "ood_1c"
    print(f"    {len(records)} records")
    all_records.extend(records)

    # Exp 1d: competing preference (compete_ conditions, crossed_ tasks)
    print("  Exp 1d: competing preference")
    competing_rates = {
        k: {tid: v for tid, v in rd.items() if tid.startswith("crossed_")}
        for k, rd in rates.items()
        if k.startswith("compete_") or k == "baseline"
    }
    records = _ood_compute_records(competing_rates, OOD_ACTS_DIR / "exp1_prompts", weights, bias)
    for r in records:
        r["context"] = "ood_1d"
    print(f"    {len(records)} records")
    all_records.extend(records)

    df = pd.DataFrame(all_records)
    # Add topics
    df["topic"] = df["task_id"].map(topic_map).fillna("unknown")
    return df


# ── Stats computation ──

def compute_stats(df: pd.DataFrame, context: str) -> dict:
    stats = {"context": context, "conditions": {}}
    for cond, sub in df.groupby("condition"):
        r = pearsonr(sub["actual"], sub["predicted"])[0] if len(sub) > 2 else float("nan")
        stats["conditions"][cond] = {
            "pearson_r": float(r),
            "mean_residual": float(sub["residual"].mean()),
            "std_residual": float(sub["residual"].std()),
            "n": len(sub),
        }

    # Per-topic stats
    stats["topics"] = {}
    for topic, sub in df.groupby("topic"):
        stats["topics"][topic] = {
            "mean_residual": float(sub["residual"].mean()),
            "std_residual": float(sub["residual"].std()),
            "n": len(sub),
        }

    return stats


# ── Report generation ──

def generate_report(
    context_dfs: dict[str, pd.DataFrame],
    context_stats: dict[str, dict],
    context_plots: dict[str, dict[str, str]],
    prompt_lookup: dict[str, str],
    cross_context_plot: str,
):
    lines = ["# Probe Error Analysis Report\n"]

    for ctx_name, df in context_dfs.items():
        lines.append(f"\n## {ctx_name}\n")
        stats = context_stats[ctx_name]
        plots = context_plots[ctx_name]

        # Overall stats table
        lines.append("### Per-condition stats\n")
        lines.append("| Condition | Pearson r | Mean residual | Std residual | n |")
        lines.append("|-----------|-----------|---------------|--------------|---|")
        for cond, cs in sorted(stats["conditions"].items()):
            lines.append(
                f"| {cond} | {cs['pearson_r']:.3f} | {cs['mean_residual']:.3f} "
                f"| {cs['std_residual']:.3f} | {cs['n']} |"
            )

        # Embed plots
        for plot_name, plot_path in plots.items():
            rel_path = str(Path(plot_path).relative_to("experiments/probe_error_analysis"))
            lines.append(f"\n### {plot_name}\n")
            lines.append(f"![{plot_name}]({rel_path})\n")

        # Top/bottom 15 tasks table
        lines.append("\n### Top 15 overshoot tasks\n")
        lines.append("| task_id | topic | prompt | actual | predicted | residual |")
        lines.append("|---------|-------|--------|--------|-----------|----------|")
        for _, row in df.nlargest(15, "residual").iterrows():
            prompt = prompt_lookup.get(row["task_id"], "")[:100].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {row['task_id'][:16]} | {row['topic']} | {prompt} "
                f"| {row['actual']:.3f} | {row['predicted']:.3f} | {row['residual']:.3f} |"
            )

        lines.append("\n### Top 15 undershoot tasks\n")
        lines.append("| task_id | topic | prompt | actual | predicted | residual |")
        lines.append("|---------|-------|--------|--------|-----------|----------|")
        for _, row in df.nsmallest(15, "residual").iterrows():
            prompt = prompt_lookup.get(row["task_id"], "")[:100].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {row['task_id'][:16]} | {row['topic']} | {prompt} "
                f"| {row['actual']:.3f} | {row['predicted']:.3f} | {row['residual']:.3f} |"
            )

    # Cross-context section
    lines.append("\n## Cross-Context Comparison\n")
    rel_path = str(Path(cross_context_plot).relative_to("experiments/probe_error_analysis"))
    lines.append(f"![Cross-context comparison]({rel_path})\n")

    # Summary: consistently problematic topics
    lines.append("\n### Consistently problematic topics\n")
    all_df = pd.concat(context_dfs.values())
    topic_summary = all_df.groupby("topic")["residual"].agg(["mean", "std", "count"])
    topic_summary = topic_summary[topic_summary["count"] >= 10].sort_values("mean")

    lines.append("| Topic | Mean residual | Std | n |")
    lines.append("|-------|---------------|-----|---|")
    for topic, row in topic_summary.iterrows():
        lines.append(f"| {topic} | {row['mean']:.3f} | {row['std']:.3f} | {int(row['count'])} |")

    report_path = Path("experiments/probe_error_analysis/probe_error_report.md")
    report_path.write_text("\n".join(lines))
    print(f"\nReport saved to {report_path}")


# ── Main ──

def main():
    ASSETS.mkdir(parents=True, exist_ok=True)

    topic_map = load_topic_map()
    print(f"Loaded {len(topic_map)} topic mappings")

    prompt_lookup = load_prompt_lookup()
    print(f"Loaded {len(prompt_lookup)} task prompts")

    context_dfs: dict[str, pd.DataFrame] = {}
    context_stats: dict[str, dict] = {}
    context_plots: dict[str, dict[str, str]] = {}

    # ── Character probes ──
    char_df = load_character_data(topic_map)
    # Add prompts for waterfall
    char_df["prompt"] = char_df["task_id"].map(prompt_lookup).fillna("")
    context_dfs["Character Probes"] = char_df
    context_stats["Character Probes"] = compute_stats(char_df, "character")

    print("\nGenerating character probe plots...")
    char_plots = {}
    char_plots["Scatter Grid"] = plot_scatter_grid(
        char_df, "Character Probes", CHARACTER_PERSONAS, ncols=2,
    )
    char_plots["Residual Violins"] = plot_residual_violins(char_df, "Character Probes")
    char_plots["Residual Heatmap"] = plot_residual_heatmap(char_df, "Character Probes")
    char_plots["Waterfall"] = plot_waterfall(char_df, "Character Probes")
    context_plots["Character Probes"] = char_plots

    # ── MRA ──
    mra_df = load_mra_data(topic_map)
    mra_df["prompt"] = mra_df["task_id"].map(prompt_lookup).fillna("")
    context_dfs["MRA"] = mra_df
    context_stats["MRA"] = compute_stats(mra_df, "mra")

    print("\nGenerating MRA plots...")
    mra_plots = {}
    mra_plots["Scatter Grid"] = plot_scatter_grid(
        mra_df, "MRA", MRA_PERSONAS, ncols=4,
    )
    mra_plots["Residual Violins"] = plot_residual_violins(mra_df, "MRA")
    mra_plots["Residual Heatmap"] = plot_residual_heatmap(mra_df, "MRA")
    mra_plots["Waterfall"] = plot_waterfall(mra_df, "MRA")
    context_plots["MRA"] = mra_plots

    # ── OOD ──
    ood_df = load_ood_data(topic_map)
    ood_df["prompt"] = ood_df["task_id"].map(prompt_lookup).fillna("")

    # Per sub-experiment plots
    for sub_exp, sub_label in [
        ("ood_1a", "OOD 1a Category"),
        ("ood_1b", "OOD 1b Hidden"),
        ("ood_1c", "OOD 1c Crossed"),
        ("ood_1d", "OOD 1d Competing"),
    ]:
        sub_df = ood_df[ood_df["context"] == sub_exp].copy()
        if len(sub_df) == 0:
            print(f"  Skipping {sub_label} (no data)")
            continue

        context_dfs[sub_label] = sub_df
        context_stats[sub_label] = compute_stats(sub_df, sub_exp)

        print(f"\nGenerating {sub_label} plots...")
        conditions = sorted(sub_df["condition"].unique())
        sub_plots = {}

        if len(conditions) <= 20:
            sub_plots["Scatter Grid"] = plot_scatter_grid(
                sub_df, sub_label, conditions, ncols=5, equal_axes=False,
            )
        sub_plots["Residual Violins"] = plot_residual_violins(sub_df, sub_label)
        if len(conditions) > 1:
            sub_plots["Residual Heatmap"] = plot_residual_heatmap(sub_df, sub_label)
        sub_plots["Waterfall"] = plot_waterfall(sub_df, sub_label)
        context_plots[sub_label] = sub_plots

    # ── Cross-context comparison ──
    print("\nGenerating cross-context comparison...")
    cross_plot = plot_cross_context_comparison(context_dfs)

    # ── Report ──
    generate_report(context_dfs, context_stats, context_plots, prompt_lookup, cross_plot)

    # ── Save stats JSON ──
    stats_path = Path("experiments/probe_error_analysis/residual_stats.json")
    # Convert stats for JSON serialization
    json_stats = {}
    for ctx_name, stats in context_stats.items():
        json_stats[ctx_name] = stats
    with open(stats_path, "w") as f:
        json.dump(json_stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
