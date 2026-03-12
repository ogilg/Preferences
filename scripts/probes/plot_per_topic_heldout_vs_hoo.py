"""Per-topic bar chart: heldout test set r vs HOO r for IT and PT task_mean probes.

For each model, trains a probe on all 10k data, evaluates on the 4k heldout
eval set broken down by topic, and compares against per-topic HOO r.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.probes.core.activations import load_activations
from src.probes.core.evaluate import score_with_probe
from src.probes.data_loading import load_thurstonian_scores, load_eval_data
from src.probes.experiments.hoo_ridge import build_ridge_xy


def train_probe_at_alpha(activations, task_ids, scores, alpha):
    """Train Ridge at fixed alpha, return weights in raw space."""
    indices, y = build_ridge_xy(task_ids, scores)
    X = activations[indices]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_scaled, y)
    coef_raw = ridge.coef_ / scaler.scale_
    intercept_raw = ridge.intercept_ - coef_raw @ scaler.mean_
    return np.append(coef_raw, intercept_raw)


def per_topic_heldout_r(
    probe_weights: np.ndarray,
    activations: np.ndarray,
    task_ids: np.ndarray,
    eval_scores: dict[str, float],
    topics: dict[str, dict],
) -> dict[str, dict]:
    """Pearson r on heldout eval set, broken down by topic."""
    topic_tasks: dict[str, list[str]] = {}
    for tid in eval_scores:
        if tid not in topics:
            continue
        topic = list(topics[tid].values())[0]["primary"]
        if topic not in topic_tasks:
            topic_tasks[topic] = []
        topic_tasks[topic].append(tid)

    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    all_predictions = score_with_probe(probe_weights, activations)

    results = {}
    for topic, tids in topic_tasks.items():
        valid_tids = [t for t in tids if t in id_to_idx]
        if len(valid_tids) < 10:
            continue
        indices = [id_to_idx[t] for t in valid_tids]
        y_true = np.array([eval_scores[t] for t in valid_tids])
        y_pred = all_predictions[indices]
        r, _ = pearsonr(y_true, y_pred)
        results[topic] = {"r": float(r), "n": len(valid_tids)}

    return results


def load_hoo_per_topic(hoo_path: str, layer_key: str) -> dict[str, dict]:
    with open(hoo_path) as f:
        hoo = json.load(f)
    results = {}
    for fold in hoo["folds"]:
        if layer_key not in fold.get("layers", {}):
            continue
        topic = fold["held_out_groups"][0]
        entry = fold["layers"][layer_key]
        results[topic] = {"hoo_r": entry["hoo_r"], "n": entry["hoo_n_samples"]}
    return results


def plot_model(ax, heldout_by_topic, hoo_by_topic, title, color_heldout, color_hoo, pretty):
    common = sorted(set(heldout_by_topic) & set(hoo_by_topic))
    common.sort(key=lambda t: heldout_by_topic[t]["r"] - hoo_by_topic[t]["hoo_r"], reverse=True)

    h_vals = [heldout_by_topic[t]["r"] for t in common]
    hoo_vals = [hoo_by_topic[t]["hoo_r"] for t in common]
    ns = [hoo_by_topic[t]["n"] for t in common]
    labels = [f"{pretty.get(t, t)}\n(n={ns[i]})" for i, t in enumerate(common)]
    deltas = [h_vals[i] - hoo_vals[i] for i in range(len(common))]

    x = np.arange(len(common))
    width = 0.38

    ax.bar(x - width / 2, h_vals, width, label="Heldout test set (per topic)", color=color_heldout)
    ax.bar(x + width / 2, hoo_vals, width, label="Leave-one-topic-out", color=color_hoo)

    for i in range(len(common)):
        top = max(h_vals[i], hoo_vals[i])
        ax.annotate(
            f"\u0394={deltas[i]:.2f}", xy=(x[i], top + 0.03),
            ha="center", va="bottom", fontsize=7,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8),
        )

    mean_h = np.mean(h_vals)
    mean_hoo = np.mean(hoo_vals)
    ax.axhline(mean_h, color=color_heldout, linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(mean_hoo, color=color_hoo, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(common) - 0.5, mean_h + 0.01, f"mean={mean_h:.2f}",
            fontsize=8, color=color_heldout, ha="right")
    ax.text(len(common) - 0.5, mean_hoo - 0.04, f"mean={mean_hoo:.2f}",
            fontsize=8, color=color_hoo, ha="right")

    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def main():
    pretty = {
        "coding": "Coding", "content_generation": "Content Gen.",
        "fiction": "Fiction", "harmful_request": "Harmful Request",
        "knowledge_qa": "Knowledge QA", "math": "Math",
        "model_manipulation": "Model Manip.", "other": "Other",
        "persuasive_writing": "Persuasive Writing",
        "security_legal": "Security & Legal",
        "sensitive_creative": "Sensitive Creative",
        "summarization": "Summarization", "value_conflict": "Value Conflict",
    }

    with open("data/topics/topics.json") as f:
        topics = json.load(f)

    train_run = Path("results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
    eval_run = Path("results/experiments/main_probes/gemma3_4k_pre_task/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")

    train_scores = load_thurstonian_scores(train_run)
    eval_scores, _ = load_eval_data(eval_run, set(train_scores.keys()))
    task_id_filter = set(train_scores.keys()) | set(eval_scores.keys())

    # Read alpha from HOO summaries (already selected on heldout)
    with open("results/probes/gemma3_10k_hoo_topic_task_mean/hoo_summary.json") as f:
        it_alpha = json.load(f)["folds"][0]["layers"]["ridge_L32"]["best_alpha"]
    with open("results/probes/gemma3_pt_10k_hoo_topic_task_mean/hoo_summary.json") as f:
        pt_alpha = json.load(f)["folds"][0]["layers"]["ridge_L31"]["best_alpha"]

    # --- IT task_mean (L32) ---
    print("Loading IT activations...")
    it_task_ids, it_acts = load_activations(
        Path("activations/gemma_3_27b_turn_boundary_sweep/activations_task_mean.npz"),
        task_id_filter=task_id_filter, layers=[32])
    it_weights = train_probe_at_alpha(it_acts[32], it_task_ids, train_scores, it_alpha)
    it_heldout = per_topic_heldout_r(it_weights, it_acts[32], it_task_ids, eval_scores, topics)
    it_hoo = load_hoo_per_topic("results/probes/gemma3_10k_hoo_topic_task_mean/hoo_summary.json", "ridge_L32")

    # --- PT task_mean (L31) ---
    print("Loading PT activations...")
    pt_task_ids, pt_acts = load_activations(
        Path("activations/gemma_3_27b_pt_task_mean/activations_task_mean.npz"),
        task_id_filter=task_id_filter, layers=[31])
    pt_weights = train_probe_at_alpha(pt_acts[31], pt_task_ids, train_scores, pt_alpha)
    pt_heldout = per_topic_heldout_r(pt_weights, pt_acts[31], pt_task_ids, eval_scores, topics)
    pt_hoo = load_hoo_per_topic("results/probes/gemma3_pt_10k_hoo_topic_task_mean/hoo_summary.json", "ridge_L31")

    # Print summary
    print(f"\nIT L32: heldout per-topic mean={np.mean([v['r'] for v in it_heldout.values()]):.4f}, "
          f"HOO mean={np.mean([v['hoo_r'] for v in it_hoo.values()]):.4f}")
    print(f"PT L31: heldout per-topic mean={np.mean([v['r'] for v in pt_heldout.values()]):.4f}, "
          f"HOO mean={np.mean([v['hoo_r'] for v in pt_hoo.values()]):.4f}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 6))
    plot_model(ax1, it_heldout, it_hoo, "Gemma-3 IT task_mean (L32)",
               "#90CAF9", "#1565C0", pretty)
    plot_model(ax2, pt_heldout, pt_hoo, "Gemma-3 PT task_mean (L31)",
               "#B0BEC5", "#546E7A", pretty)
    fig.suptitle("Per-topic: Heldout test set r vs Leave-one-topic-out r\nSorted by generalisation gap",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = "experiments/eot_probes/turn_boundary_sweep/assets/plot_031026_per_topic_heldout_vs_hoo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
