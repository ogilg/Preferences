"""Analyze noprompt probe transfer to evil personas with per-topic breakdowns.

For each persona, generates:
1. Scatter: probe predictions vs persona utilities, colored by topic
2. Per-topic table (means, within-topic r)
3. Summary bar chart of overall Pearson r across personas and layers

Usage: python -m scripts.multi_role_ablation.analyze_evil_persona_transfer
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.probes.core.activations import load_probe_data
from src.probes.core.linear_probe import get_default_alphas
from src.probes.data_loading import load_thurstonian_scores

DATE_PREFIX = "030326"
ASSETS = Path("experiments/probe_generalization/multi_role_ablation/assets")
TOPICS_PATH = Path("data/topics/topics.json")

TOPIC_COLORS = {
    "harmful_request": "#e74c3c",
    "security_legal": "#e67e22",
    "math": "#3498db",
    "knowledge_qa": "#2ecc71",
    "content_generation": "#9b59b6",
    "fiction": "#1abc9c",
    "coding": "#34495e",
    "model_manipulation": "#f39c12",
    "persuasive_writing": "#e84393",
    "sensitive_creative": "#fd79a8",
    "summarization": "#636e72",
    "other": "#95a5a6",
}

ALL_PERSONAS = [
    "noprompt", "villain", "aesthete", "midwest",
    "provocateur", "trickster", "autocrat", "sadist",
]
EVIL_PERSONAS = ["provocateur", "trickster", "autocrat", "sadist"]
ORIGINAL_PERSONAS = ["villain", "aesthete", "midwest"]

PERSONA_LABELS = {
    "noprompt": "No prompt",
    "villain": "Villain (Mortivex)",
    "aesthete": "Aesthete (Celestine)",
    "midwest": "Midwest (Glenn)",
    "provocateur": "Provocateur (Saul Vickers)",
    "trickster": "Trickster (Wraith)",
    "autocrat": "Autocrat (Gen. Volkov)",
    "sadist": "Sadist (Damien Kross)",
}

ACTIVATION_PATHS = {
    "noprompt": Path("activations/gemma_3_27b/activations_prompt_last.npz"),
    "villain": Path("activations/gemma_3_27b_villain/activations_prompt_last.npz"),
    "midwest": Path("activations/gemma_3_27b_midwest/activations_prompt_last.npz"),
    "aesthete": Path("activations/gemma_3_27b_aesthete/activations_prompt_last.npz"),
    "provocateur": Path("activations/gemma_3_27b_provocateur/activations_prompt_last.npz"),
    "trickster": Path("activations/gemma_3_27b_trickster/activations_prompt_last.npz"),
    "autocrat": Path("activations/gemma_3_27b_autocrat/activations_prompt_last.npz"),
    "sadist": Path("activations/gemma_3_27b_sadist/activations_prompt_last.npz"),
}

PERSONA_RUNS = {
    "noprompt": (Path("results/experiments/mra_exp2/pre_task_active_learning"), ""),
    "villain": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "syse8f24ac6"),
    "aesthete": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys021d8ca1"),
    "midwest": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys5d504504"),
    "provocateur": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sysf4d93514"),
    "trickster": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys09a42edc"),
    "autocrat": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys1c18219a"),
    "sadist": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys39e01d59"),
}

SPLIT_TASK_ID_FILES = {
    "a": Path("configs/measurement/active_learning/mra_exp2_split_a_1000_task_ids.txt"),
    "b": Path("configs/measurement/active_learning/mra_exp2_split_b_500_task_ids.txt"),
    "c": Path("configs/measurement/active_learning/mra_exp2_split_c_1000_task_ids.txt"),
}

LAYERS = [31, 43, 55]
FOCUS_LAYER = 55  # best cross-persona transfer for most evil personas
ALPHAS = get_default_alphas(10)


def load_split_task_ids(split: str) -> set[str]:
    with open(SPLIT_TASK_ID_FILES[split]) as f:
        return {line.strip() for line in f if line.strip()}


def get_run_dir(persona: str, split: str) -> Path:
    results_dir, sys_hash = PERSONA_RUNS[persona]
    n = {"a": 1000, "b": 500, "c": 1000}[split]
    prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"
    suffix = f"mra_exp2_split_{split}_{n}_task_ids"
    dirname = f"{prefix}_{sys_hash}_{suffix}" if sys_hash else f"{prefix}_{suffix}"
    return results_dir / dirname


def load_persona_split_data(persona: str, split: str, layer: int):
    run_dir = get_run_dir(persona, split)
    scores = load_thurstonian_scores(run_dir)
    task_ids = sorted(load_split_task_ids(split) & set(scores.keys()))
    return load_probe_data(ACTIVATION_PATHS[persona], scores, task_ids, layer)


def load_persona_train_data(persona: str, layer: int):
    X_a, y_a, ids_a = load_persona_split_data(persona, "a", layer)
    X_c, y_c, ids_c = load_persona_split_data(persona, "c", layer)
    return np.concatenate([X_a, X_c]), np.concatenate([y_a, y_c]), list(ids_a) + list(ids_c)


def load_topics() -> dict[str, str]:
    with open(TOPICS_PATH) as f:
        raw = json.load(f)
    # topics.json: {task_id: {model: {primary: str, secondary: str}}}
    topics = {}
    for task_id, models in raw.items():
        for model_data in models.values():
            topics[task_id] = model_data["primary"]
            break
    return topics


def train_noprompt_probe(layer: int):
    X_train, y_train, train_ids = load_persona_train_data("noprompt", layer)
    X_b, y_b, ids_b = load_persona_split_data("noprompt", "b", layer)
    rng = np.random.RandomState(42)
    n = len(y_b)
    idx = rng.permutation(n)
    half = n // 2
    X_val, y_val = X_b[idx[:half]], y_b[idx[:half]]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    best_alpha, best_r2 = None, -np.inf
    for alpha in ALPHAS:
        probe = Ridge(alpha=alpha)
        probe.fit(X_train_s, y_train)
        y_pred = probe.predict(X_val_s)
        r2 = 1 - np.sum((y_val - y_pred)**2) / np.sum((y_val - np.mean(y_val))**2)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    probe = Ridge(alpha=best_alpha)
    probe.fit(X_train_s, y_train)
    return probe, scaler, best_alpha


def get_eval_data(persona: str, layer: int):
    """Load eval half of split_b for a persona."""
    X_b, y_b, ids_b = load_persona_split_data(persona, "b", layer)
    rng = np.random.RandomState(42)
    n = len(y_b)
    idx = rng.permutation(n)
    half = n // 2
    eval_idx = idx[half:]
    return X_b[eval_idx], y_b[eval_idx], [ids_b[i] for i in eval_idx]


def get_noprompt_scores_for_ids(task_ids: list[str]) -> dict[str, float]:
    """Load noprompt utilities for specific task IDs (pooling all splits)."""
    all_scores = {}
    for split in ["a", "b", "c"]:
        run_dir = get_run_dir("noprompt", split)
        scores = load_thurstonian_scores(run_dir)
        all_scores.update(scores)
    return {tid: all_scores[tid] for tid in task_ids if tid in all_scores}


def plot_scatter_per_persona(persona: str, y_true, y_pred, task_ids, topics, layer):
    """Scatter: probe predictions vs persona utilities, colored by topic."""
    topic_arr = np.array([topics.get(tid, "other") for tid in task_ids])
    unique_topics = sorted(set(topic_arr))

    fig, ax = plt.subplots(figsize=(8, 7))

    for topic in unique_topics:
        mask = topic_arr == topic
        color = TOPIC_COLORS.get(topic, "#95a5a6")
        n_t = mask.sum()
        ax.scatter(y_true[mask], y_pred[mask], c=color, alpha=0.6, s=20,
                   edgecolors="white", linewidths=0.3, label=f"{topic} (n={n_t})")

    r, _ = pearsonr(y_true, y_pred)
    lims = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, "--", color="gray", linewidth=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"{PERSONA_LABELS[persona]} utility", fontsize=11)
    ax.set_ylabel("Noprompt probe prediction", fontsize=11)
    ax.set_title(f"Noprompt probe → {PERSONA_LABELS[persona]} (L{layer}, r={r:.2f})", fontsize=12)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    ax.set_aspect("equal")
    fig.tight_layout()

    path = ASSETS / f"plot_{DATE_PREFIX}_scatter_{persona}_L{layer}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return path


def compute_utility_correlations() -> dict[str, float]:
    """Pearson r between noprompt and each persona's Thurstonian utilities (pooled splits)."""
    noprompt_scores = {}
    for split in ["a", "b", "c"]:
        run_dir = get_run_dir("noprompt", split)
        noprompt_scores.update(load_thurstonian_scores(run_dir))

    correlations = {}
    for persona in ALL_PERSONAS:
        if persona == "noprompt":
            continue
        persona_scores = {}
        for split in ["a", "b", "c"]:
            run_dir = get_run_dir(persona, split)
            persona_scores.update(load_thurstonian_scores(run_dir))
        common_ids = sorted(set(noprompt_scores) & set(persona_scores))
        np_vals = np.array([noprompt_scores[tid] for tid in common_ids])
        p_vals = np.array([persona_scores[tid] for tid in common_ids])
        correlations[persona] = float(pearsonr(np_vals, p_vals)[0])
    return correlations


def plot_summary_bar(all_results: dict, utility_corrs: dict[str, float]):
    """Bar chart: Pearson r across all personas for each layer, with utility correlation baseline."""
    personas = [p for p in ALL_PERSONAS if p != "noprompt"]
    n_personas = len(personas)
    n_layers = len(LAYERS)
    n_groups = n_layers + 1  # layers + utility correlation

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_personas)
    width = 0.8 / n_groups

    # Utility correlation bars (grey, leftmost)
    util_rs = [utility_corrs[p] for p in personas]
    offset = (0 - n_groups / 2 + 0.5) * width
    bars = ax.bar(x + offset, util_rs, width, label="Utility correlation",
                  color="#bdc3c7", alpha=0.8, edgecolor="#95a5a6", linewidth=0.5)
    for bar, r in zip(bars, util_rs):
        y_pos = max(bar.get_height() + 0.01, 0.03)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{r:.2f}", ha="center", va="bottom", fontsize=6, color="#7f8c8d")

    # Layer bars
    layer_colors = ["#3498db", "#e74c3c", "#2ecc71"]
    for i, layer in enumerate(LAYERS):
        layer_key = f"L{layer}"
        rs = [all_results[layer_key][p]["pearson_r"] for p in personas]
        offset = (i + 1 - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, rs, width, label=f"Layer {layer}",
                      color=layer_colors[i], alpha=0.8)
        for bar, r in zip(bars, rs):
            y_pos = max(bar.get_height() + 0.01, 0.03)
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{r:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([PERSONA_LABELS[p] for p in personas], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_title("Noprompt probe transfer to each persona", fontsize=13)
    ax.set_ylim(-0.1, 1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    path = ASSETS / f"plot_{DATE_PREFIX}_noprompt_transfer_summary.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return path, utility_corrs


def plot_topic_means_per_persona(persona: str, topic_stats: list[dict], layer: int):
    """Grouped bar: persona utility vs noprompt utility vs probe prediction per topic."""
    topic_stats = sorted(topic_stats, key=lambda x: -x["n"])

    topics_list = [s["topic"] for s in topic_stats]
    n_topics = len(topics_list)
    y_pos = np.arange(n_topics)

    fig, ax = plt.subplots(figsize=(10, max(4, n_topics * 0.5 + 1)))
    width = 0.25

    persona_means = [s["persona_mean"] for s in topic_stats]
    noprompt_means = [s["noprompt_mean"] for s in topic_stats]
    probe_means = [s["probe_mean"] for s in topic_stats]

    ax.barh(y_pos - width, persona_means, width, label=f"{PERSONA_LABELS[persona]} utility",
            color="#e74c3c", alpha=0.8)
    ax.barh(y_pos, noprompt_means, width, label="Noprompt utility",
            color="#3498db", alpha=0.8)
    ax.barh(y_pos + width, probe_means, width, label="Probe prediction",
            color="#2ecc71", alpha=0.8)

    labels = [f"{s['topic']} (n={s['n']})" for s in topic_stats]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean utility / prediction", fontsize=10)
    ax.set_title(f"Per-topic means: noprompt probe → {PERSONA_LABELS[persona]} (L{layer})", fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)
    ax.invert_yaxis()
    fig.tight_layout()

    path = ASSETS / f"plot_{DATE_PREFIX}_topic_means_{persona}_L{layer}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return path


def compute_topic_stats(persona, y_true, y_pred, task_ids, topics, noprompt_scores):
    topic_arr = np.array([topics.get(tid, "other") for tid in task_ids])
    unique_topics = sorted(set(topic_arr))

    stats = []
    for topic in unique_topics:
        mask = topic_arr == topic
        n_t = int(mask.sum())
        if n_t < 3:
            continue
        y_t = y_true[mask]
        pred_t = y_pred[mask]
        np_scores = np.array([noprompt_scores.get(task_ids[i], np.nan)
                              for i in range(len(task_ids)) if mask[i]])
        np_scores = np_scores[~np.isnan(np_scores)]

        within_r = float(pearsonr(y_t, pred_t)[0]) if n_t >= 3 else np.nan

        stats.append({
            "topic": topic,
            "n": n_t,
            "persona_mean": float(np.mean(y_t)),
            "noprompt_mean": float(np.mean(np_scores)) if len(np_scores) > 0 else np.nan,
            "probe_mean": float(np.mean(pred_t)),
            "within_r": within_r,
        })

    return stats


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    topics = load_topics()
    all_results = {}  # layer -> persona -> metrics
    all_topic_stats = {}  # persona -> topic stats at focus layer
    scatter_paths = {}  # persona -> path

    for layer in LAYERS:
        print(f"\n=== Layer {layer} ===")
        layer_key = f"L{layer}"
        probe, scaler, best_alpha = train_noprompt_probe(layer)
        print(f"  Noprompt probe: alpha={best_alpha:.1f}")

        all_results[layer_key] = {}

        for persona in ALL_PERSONAS:
            if persona == "noprompt":
                # Eval on noprompt itself
                X_eval, y_eval, eval_ids = get_eval_data("noprompt", layer)
            else:
                X_eval, y_eval, eval_ids = get_eval_data(persona, layer)

            y_pred = probe.predict(scaler.transform(X_eval))
            r, _ = pearsonr(y_eval, y_pred)
            r2 = 1 - np.sum((y_eval - y_pred)**2) / np.sum((y_eval - np.mean(y_eval))**2)
            y_pred_adj = y_pred - np.mean(y_pred) + np.mean(y_eval)
            r2_adj = 1 - np.sum((y_eval - y_pred_adj)**2) / np.sum((y_eval - np.mean(y_eval))**2)

            all_results[layer_key][persona] = {
                "pearson_r": float(r),
                "r2": float(r2),
                "r2_adjusted": float(r2_adj),
                "n_samples": len(y_eval),
            }
            print(f"  {persona}: r={r:.3f}, R²_adj={r2_adj:.3f} (n={len(y_eval)})")

            # Generate scatter + topic stats at focus layer
            if layer == FOCUS_LAYER and persona != "noprompt":
                scatter_paths[persona] = plot_scatter_per_persona(
                    persona, y_eval, y_pred, eval_ids, topics, layer
                )
                noprompt_scores = get_noprompt_scores_for_ids(eval_ids)
                stats = compute_topic_stats(persona, y_eval, y_pred, eval_ids, topics, noprompt_scores)
                all_topic_stats[persona] = stats
                plot_topic_means_per_persona(persona, stats, layer)

    # Summary bar chart
    utility_corrs = compute_utility_correlations()
    print("\nUtility correlations (noprompt vs persona):")
    for p, r in sorted(utility_corrs.items(), key=lambda x: -x[1]):
        print(f"  {p}: r={r:.3f}")
    summary_path, _ = plot_summary_bar(all_results, utility_corrs)

    # Save raw results
    output_path = Path("results/experiments/mra_exp3/probes/evil_persona_transfer.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"results": all_results, "topic_stats": all_topic_stats}, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table for report
    print(f"\n{'='*70}")
    print("REPORT DATA")
    print(f"{'='*70}")

    print(f"\n### Overall transfer (Pearson r)")
    print(f"| Persona | L31 | L43 | L55 |")
    print(f"|---|---|---|---|")
    for persona in ALL_PERSONAS:
        row = f"| {PERSONA_LABELS[persona]} |"
        for layer in LAYERS:
            r = all_results[f"L{layer}"][persona]["pearson_r"]
            row += f" {r:.3f} |"
        print(row)

    for persona in EVIL_PERSONAS + ORIGINAL_PERSONAS:
        if persona in all_topic_stats:
            print(f"\n### {PERSONA_LABELS[persona]} per-topic (L{FOCUS_LAYER})")
            print(f"| Topic | n | Persona utility | Noprompt utility | Probe prediction | Within-topic r |")
            print(f"|---|---|---|---|---|---|")
            for s in sorted(all_topic_stats[persona], key=lambda x: -x["n"]):
                nr = f"**{s['within_r']:.2f}**" if abs(s["within_r"]) > 0.5 else f"{s['within_r']:.2f}"
                print(f"| {s['topic']} | {s['n']} | {s['persona_mean']:.2f} | "
                      f"{s['noprompt_mean']:.2f} | {s['probe_mean']:.2f} | {nr} |")


if __name__ == "__main__":
    main()
