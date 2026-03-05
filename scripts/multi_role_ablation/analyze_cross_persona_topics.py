"""Analyze cross-persona probe generalization broken down by topic.

Key question: when a noprompt-trained probe evaluates villain activations,
does it predict villain utilities accurately per-topic? In particular:
- Harmful tasks: does the probe predict ~0 (villain's value) or negative (assistant's value)?
- Which topics drive the cross-persona R², and which break?

Generates scatter plots colored by topic and per-topic metrics.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.probes.core.activations import load_probe_data
from src.probes.core.linear_probe import get_default_alphas
from src.probes.data_loading import load_thurstonian_scores

PERSONAS = ["noprompt", "villain", "aesthete", "midwest"]
LAYER = 31

ACTIVATION_PATHS = {
    "noprompt": Path("activations/gemma_3_27b/activations_prompt_last.npz"),
    "villain": Path("activations/gemma_3_27b_villain/activations_prompt_last.npz"),
    "midwest": Path("activations/gemma_3_27b_midwest/activations_prompt_last.npz"),
    "aesthete": Path("activations/gemma_3_27b_aesthete/activations_prompt_last.npz"),
}

SYS_HASHES = {
    "noprompt": "",
    "villain": "syse8f24ac6",
    "aesthete": "sys021d8ca1",
    "midwest": "sys5d504504",
}

SPLIT_TASK_ID_FILES = {
    "a": Path("configs/measurement/active_learning/mra_exp2_split_a_1000_task_ids.txt"),
    "b": Path("configs/measurement/active_learning/mra_exp2_split_b_500_task_ids.txt"),
    "c": Path("configs/measurement/active_learning/mra_exp2_split_c_1000_task_ids.txt"),
}

TOPICS_PATH = Path("data/topics/topics.json")
ASSETS_DIR = Path("experiments/probe_generalization/multi_role_ablation/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = get_default_alphas(10)

# Topic display config
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

DATE_PREFIX = "030226"


def load_split_task_ids(split: str) -> set[str]:
    with open(SPLIT_TASK_ID_FILES[split]) as f:
        return {line.strip() for line in f if line.strip()}


def get_run_dir(persona: str, split: str) -> Path:
    n = {"a": 1000, "b": 500, "c": 1000}[split]
    sys = SYS_HASHES[persona]
    prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"
    suffix = f"mra_exp2_split_{split}_{n}_task_ids"
    dirname = f"{prefix}_{sys}_{suffix}" if sys else f"{prefix}_{suffix}"
    return Path("results/experiments/mra_exp2/pre_task_active_learning") / dirname


def load_persona_split_data(persona: str, split: str, layer: int):
    run_dir = get_run_dir(persona, split)
    scores = load_thurstonian_scores(run_dir)
    task_ids = sorted(load_split_task_ids(split) & set(scores.keys()))
    X, y, matched_ids = load_probe_data(
        ACTIVATION_PATHS[persona], scores, task_ids, layer
    )
    return X, y, matched_ids


def load_persona_train_data(persona: str, layer: int):
    X_a, y_a, ids_a = load_persona_split_data(persona, "a", layer)
    X_c, y_c, ids_c = load_persona_split_data(persona, "c", layer)
    return (
        np.concatenate([X_a, X_c]),
        np.concatenate([y_a, y_c]),
        list(ids_a) + list(ids_c),
    )


def train_probe(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    best_alpha = None
    best_r2 = -np.inf
    for alpha in ALPHAS:
        probe = Ridge(alpha=alpha)
        probe.fit(X_train_s, y_train)
        y_pred = probe.predict(X_val_s)
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    probe = Ridge(alpha=best_alpha)
    probe.fit(X_train_s, y_train)
    return probe, scaler, best_alpha


def load_topics() -> dict[str, str]:
    with open(TOPICS_PATH) as f:
        topics_raw = json.load(f)
    topic_map = {}
    for tid, models in topics_raw.items():
        for model_name, cats in models.items():
            topic_map[tid] = cats["primary"]
            break
    return topic_map


def main():
    topic_map = load_topics()
    rng = np.random.RandomState(42)

    # Load split_b data for both personas
    X_np_b, y_np_b, ids_np_b = load_persona_split_data("noprompt", "b", LAYER)
    X_v_b, y_v_b, ids_v_b = load_persona_split_data("villain", "b", LAYER)

    # Split_b into sweep/eval halves (same RNG as run_mra_probes_v2)
    n = len(y_np_b)
    idx = rng.permutation(n)
    half = n // 2
    sweep_idx, eval_idx = idx[:half], idx[half:]

    # Train noprompt probe on split_a + split_c
    X_train, y_train, train_ids = load_persona_train_data("noprompt", LAYER)
    probe, scaler, best_alpha = train_probe(
        X_train, y_train,
        X_np_b[sweep_idx], y_np_b[sweep_idx],
    )
    print(f"Noprompt probe: alpha={best_alpha:.1f}, n_train={len(train_ids)}")

    # Also train villain probe (for comparison)
    X_v_train, y_v_train, v_train_ids = load_persona_train_data("villain", LAYER)
    probe_v, scaler_v, best_alpha_v = train_probe(
        X_v_train, y_v_train,
        X_v_b[sweep_idx], y_v_b[sweep_idx],
    )
    print(f"Villain probe: alpha={best_alpha_v:.1f}, n_train={len(v_train_ids)}")

    # Get predictions on villain eval data
    eval_ids_v = [ids_v_b[i] for i in eval_idx]
    X_v_eval = X_v_b[eval_idx]
    y_v_eval = y_v_b[eval_idx]

    # Noprompt probe -> villain activations
    pred_np_on_v = probe.predict(scaler.transform(X_v_eval))
    # Villain probe -> villain activations (within-persona baseline)
    pred_v_on_v = probe_v.predict(scaler_v.transform(X_v_eval))

    # Also get noprompt utilities for same tasks
    noprompt_scores = load_thurstonian_scores(get_run_dir("noprompt", "b"))
    y_np_for_v_tasks = np.array([noprompt_scores[tid] for tid in eval_ids_v])

    # Topics for eval tasks
    eval_topics = [topic_map.get(tid, "unknown") for tid in eval_ids_v]

    # --- Overall metrics ---
    r_cross, _ = pearsonr(y_v_eval, pred_np_on_v)
    pred_adj = pred_np_on_v - np.mean(pred_np_on_v) + np.mean(y_v_eval)
    ss_res = np.sum((y_v_eval - pred_adj) ** 2)
    ss_tot = np.sum((y_v_eval - np.mean(y_v_eval)) ** 2)
    r2_adj = 1 - ss_res / ss_tot

    r_within, _ = pearsonr(y_v_eval, pred_v_on_v)

    print(f"\nOverall (n={len(y_v_eval)}):")
    print(f"  Noprompt->villain: r={r_cross:.4f}, R²_adj={r2_adj:.4f}")
    print(f"  Villain->villain:  r={r_within:.4f}")

    # --- Per-topic breakdown ---
    topics_arr = np.array(eval_topics)
    unique_topics = sorted(set(eval_topics))

    print(f"\n{'Topic':<22} {'n':>4} {'villain_mu':>10} {'noprompt_mu':>11} "
          f"{'probe_pred':>10} {'pred_adj':>10} {'r':>6}")
    print("-" * 80)

    topic_stats = {}
    for topic in unique_topics:
        mask = topics_arr == topic
        n_t = mask.sum()
        if n_t < 5:
            continue
        y_v_t = y_v_eval[mask]
        pred_t = pred_np_on_v[mask]
        pred_adj_t = pred_adj[mask]
        y_np_t = y_np_for_v_tasks[mask]

        r_t = pearsonr(y_v_t, pred_t)[0] if n_t > 2 else float("nan")

        topic_stats[topic] = {
            "n": int(n_t),
            "villain_mean": float(np.mean(y_v_t)),
            "noprompt_mean": float(np.mean(y_np_t)),
            "probe_pred_mean": float(np.mean(pred_t)),
            "probe_pred_adj_mean": float(np.mean(pred_adj_t)),
            "within_r": float(r_t),
        }

        print(f"{topic:<22} {n_t:>4} {np.mean(y_v_t):>10.2f} {np.mean(y_np_t):>11.2f} "
              f"{np.mean(pred_t):>10.2f} {np.mean(pred_adj_t):>10.2f} {r_t:>6.3f}")

    # ===== PLOTS =====

    # --- Plot 1: Main scatter — probe predictions vs villain utilities, colored by topic ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, (pred, pred_label, title_suffix) in enumerate([
        (pred_np_on_v, "Noprompt probe prediction", "noprompt probe"),
        (pred_v_on_v, "Villain probe prediction", "villain probe (within-persona)"),
    ]):
        ax = axes[ax_idx]
        for topic in unique_topics:
            mask = topics_arr == topic
            if mask.sum() < 3:
                continue
            color = TOPIC_COLORS.get(topic, "#95a5a6")
            ax.scatter(
                y_v_eval[mask], pred[mask],
                alpha=0.6, s=20, color=color, label=topic, edgecolors="white", linewidths=0.3,
            )

        lims = [min(y_v_eval.min(), pred.min()) - 2, max(y_v_eval.max(), pred.max()) + 2]
        ax.plot(lims, lims, "--", color="gray", linewidth=0.8)
        r_val = pearsonr(y_v_eval, pred)[0]
        ax.set_xlabel("Villain utility (behavioral)")
        ax.set_ylabel(pred_label)
        ax.set_title(f"Villain eval, {title_suffix} (r={r_val:.3f})")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")

    # Legend on right plot
    axes[1].legend(fontsize=7, loc="upper left", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / f"plot_{DATE_PREFIX}_cross_persona_scatter_by_topic.png", dpi=150)
    plt.close()
    print(f"\nSaved scatter plot")

    # --- Plot 2: Per-topic mean comparison — villain utility vs probe prediction vs noprompt utility ---
    topics_with_stats = [t for t in unique_topics if t in topic_stats and topic_stats[t]["n"] >= 5]
    # Sort by villain mean
    topics_with_stats.sort(key=lambda t: topic_stats[t]["villain_mean"])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(topics_with_stats))
    width = 0.25

    villain_means = [topic_stats[t]["villain_mean"] for t in topics_with_stats]
    noprompt_means = [topic_stats[t]["noprompt_mean"] for t in topics_with_stats]
    probe_means = [topic_stats[t]["probe_pred_adj_mean"] for t in topics_with_stats]

    ax.barh(x - width, villain_means, width, label="Villain utility", color="#e74c3c", alpha=0.8)
    ax.barh(x, probe_means, width, label="Noprompt probe prediction (mean-adj)", color="#3498db", alpha=0.8)
    ax.barh(x + width, noprompt_means, width, label="Noprompt utility", color="#2ecc71", alpha=0.8)

    labels = [f"{t}  (n={topic_stats[t]['n']})" for t in topics_with_stats]
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean utility / prediction")
    ax.set_title("Per-topic: villain utility vs noprompt probe prediction vs noprompt utility")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / f"plot_{DATE_PREFIX}_cross_persona_topic_means.png", dpi=150)
    plt.close()
    print("Saved topic means plot")

    # --- Plot 3: Per-topic within-r for cross-persona probe ---
    fig, ax = plt.subplots(figsize=(9, 5))
    r_values = [topic_stats[t]["within_r"] for t in topics_with_stats]
    colors_r = [TOPIC_COLORS.get(t, "#95a5a6") for t in topics_with_stats]
    ax.barh(range(len(topics_with_stats)), r_values, color=colors_r, alpha=0.8)
    ax.set_yticks(range(len(topics_with_stats)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r (noprompt probe prediction vs villain utility)")
    ax.set_title("Within-topic correlation: noprompt probe on villain eval")
    ax.set_xlim(-0.6, 0.6)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / f"plot_{DATE_PREFIX}_cross_persona_topic_within_r.png", dpi=150)
    plt.close()
    print("Saved within-topic r plot")

    # --- Plot 4: Probe prediction vs noprompt utility (how much does the probe "update" from assistant?) ---
    fig, ax = plt.subplots(figsize=(8, 8))
    for topic in unique_topics:
        mask = topics_arr == topic
        if mask.sum() < 3:
            continue
        color = TOPIC_COLORS.get(topic, "#95a5a6")
        ax.scatter(
            y_np_for_v_tasks[mask], pred_np_on_v[mask],
            alpha=0.6, s=20, color=color, label=topic, edgecolors="white", linewidths=0.3,
        )
    lims = [min(y_np_for_v_tasks.min(), pred_np_on_v.min()) - 2,
            max(y_np_for_v_tasks.max(), pred_np_on_v.max()) + 2]
    ax.plot(lims, lims, "--", color="gray", linewidth=0.8)
    r_update = pearsonr(y_np_for_v_tasks, pred_np_on_v)[0]
    ax.set_xlabel("Noprompt utility (behavioral, same tasks)")
    ax.set_ylabel("Noprompt probe on villain activations")
    ax.set_title(f"How much does the probe 'update' under villain? (r={r_update:.3f})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / f"plot_{DATE_PREFIX}_probe_vs_noprompt_utility.png", dpi=150)
    plt.close()
    print("Saved probe-vs-noprompt scatter")

    # Save stats for the report
    output = {
        "overall": {
            "n_eval": len(y_v_eval),
            "cross_persona_r": float(r_cross),
            "cross_persona_r2_adj": float(r2_adj),
            "within_persona_r": float(r_within),
            "probe_vs_noprompt_utility_r": float(r_update),
        },
        "per_topic": topic_stats,
    }
    output_path = ASSETS_DIR / f"cross_persona_topic_stats_{DATE_PREFIX}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nStats saved to {output_path}")


if __name__ == "__main__":
    main()
