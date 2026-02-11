"""Final analysis and plots for competing preferences experiment.

Key results:
1. Cross-task probe analysis: does the probe give different scores to same-topic
   and same-shell tasks under competing prompts that mention the same content?
2. Behavioral analysis: do competing conditions produce different choice rates?
3. Summary statistics and plots for the research log.
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = Path("experiments/competing_preferences")
CROSSED_DIR = Path("experiments/crossed_preferences")
ACT_DIR = EXP_DIR / "activations"
PROBE_DIR = Path("results/probes/gemma3_3k_completion_preference/probes")
RESULTS_DIR = EXP_DIR / "results"
ASSETS_DIR = Path("docs/logs/assets/competing_preferences")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [31, 43, 55]
DATE_STR = datetime.now().strftime("%m%d%y")


def load_probe(layer: int) -> tuple[np.ndarray, float]:
    weights = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return weights[:-1], weights[-1]


def score_activations(activations: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return activations @ coef + intercept


def load_activations(filename: str) -> dict[int, np.ndarray]:
    data = np.load(ACT_DIR / filename, allow_pickle=True)
    return {layer: data[f"layer_{layer}"] for layer in LAYERS}


def load_task_metadata() -> list[dict]:
    with open(CROSSED_DIR / "crossed_tasks.json") as f:
        return json.load(f)


def compute_cross_task_data():
    """Compute full cross-task probe analysis for all pairs."""
    tasks = load_task_metadata()
    task_ids = [t["task_id"] for t in tasks]
    task_idx = {tid: i for i, tid in enumerate(task_ids)}

    coef, intercept = load_probe(31)
    baseline_acts = load_activations("baseline.npz")
    baseline_scores = score_activations(baseline_acts[31], coef, intercept)

    with open(EXP_DIR / "competing_prompts.json") as f:
        prompts = json.load(f)["prompts"]

    pairs: dict[str, dict] = {}
    for p in prompts:
        key = f"{p['target_topic']}_{p['category_shell']}"
        pairs.setdefault(key, {})
        pairs[key][p["favored_dim"]] = p

    results = []
    for pair_key, pair in sorted(pairs.items()):
        topic_prompt = pair["topic"]
        shell_prompt = pair["shell"]
        target_topic = topic_prompt["target_topic"]
        target_shell = topic_prompt["category_shell"]

        topic_acts = load_activations(f"{topic_prompt['id']}.npz")
        shell_acts = load_activations(f"{shell_prompt['id']}.npz")

        topic_scores = score_activations(topic_acts[31], coef, intercept)
        shell_scores = score_activations(shell_acts[31], coef, intercept)

        topic_deltas = topic_scores - baseline_scores
        shell_deltas = shell_scores - baseline_scores

        same_topic = [t for t in tasks if t["topic"] == target_topic and t["category_shell"] != target_shell]
        same_shell = [t for t in tasks if t["category_shell"] == target_shell and t["topic"] != target_topic]
        unrelated = [t for t in tasks if t["topic"] != target_topic and t["category_shell"] != target_shell]

        def mean_delta(task_list, deltas):
            return np.mean([float(deltas[task_idx[t["task_id"]]]) for t in task_list])

        topic_pos_same_topic = mean_delta(same_topic, topic_deltas)
        shell_pos_same_topic = mean_delta(same_topic, shell_deltas)
        topic_pos_same_shell = mean_delta(same_shell, topic_deltas)
        shell_pos_same_shell = mean_delta(same_shell, shell_deltas)
        topic_pos_unrelated = mean_delta(unrelated, topic_deltas)
        shell_pos_unrelated = mean_delta(unrelated, shell_deltas)

        results.append({
            "pair": pair_key,
            "topic": target_topic,
            "shell": target_shell,
            "same_topic_diff": topic_pos_same_topic - shell_pos_same_topic,
            "same_shell_diff": topic_pos_same_shell - shell_pos_same_shell,
            "unrelated_diff": topic_pos_unrelated - shell_pos_unrelated,
            "same_topic_topic_pos": topic_pos_same_topic,
            "same_topic_shell_pos": shell_pos_same_topic,
            "same_shell_topic_pos": topic_pos_same_shell,
            "same_shell_shell_pos": shell_pos_same_shell,
        })

    return results


def plot_cross_task_bar(data: list[dict], filename: str):
    """Bar chart showing probe delta difference by task group."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pairs = [d["pair"] for d in data]
    x = np.arange(len(pairs))
    width = 0.25

    topic_diffs = [d["same_topic_diff"] for d in data]
    shell_diffs = [d["same_shell_diff"] for d in data]
    unrelated_diffs = [d["unrelated_diff"] for d in data]

    bars1 = ax.bar(x - width, topic_diffs, width, label="Same-subject tasks", color="#4CAF50", alpha=0.8, edgecolor="k", linewidth=0.5)
    bars2 = ax.bar(x, shell_diffs, width, label="Same-task-type tasks", color="#2196F3", alpha=0.8, edgecolor="k", linewidth=0.5)
    bars3 = ax.bar(x + width, unrelated_diffs, width, label="Unrelated tasks", color="#9E9E9E", alpha=0.6, edgecolor="k", linewidth=0.5)

    ax.set_xticks(x)
    labels = [d["pair"].replace("_", "\n", 1) for d in data]
    ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_ylabel("Probe Delta Diff (\"love subject\" − \"love task type\")", fontsize=11)
    ax.set_title("Probe Tracks Evaluation, Not Content\n(competing prompts mention same words, differ only in evaluation)", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=10)

    # Significance markers
    topic_mean = np.mean(topic_diffs)
    shell_mean = np.mean(shell_diffs)
    ax.axhline(topic_mean, color="#4CAF50", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(shell_mean, color="#2196F3", linestyle="--", alpha=0.4, linewidth=1)

    plt.tight_layout()
    path = ASSETS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_probe_heatmap(data: list[dict], filename: str):
    """Heatmap showing same-topic and same-shell probe deltas side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pairs = [d["pair"] for d in data]
    n = len(pairs)

    # Same-topic: topic_pos (left bar) vs shell_pos (right bar)
    ax = axes[0]
    topic_pos_vals = [d["same_topic_topic_pos"] for d in data]
    shell_pos_vals = [d["same_topic_shell_pos"] for d in data]

    x = np.arange(n)
    width = 0.35
    ax.barh(x - width/2, topic_pos_vals, width, label="Under 'love subject'", color="#4CAF50", alpha=0.8)
    ax.barh(x + width/2, shell_pos_vals, width, label="Under 'hate subject'", color="#FF5722", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels([p.replace("_", "\n", 1) for p in pairs], fontsize=8)
    ax.set_xlabel("Mean Probe Delta (L31)")
    ax.set_title("Same-Subject Tasks", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=9, loc="lower right")
    ax.invert_yaxis()

    # Same-shell
    ax = axes[1]
    topic_pos_vals = [d["same_shell_topic_pos"] for d in data]
    shell_pos_vals = [d["same_shell_shell_pos"] for d in data]

    ax.barh(x - width/2, topic_pos_vals, width, label="Under 'hate task type'", color="#FF5722", alpha=0.8)
    ax.barh(x + width/2, shell_pos_vals, width, label="Under 'love task type'", color="#4CAF50", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels([p.replace("_", "\n", 1) for p in pairs], fontsize=8)
    ax.set_xlabel("Mean Probe Delta (L31)")
    ax.set_title("Same-Task-Type Tasks", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=9, loc="lower right")
    ax.invert_yaxis()

    plt.suptitle("Probe Responds to Evaluation Direction, Not Content Mentions", fontsize=13, y=1.02)
    plt.tight_layout()
    path = ASSETS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_summary_scatter(data: list[dict], filename: str):
    """Scatter: same-topic diff vs same-shell diff for each pair."""
    fig, ax = plt.subplots(figsize=(7, 6))

    topic_diffs = [d["same_topic_diff"] for d in data]
    shell_diffs = [d["same_shell_diff"] for d in data]

    ax.scatter(topic_diffs, shell_diffs, s=80, alpha=0.7, edgecolors="k", linewidth=0.5, zorder=5)

    for d, td, sd in zip(data, topic_diffs, shell_diffs):
        label = d["topic"][:8]
        ax.annotate(label, (td, sd), fontsize=8, alpha=0.7, xytext=(5, 5), textcoords="offset points")

    # Quadrant lines and shading
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    # Ideal quadrant: top-right → topic diff > 0 AND shell diff < 0
    ax.fill_between([0, 600], 0, -600, alpha=0.05, color="green")
    ax.text(300, -300, "Predicted\nquadrant", ha="center", va="center", fontsize=10, alpha=0.3, color="green")

    ax.set_xlabel("Same-Subject Diff (should be > 0)", fontsize=11)
    ax.set_ylabel("Same-Task-Type Diff (should be < 0)", fontsize=11)
    ax.set_title("Probe Dissociates Subject vs Task Type Evaluation\n(each point = one competing pair)", fontsize=12)

    plt.tight_layout()
    path = ASSETS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    data = compute_cross_task_data()

    # Print summary
    print("=" * 80)
    print("COMPETING PREFERENCES: FINAL ANALYSIS")
    print("=" * 80)

    topic_diffs = [d["same_topic_diff"] for d in data]
    shell_diffs = [d["same_shell_diff"] for d in data]
    unrelated_diffs = [d["unrelated_diff"] for d in data]

    print(f"\n--- Same-Topic Tasks ---")
    print(f"Prediction: diff > 0 (probe favors topic tasks under 'love topic')")
    topic_correct = sum(1 for d in topic_diffs if d > 0)
    t_topic, p_topic = stats.ttest_1samp(topic_diffs, 0)
    print(f"Correct: {topic_correct}/{len(topic_diffs)}")
    print(f"Mean diff: {np.mean(topic_diffs):+.1f} (std={np.std(topic_diffs):.1f})")
    print(f"t={t_topic:.2f}, p={p_topic:.2e}")

    print(f"\n--- Same-Shell Tasks ---")
    print(f"Prediction: diff < 0 (probe disfavors shell tasks under 'hate shell')")
    shell_correct = sum(1 for d in shell_diffs if d < 0)
    t_shell, p_shell = stats.ttest_1samp(shell_diffs, 0)
    print(f"Correct: {shell_correct}/{len(shell_diffs)}")
    print(f"Mean diff: {np.mean(shell_diffs):+.1f} (std={np.std(shell_diffs):.1f})")
    print(f"t={t_shell:.2f}, p={p_shell:.2e}")

    print(f"\n--- Unrelated Tasks (control) ---")
    t_unrel, p_unrel = stats.ttest_1samp(unrelated_diffs, 0)
    print(f"Mean diff: {np.mean(unrelated_diffs):+.1f} (std={np.std(unrelated_diffs):.1f})")
    print(f"t={t_unrel:.2f}, p={p_unrel:.2e}")

    # Effect size comparison
    print(f"\n--- Effect Size Comparison ---")
    print(f"Same-topic mean |diff|: {np.mean(np.abs(topic_diffs)):.1f}")
    print(f"Same-shell mean |diff|: {np.mean(np.abs(shell_diffs)):.1f}")
    print(f"Unrelated mean |diff|: {np.mean(np.abs(unrelated_diffs)):.1f}")
    print(f"Ratio topic/unrelated: {np.mean(np.abs(topic_diffs))/np.mean(np.abs(unrelated_diffs)):.1f}x")
    print(f"Ratio shell/unrelated: {np.mean(np.abs(shell_diffs))/np.mean(np.abs(unrelated_diffs)):.1f}x")

    # Generate plots
    print(f"\n--- Generating Plots ---")
    plot_cross_task_bar(data, f"plot_{DATE_STR}_cross_task_bar.png")
    plot_probe_heatmap(data, f"plot_{DATE_STR}_probe_heatmap.png")
    plot_summary_scatter(data, f"plot_{DATE_STR}_summary_scatter.png")

    # Save data
    with open(RESULTS_DIR / "final_cross_task.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved final data to {RESULTS_DIR / 'final_cross_task.json'}")

    # Load behavioral if available
    beh_path = RESULTS_DIR / "behavioral_competing.json"
    if beh_path.exists():
        with open(beh_path) as f:
            behavioral = json.load(f)
        print(f"\n--- Behavioral Results ({len(behavioral)} conditions) ---")
        pair_ids = sorted(set(b["pair_id"] for b in behavioral))
        for pid in pair_ids:
            topic_b = [b for b in behavioral if b["pair_id"] == pid and b["favored_dim"] == "topic"]
            shell_b = [b for b in behavioral if b["pair_id"] == pid and b["favored_dim"] == "shell"]
            if topic_b and shell_b:
                beh_diff = topic_b[0]["delta"] - shell_b[0]["delta"]
                print(f"  {pid:<25}: beh_diff={beh_diff:+.3f}")


if __name__ == "__main__":
    main()
