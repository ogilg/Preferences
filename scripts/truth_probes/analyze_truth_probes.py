"""Analyze whether preference probe directions separate true from false CREAK claims."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

from src.probes.core.activations import load_activations
from src.probes.core.evaluate import score_with_probe

ROOT = Path(__file__).resolve().parents[2]
ACTIVATIONS_DIR = ROOT / "activations"
PROBES_DIR = ROOT / "results" / "probes"
LABELS_PATH = ROOT / "src" / "task_data" / "data" / "creak.jsonl"
OUTPUT_DIR = ROOT / "experiments" / "truth_probes"
ASSETS_DIR = OUTPUT_DIR / "assets"

FRAMINGS = {
    "raw": ACTIVATIONS_DIR / "gemma_3_27b_creak_raw",
    "repeat": ACTIVATIONS_DIR / "gemma_3_27b_creak_repeat",
}

PROBES = {
    "tb-2": {
        "selector": "turn_boundary:-2",
        "probe_dir": PROBES_DIR / "heldout_eval_gemma3_tb-2" / "probes",
    },
    "tb-5": {
        "selector": "turn_boundary:-5",
        "probe_dir": PROBES_DIR / "heldout_eval_gemma3_tb-5" / "probes",
    },
}

LAYERS = [25, 32, 39, 46, 53]


def load_labels() -> dict[str, str]:
    """Load CREAK labels: {ex_id: 'true'/'false'}."""
    labels = {}
    with open(LABELS_PATH) as f:
        for line in f:
            row = json.loads(line)
            labels[row["ex_id"]] = row["label"]
    return labels


def load_task_ids(framing_dir: Path) -> list[str]:
    """Load task_ids from completions JSON to get ordering."""
    with open(framing_dir / "completions_with_activations.json") as f:
        completions = json.load(f)
    return [c["task_id"] for c in completions]


def compute_metrics(true_scores: np.ndarray, false_scores: np.ndarray) -> dict:
    mean_diff = true_scores.mean() - false_scores.mean()
    pooled_std = np.sqrt(
        (true_scores.var(ddof=1) * (len(true_scores) - 1)
         + false_scores.var(ddof=1) * (len(false_scores) - 1))
        / (len(true_scores) + len(false_scores) - 2)
    )
    cohens_d = mean_diff / pooled_std
    _, p_value = ttest_ind(true_scores, false_scores, equal_var=False)
    return {
        "mean_true": float(true_scores.mean()),
        "mean_false": float(false_scores.mean()),
        "mean_diff": float(mean_diff),
        "cohens_d": float(cohens_d),
        "p_value": float(p_value),
        "n_true": len(true_scores),
        "n_false": len(false_scores),
    }


def run_analysis() -> dict:
    labels = load_labels()
    all_results = {}

    for framing_name, framing_dir in FRAMINGS.items():
        task_ids = load_task_ids(framing_dir)
        all_results[framing_name] = {}

        for probe_name, probe_info in PROBES.items():
            selector = probe_info["selector"]
            act_path = framing_dir / f"activations_{selector}.npz"
            act_task_ids, layer_acts = load_activations(act_path, layers=LAYERS)

            # Build label array aligned to activation order
            act_labels = np.array([labels[tid] for tid in act_task_ids])
            true_mask = act_labels == "true"
            false_mask = act_labels == "false"

            all_results[framing_name][probe_name] = {}

            for layer in LAYERS:
                probe_path = probe_info["probe_dir"] / f"probe_ridge_L{layer}.npy"
                probe_weights = np.load(probe_path)
                scores = score_with_probe(probe_weights, layer_acts[layer])

                true_scores = scores[true_mask]
                false_scores = scores[false_mask]
                metrics = compute_metrics(true_scores, false_scores)
                all_results[framing_name][probe_name][str(layer)] = metrics

                print(
                    f"{framing_name:>6} | {probe_name} | L{layer:02d} | "
                    f"d={metrics['cohens_d']:+.4f} | "
                    f"diff={metrics['mean_diff']:+.4f} | "
                    f"p={metrics['p_value']:.2e} | "
                    f"n_true={metrics['n_true']} n_false={metrics['n_false']}"
                )

    return all_results


def plot_distributions(results: dict, labels: dict, framings: dict, probes: dict):
    """Plot 1: Violin plots for best probe (tb-2 L32), one panel per framing."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, (framing_name, framing_dir) in zip(axes, framings.items()):
        selector = probes["tb-2"]["selector"]
        act_path = framing_dir / f"activations_{selector}.npz"
        act_task_ids, layer_acts = load_activations(act_path, layers=[32])

        probe_path = probes["tb-2"]["probe_dir"] / "probe_ridge_L32.npy"
        probe_weights = np.load(probe_path)
        scores = score_with_probe(probe_weights, layer_acts[32])

        act_labels = np.array([labels[tid] for tid in act_task_ids])
        true_scores = scores[act_labels == "true"]
        false_scores = scores[act_labels == "false"]

        parts = ax.violinplot([false_scores, true_scores], positions=[0, 1], showmeans=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["False", "True"])
        d = results[framing_name]["tb-2"]["32"]["cohens_d"]
        ax.set_title(f"{framing_name.capitalize()} framing (d={d:+.3f})")
        ax.set_ylabel("Preference probe score" if ax == axes[0] else "")

    fig.suptitle("Preference probe scores on true vs false CREAK claims\n(tb-2 probe, layer 32)", fontsize=12)
    plt.tight_layout()
    out = ASSETS_DIR / "plot_031126_truth_probe_score_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_effect_sizes(results: dict):
    """Plot 2: Cohen's d by layer, one line per probe, one panel per framing."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, framing_name in zip(axes, results):
        for probe_name in results[framing_name]:
            ds = [results[framing_name][probe_name][str(l)]["cohens_d"] for l in LAYERS]
            ax.plot(LAYERS, ds, marker="o", label=probe_name)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cohen's d" if ax == axes[0] else "")
        ax.set_title(f"{framing_name.capitalize()} framing")
        ax.legend()
        ax.set_xticks(LAYERS)

    fig.suptitle("Truth effect size (Cohen's d) by layer", fontsize=12)
    plt.tight_layout()
    out = ASSETS_DIR / "plot_031126_truth_effect_size_by_layer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Truth Probes Analysis")
    print("=" * 80)

    results = run_analysis()

    print("\n" + "=" * 80)
    print("Generating plots...")
    labels = load_labels()
    plot_distributions(results, labels, FRAMINGS, PROBES)
    plot_effect_sizes(results)

    out_path = OUTPUT_DIR / "truth_probes_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
