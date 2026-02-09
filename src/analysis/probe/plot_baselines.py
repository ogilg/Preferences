"""Plot probe R² vs noise baselines comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run(
    probe_manifest_dir: Path,
    baseline_manifest_path: Path,
    output: Path | None = None,
) -> None:
    probe_manifest = json.loads((probe_manifest_dir / "manifest.json").read_text())
    baseline_manifest = json.loads(baseline_manifest_path.read_text())

    # Extract real probe R² per layer
    probes_by_layer = {}
    for p in probe_manifest["probes"]:
        if p["method"] == "ridge":
            probes_by_layer[p["layer"]] = p

    # Extract baselines per (type, layer)
    baselines_by_key: dict[tuple[str, int], dict] = {}
    for b in baseline_manifest["baselines"]:
        baselines_by_key[(b["baseline_type"], b["layer"])] = b

    layers = sorted(probes_by_layer.keys())

    # Data for plotting
    real_r2 = [probes_by_layer[l]["cv_r2_mean"] for l in layers]
    real_std = [probes_by_layer[l]["cv_r2_std"] for l in layers]
    shuffled_r2 = [baselines_by_key[("shuffled_labels", l)]["cv_r2_mean"] for l in layers]
    shuffled_std = [baselines_by_key[("shuffled_labels", l)]["cv_r2_std"] for l in layers]
    random_r2 = [baselines_by_key[("random_activations", l)]["cv_r2_mean"] for l in layers]
    random_std = [baselines_by_key[("random_activations", l)]["cv_r2_std"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(layers))
    width = 0.25

    bars_real = ax.bar(x - width, real_r2, width, yerr=real_std, capsize=4,
                       color="tab:blue", edgecolor="black", alpha=0.85, label="Real probe")
    bars_shuffled = ax.bar(x, shuffled_r2, width, yerr=shuffled_std, capsize=4,
                           color="tab:red", edgecolor="black", alpha=0.7, label="Shuffled labels")
    bars_random = ax.bar(x + width, random_r2, width, yerr=random_std, capsize=4,
                         color="tab:orange", edgecolor="black", alpha=0.7, label="Random activations")

    # Annotate real probe bars
    for bar, mean in zip(bars_real, real_r2):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.03,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cross-validated R²", fontsize=12)
    ax.set_title("Probe R² vs Noise Baselines (Gemma-3-27B, 3k tasks)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if output is None:
        output = Path("src/analysis/probe/plots/plot_020926_baselines_comparison.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output}")


if __name__ == "__main__":
    run(
        probe_manifest_dir=Path("results/probes/gemma3_3k_completion_preference"),
        baseline_manifest_path=Path("results/baselines/gemma3_3k_completion_preference/baselines_manifest.json"),
    )
