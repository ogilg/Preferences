"""Plot alpha sweep: train vs val R2 across regularization strengths."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.probes.core.storage import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot alpha sweep (train vs val R²)")
    parser.add_argument("manifest_dir", type=Path, help="Directory with manifest.json")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest_dir)
    ridge_probes = [p for p in manifest["probes"] if p["method"] == "ridge" and "alpha_sweep" in p]

    if not ridge_probes:
        print("No ridge probes with alpha_sweep data found")
        return

    ridge_probes = sorted(ridge_probes, key=lambda p: p["layer"])

    if args.output is None:
        date_str = datetime.now().strftime("%m%d%y")
        args.output = Path(f"src/analysis/probe/plots/plot_{date_str}_alpha_sweep.png")

    fig, axes = plt.subplots(1, len(ridge_probes), figsize=(6 * len(ridge_probes), 5), squeeze=False)

    for ax, probe in zip(axes[0], ridge_probes):
        sweep = probe["alpha_sweep"]
        alphas = [s["alpha"] for s in sweep]
        train_r2 = [s["train_r2"] for s in sweep]
        val_r2_mean = [s["val_r2_mean"] for s in sweep]
        val_r2_std = [s["val_r2_std"] for s in sweep]

        val_r2_mean = np.array(val_r2_mean)
        val_r2_std = np.array(val_r2_std)

        ax.semilogx(alphas, train_r2, "o-", label="Train R²", color="tab:blue")
        ax.semilogx(alphas, val_r2_mean, "o-", label="Val R² (CV mean)", color="tab:orange")
        ax.fill_between(alphas, val_r2_mean - val_r2_std, val_r2_mean + val_r2_std,
                        alpha=0.2, color="tab:orange")

        best_alpha = probe["best_alpha"]
        ax.axvline(best_alpha, color="red", linestyle="--", alpha=0.7, label=f"Best α={best_alpha:.0f}")

        ax.set_xlabel("Alpha (regularization)")
        ax.set_ylabel("R²")
        ax.set_ylim(0, 1)
        ax.set_title(f"Layer {probe['layer']}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Alpha Sweep: Train vs Validation R²", fontsize=13)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
