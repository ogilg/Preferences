"""Smoke-test experiment: draw N=1000 from a standard normal, compute mean/variance/pearson_r.

Outputs:
    experiments/test_basic_compute/results.json
    experiments/test_basic_compute/assets/plot_<mmddYY>_normal_histogram.png
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, pearsonr

SEED = 0
N = 1000
EXP_DIR = Path(__file__).resolve().parents[2] / "experiments" / "test_basic_compute"
ASSETS = EXP_DIR / "assets"


def main() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(N)
    y = rng.standard_normal(N)

    mean = float(x.mean())
    variance = float(x.var(ddof=1))
    r, _ = pearsonr(x, y)

    results = {"seed": SEED, "N": N, "mean": mean, "variance": variance, "pearson_r": float(r)}
    (EXP_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(x, bins=40, density=True, alpha=0.7, edgecolor="white", linewidth=0.5, label=f"sample (N={N})")
    grid = np.linspace(x.min(), x.max(), 200)
    ax.plot(grid, norm.pdf(grid), color="crimson", linewidth=1.5, label="N(0, 1)")
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.set_title(f"Standard normal sample: mean={mean:.3f}, var={variance:.3f}")
    ax.legend(frameon=False)
    fig.tight_layout()

    ASSETS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%m%d%y")
    out = ASSETS / f"plot_{stamp}_normal_histogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(json.dumps(results, indent=2))
    print(f"plot: {out.relative_to(EXP_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
