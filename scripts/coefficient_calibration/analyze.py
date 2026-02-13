"""Phase 3: Analyze judged results and generate plots for coefficient calibration."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path("experiments/steering_program/coefficient_calibration")
INPUT_PATH = OUTPUT_DIR / "judged_results.json"
ASSETS_DIR = OUTPUT_DIR / "assets"


def load_data() -> dict:
    with open(INPUT_PATH) as f:
        return json.load(f)


def extract_by_category(results: list[dict]) -> dict[str, list[dict]]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    return dict(by_cat)


def compute_coherence_by_coef(results: list[dict]) -> dict[int, list[int]]:
    by_coef: dict[int, list[int]] = defaultdict(list)
    for r in results:
        if isinstance(r.get("coherence"), int):
            by_coef[r["coefficient"]].append(r["coherence"])
    return dict(by_coef)


def compute_valence_by_coef(results: list[dict]) -> dict[int, list[float]]:
    by_coef: dict[int, list[float]] = defaultdict(list)
    for r in results:
        if isinstance(r.get("valence"), float):
            by_coef[r["coefficient"]].append(r["valence"])
    return dict(by_coef)


def plot_coherence_by_category(data: dict, categories: list[str]):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, cat in enumerate(categories):
        ax = axes[idx]
        cat_results = data.get(cat, [])
        coh_by_coef = compute_coherence_by_coef(cat_results)

        coefs = sorted(coh_by_coef.keys())
        means = [np.mean(coh_by_coef[c]) for c in coefs]
        stds = [np.std(coh_by_coef[c]) / np.sqrt(len(coh_by_coef[c])) for c in coefs]

        ax.errorbar(coefs, means, yerr=stds, marker="o", capsize=3)
        ax.axhline(y=4.0, color="r", linestyle="--", alpha=0.5, label="coherence=4 threshold")
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Mean Coherence (1-5)")
        ax.set_title(cat)
        ax.set_ylim(0.5, 5.5)
        ax.legend(fontsize=8)

    plt.suptitle("Coherence vs Steering Coefficient by Category", fontsize=14)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_021326_coherence_by_category.png", dpi=150)
    plt.close()


def plot_valence_dose_response(data: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, cat in enumerate(["D_valence", "F_affect"]):
        ax = axes[idx]
        cat_results = data.get(cat, [])
        val_by_coef = compute_valence_by_coef(cat_results)

        coefs = sorted(val_by_coef.keys())
        means = [np.mean(val_by_coef[c]) for c in coefs]
        stds = [np.std(val_by_coef[c]) / np.sqrt(len(val_by_coef[c])) for c in coefs]

        ax.errorbar(coefs, means, yerr=stds, marker="o", capsize=3, color="blue")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Mean Valence (-1 to +1)")
        ax.set_title(f"Valence Dose-Response: {cat}")
        ax.set_ylim(-1.1, 1.1)

        # Spearman correlation
        all_coefs_flat = []
        all_vals_flat = []
        for c in coefs:
            for v in val_by_coef[c]:
                all_coefs_flat.append(c)
                all_vals_flat.append(v)
        if all_coefs_flat:
            rho, pval = stats.spearmanr(all_coefs_flat, all_vals_flat)
            ax.text(0.05, 0.95, f"Spearman ρ={rho:.3f}\np={pval:.2e}",
                    transform=ax.transAxes, fontsize=10, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Valence Dose-Response (Categories D, F)", fontsize=14)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_021326_valence_dose_response.png", dpi=150)
    plt.close()


def plot_parse_rates(data: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Category A: parse success for pairwise (Task A / Task B)
    ax = axes[0]
    cat_results = data.get("A_pairwise", [])
    by_coef: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "parsed": 0})
    for r in cat_results:
        coef = r["coefficient"]
        by_coef[coef]["total"] += 1
        resp = r["response"].strip().lower()
        if "task a" in resp or "task b" in resp:
            by_coef[coef]["parsed"] += 1

    coefs = sorted(by_coef.keys())
    rates = [by_coef[c]["parsed"] / max(by_coef[c]["total"], 1) for c in coefs]
    ax.bar(range(len(coefs)), rates, tick_label=[str(c) for c in coefs])
    ax.axhline(y=0.9, color="r", linestyle="--", alpha=0.5, label="90% threshold")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Parse Success Rate")
    ax.set_title("A: Pairwise Choice Parse Rate")
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    # Category B: parse success for rating (good / bad)
    ax = axes[1]
    cat_results = data.get("B_rating", [])
    by_coef2: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "parsed": 0})
    for r in cat_results:
        coef = r["coefficient"]
        by_coef2[coef]["total"] += 1
        resp = r["response"].strip().lower()
        if resp in ("good", "bad") or resp.startswith("good") or resp.startswith("bad"):
            by_coef2[coef]["parsed"] += 1

    coefs = sorted(by_coef2.keys())
    rates = [by_coef2[c]["parsed"] / max(by_coef2[c]["total"], 1) for c in coefs]
    ax.bar(range(len(coefs)), rates, tick_label=[str(c) for c in coefs])
    ax.axhline(y=0.9, color="r", linestyle="--", alpha=0.5, label="90% threshold")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Parse Success Rate")
    ax.set_title("B: Rating Parse Rate")
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    plt.suptitle("Parse Success Rates (Categories A, B)", fontsize=14)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_021326_parse_rates.png", dpi=150)
    plt.close()


def plot_response_length(data: dict, categories: list[str]):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, cat in enumerate(categories):
        ax = axes[idx]
        cat_results = data.get(cat, [])
        by_coef: dict[int, list[int]] = defaultdict(list)
        for r in cat_results:
            by_coef[r["coefficient"]].append(r["response_length"])

        coefs = sorted(by_coef.keys())
        means = [np.mean(by_coef[c]) for c in coefs]
        stds = [np.std(by_coef[c]) / np.sqrt(len(by_coef[c])) for c in coefs]

        ax.errorbar(coefs, means, yerr=stds, marker="o", capsize=3)
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Mean Response Length (chars)")
        ax.set_title(cat)

    plt.suptitle("Response Length vs Steering Coefficient by Category", fontsize=14)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_021326_response_length.png", dpi=150)
    plt.close()


def compute_coherent_range(data: dict, threshold: float = 4.0) -> tuple[int, int]:
    """Find the coefficient range where mean coherence >= threshold across all categories."""
    all_results = []
    for cat_results in data.values():
        all_results.extend(cat_results)

    coh_by_coef = compute_coherence_by_coef(all_results)
    coefs = sorted(coh_by_coef.keys())

    coherent_coefs = [c for c in coefs if np.mean(coh_by_coef[c]) >= threshold]
    if not coherent_coefs:
        return (0, 0)
    return (min(coherent_coefs), max(coherent_coefs))


def print_summary(data: dict, categories: list[str]):
    print("=" * 60)
    print("COEFFICIENT CALIBRATION SUMMARY")
    print("=" * 60)

    # Overall coherence
    print("\n--- Coherence by coefficient (all categories) ---")
    all_results = []
    for cat in categories:
        all_results.extend(data.get(cat, []))
    coh_by_coef = compute_coherence_by_coef(all_results)
    for c in sorted(coh_by_coef.keys()):
        vals = coh_by_coef[c]
        print(f"  coef={c:+6d}: mean={np.mean(vals):.2f}, n={len(vals)}")

    # Coherent range
    lo, hi = compute_coherent_range(data)
    print(f"\nCoherent range (mean coh >= 4.0): [{lo}, {hi}]")

    # Valence dose-response
    print("\n--- Valence dose-response ---")
    for cat in ["D_valence", "F_affect"]:
        cat_results = data.get(cat, [])
        val_by_coef = compute_valence_by_coef(cat_results)
        coefs_flat = []
        vals_flat = []
        for c in sorted(val_by_coef.keys()):
            for v in val_by_coef[c]:
                coefs_flat.append(c)
                vals_flat.append(v)
        if coefs_flat:
            rho, pval = stats.spearmanr(coefs_flat, vals_flat)
            print(f"  {cat}: Spearman ρ={rho:.3f}, p={pval:.2e}, n={len(coefs_flat)}")

    # Rating shifts (Category B)
    print("\n--- Category B: Rating shifts ---")
    cat_results = data.get("B_rating", [])
    by_coef_rating: dict[int, dict[str, int]] = defaultdict(lambda: {"good": 0, "bad": 0, "other": 0})
    for r in cat_results:
        coef = r["coefficient"]
        resp = r["response"].strip().lower()
        if resp.startswith("good"):
            by_coef_rating[coef]["good"] += 1
        elif resp.startswith("bad"):
            by_coef_rating[coef]["bad"] += 1
        else:
            by_coef_rating[coef]["other"] += 1
    for c in sorted(by_coef_rating.keys()):
        d = by_coef_rating[c]
        total = d["good"] + d["bad"] + d["other"]
        good_rate = d["good"] / max(total, 1)
        print(f"  coef={c:+6d}: good={good_rate:.2f}, n={total}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading judged results...")
    raw = load_data()
    results = raw["results"]
    print(f"Loaded {len(results)} results")

    categories = ["A_pairwise", "B_rating", "C_completion", "D_valence", "E_neutral", "F_affect"]
    data = extract_by_category(results)

    print("\nGenerating plots...")
    plot_coherence_by_category(data, categories)
    print("  Saved coherence_by_category.png")

    plot_valence_dose_response(data)
    print("  Saved valence_dose_response.png")

    plot_parse_rates(data)
    print("  Saved parse_rates.png")

    plot_response_length(data, categories)
    print("  Saved response_length.png")

    print_summary(data, categories)


if __name__ == "__main__":
    main()
