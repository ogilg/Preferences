"""Plot followup v2 results: multi-panel summary."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUTPUT_DIR = Path("experiments/steering/revealed_preference/confounders/followup_v2")
PLOT_DIR = Path("docs/logs/assets/steering_confounders")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_results(name: str) -> list[dict]:
    path = OUTPUT_DIR / f"{name}_results.json"
    with open(path) as f:
        return json.load(f)


def compute_pa(results, filters=None):
    filtered = results
    if filters:
        for k, v in filters.items():
            filtered = [r for r in filtered if r.get(k) == v]
    valid = [r for r in filtered if r.get("choice") is not None]
    if not valid:
        return float("nan")
    return sum(1 for r in valid if r["choice"] == "a") / len(valid)


def compute_slope(results, filters=None):
    by_coef = {}
    filtered = results
    if filters:
        for k, v in filters.items():
            filtered = [r for r in filtered if r.get(k) == v]
    for r in filtered:
        if r.get("choice") is None:
            continue
        coef = r["coefficient"]
        if coef not in by_coef:
            by_coef[coef] = []
        by_coef[coef].append(1 if r["choice"] == "a" else 0)
    if len(by_coef) < 2:
        return float("nan"), float("nan"), float("nan")
    coefs = sorted(by_coef.keys())
    x = np.array(coefs)
    y = np.array([np.mean(by_coef[c]) for c in coefs])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, p_value, r_value


def plot_summary():
    """6-panel summary figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    probe = load_results("probe_differential")
    same = load_results("same_task")
    header = load_results("header_only")

    coefs = sorted(set(r["coefficient"] for r in probe))

    # Panel 1: Probe differential (both orderings)
    ax = axes[0, 0]
    for ordering, color, label in [("original", "tab:blue", "Original (A=higher μ)"),
                                    ("swapped", "tab:orange", "Swapped (A=lower μ)")]:
        pas = [compute_pa(probe, {"ordering": ordering, "coefficient": c}) for c in coefs]
        ax.plot(coefs, pas, "o-", color=color, label=label, markersize=5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("P(choose A)")
    ax.set_title("Probe differential (all pairs)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # Panel 2: Same-task
    ax = axes[0, 1]
    pas = [compute_pa(same, {"coefficient": c}) for c in coefs]
    ax.plot(coefs, pas, "o-", color="tab:green", markersize=5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("P(choose A)")
    ax.set_title("Same-task (pure position)")
    ax.set_ylim(0, 1)

    # Panel 3: Header-only
    ax = axes[0, 2]
    for ordering, color, label in [("original", "tab:blue", "Original"),
                                    ("swapped", "tab:orange", "Swapped")]:
        pas = [compute_pa(header, {"ordering": ordering, "coefficient": c}) for c in coefs]
        ax.plot(coefs, pas, "o-", color=color, label=label, markersize=5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("P(choose A)")
    ax.set_title("Header-only steering")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # Panel 4: Probe by Δmu bin
    ax = axes[1, 0]
    bins = ["0-1", "1-2", "2-3", "3-5", "5-20"]
    colors_bin = plt.cm.viridis(np.linspace(0.2, 0.9, len(bins)))
    for b, color in zip(bins, colors_bin):
        bin_data = [r for r in probe if r.get("delta_mu_bin") == b and r.get("ordering") == "original"]
        if bin_data:
            pas = [compute_pa(bin_data, {"coefficient": c}) for c in coefs]
            ax.plot(coefs, pas, "o-", color=color, label=f"Δμ {b}", markersize=4)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("P(choose A)")
    ax.set_title("Probe by Δμ bin (original order)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)

    # Panel 5: Position vs content decomposition by Δmu bin
    ax = axes[1, 1]
    with open(OUTPUT_DIR / "utility_matched_pairs.json") as f:
        pairs = json.load(f)
    pair_info = {p["pair_idx"]: p for p in pairs}

    bin_positions = []
    bin_contents = []
    bin_labels = []
    for b in bins:
        bin_data = [r for r in probe if r.get("delta_mu_bin") == b]
        if not bin_data:
            continue
        s_orig, _, _ = compute_slope(bin_data, {"ordering": "original"})
        s_swap, _, _ = compute_slope(bin_data, {"ordering": "swapped"})
        bin_positions.append((s_orig + s_swap) / 2)
        bin_contents.append((s_orig - s_swap) / 2)
        bin_labels.append(b)

    x_pos = np.arange(len(bin_labels))
    width = 0.35
    ax.bar(x_pos - width / 2, [p * 6000 for p in bin_positions], width, label="Position", color="tab:red", alpha=0.7)
    ax.bar(x_pos + width / 2, [c * 6000 for c in bin_contents], width, label="Content", color="tab:blue", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Δμ bin")
    ax.set_ylabel("ΔP(A) over coef range")
    ax.set_title("Position vs content decomposition")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

    # Panel 6: Δmu vs position-adjusted effect scatter
    ax = axes[1, 2]
    effects_path = OUTPUT_DIR / "delta_mu_effects.json"
    if effects_path.exists():
        with open(effects_path) as f:
            effects = json.load(f)
        delta_mus = [e["delta_mu"] for e in effects]
        content_effects = [e["content_effect"] * 6000 for e in effects]  # Scale to ΔP(A) over range
        position_effects = [e["position_effect"] * 6000 for e in effects]

        ax.scatter(delta_mus, content_effects, alpha=0.4, s=20, color="tab:blue", label="Content")
        ax.scatter(delta_mus, position_effects, alpha=0.4, s=20, color="tab:red", label="Position")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Δμ (utility gap)")
        ax.set_ylabel("ΔP(A) over coef range")
        ax.set_title("Per-pair effects vs Δμ")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = PLOT_DIR / "plot_021126_v2_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")


def plot_random_comparison():
    """Plot random vs probe direction comparison."""
    try:
        random_results = load_results("random_directions")
    except FileNotFoundError:
        print("Random directions not available yet.")
        return

    probe = load_results("probe_differential")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Per-direction ΔP(A) bar chart
    ax = axes[0]
    dir_effects = []
    for dir_idx in range(20):
        dir_data = [r for r in random_results if r["dir_idx"] == dir_idx]
        pa_neg = compute_pa(dir_data, {"coefficient": -3000.0})
        pa_pos = compute_pa(dir_data, {"coefficient": 3000.0})
        dir_effects.append(pa_pos - pa_neg)

    borderline_probe = [r for r in probe
                       if r.get("delta_mu_bin") == "0-1" and r.get("ordering") == "original"]
    probe_pa_neg = compute_pa(borderline_probe, {"coefficient": -3000.0})
    probe_pa_pos = compute_pa(borderline_probe, {"coefficient": 3000.0})
    probe_delta = probe_pa_pos - probe_pa_neg

    sorted_effects = sorted(enumerate(dir_effects), key=lambda x: x[1])
    colors = ["tab:gray"] * len(sorted_effects)
    bar_vals = [e[1] for e in sorted_effects]

    ax.barh(range(len(bar_vals)), bar_vals, color=colors, alpha=0.6)
    ax.axvline(probe_delta, color="tab:red", linewidth=2, label=f"Probe (Δ={probe_delta:+.3f})")
    ax.set_ylabel("Direction (sorted)")
    ax.set_xlabel("ΔP(A)")
    ax.set_title("Probe vs random directions")
    ax.legend(fontsize=8)

    # Panel 2: |ΔP(A)| histogram
    ax = axes[1]
    ax.hist(np.abs(dir_effects), bins=10, color="tab:gray", alpha=0.6, label="Random")
    ax.axvline(abs(probe_delta), color="tab:red", linewidth=2, label=f"Probe |Δ|={abs(probe_delta):.3f}")
    ax.set_xlabel("|ΔP(A)|")
    ax.set_ylabel("Count")
    ax.set_title(f"|Δ| distribution (mean random={np.mean(np.abs(dir_effects)):.3f})")
    ax.legend(fontsize=8)

    # Panel 3: Dose-response for probe vs top/bottom random
    ax = axes[2]
    coefs = sorted(set(r["coefficient"] for r in random_results))

    # Probe
    pas_probe = [compute_pa(borderline_probe, {"coefficient": c}) for c in coefs]
    ax.plot(coefs, pas_probe, "o-", color="tab:red", label="Probe", linewidth=2, markersize=5)

    # Random mean ± std
    random_pas = np.zeros((20, len(coefs)))
    for dir_idx in range(20):
        dir_data = [r for r in random_results if r["dir_idx"] == dir_idx]
        for ci, c in enumerate(coefs):
            random_pas[dir_idx, ci] = compute_pa(dir_data, {"coefficient": c})

    mean_pa = np.nanmean(random_pas, axis=0)
    std_pa = np.nanstd(random_pas, axis=0)
    ax.plot(coefs, mean_pa, "s-", color="tab:gray", label="Random mean", markersize=4)
    ax.fill_between(coefs, mean_pa - std_pa, mean_pa + std_pa, color="tab:gray", alpha=0.2)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("P(choose A)")
    ax.set_title("Dose-response: probe vs random")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = PLOT_DIR / "plot_021126_v2_random_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")


def plot_condition_comparison():
    """Condition comparison: all conditions on one plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    probe = load_results("probe_differential")
    same = load_results("same_task")
    header = load_results("header_only")

    coefs = sorted(set(r["coefficient"] for r in probe))

    # Probe differential (borderline only for fair comparison)
    borderline = [r for r in probe if r.get("delta_mu_bin") == "0-1"]
    for ordering, style, label in [("original", "o-", "Probe (original)"),
                                     ("swapped", "o--", "Probe (swapped)")]:
        pas = [compute_pa(borderline, {"ordering": ordering, "coefficient": c}) for c in coefs]
        ax.plot(coefs, pas, style, label=label, markersize=5)

    # Same-task
    pas = [compute_pa(same, {"coefficient": c}) for c in coefs]
    ax.plot(coefs, pas, "s-", color="tab:green", label="Same-task", markersize=5)

    # Header-only
    for ordering, style, label in [("original", "^-", "Header (original)"),
                                     ("swapped", "^--", "Header (swapped)")]:
        pas = [compute_pa(header, {"ordering": ordering, "coefficient": c}) for c in coefs]
        ax.plot(coefs, pas, style, label=label, markersize=5)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("P(choose A)")
    ax.set_title("All conditions: borderline pairs")
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = PLOT_DIR / "plot_021126_v2_condition_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")


def main():
    plot_summary()
    plot_condition_comparison()
    plot_random_comparison()


if __name__ == "__main__":
    main()
