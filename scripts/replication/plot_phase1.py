import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("experiments/steering/replication/results/steering_phase1.json")
ASSETS_DIR = Path("experiments/steering/replication/assets")

STEERED_CHOICE = {
    "boost_a": "a",
    "boost_b": "b",
    "suppress_a": "a",
    "suppress_b": "b",
    "diff_ab": "a",
    "diff_ba": "b",
}

CONDITION_ORDER = ["boost_a", "boost_b", "suppress_a", "suppress_b", "diff_ab", "diff_ba"]

COLOR_A = "#2171b5"
COLOR_B = "#d94801"
CONDITION_COLORS = {
    "boost_a": COLOR_A,
    "boost_b": COLOR_B,
    "suppress_a": COLOR_A,
    "suppress_b": COLOR_B,
    "diff_ab": COLOR_A,
    "diff_ba": COLOR_B,
}


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def compute_control_p_a(results):
    control_entries = [r for r in results if r["condition"] == "control"]
    per_entry_p_a = []
    for r in control_entries:
        valid = [x for x in r["responses"] if x != "parse_fail"]
        if valid:
            per_entry_p_a.append(sum(1 for x in valid if x == "a") / len(valid))
    return float(np.mean(per_entry_p_a))


def compute_p_steered(results, condition, coefficient):
    target = STEERED_CHOICE[condition]
    entries = [
        r for r in results
        if r["condition"] == condition and abs(r["coefficient"] - coefficient) < 1.0
    ]
    all_valid = [x for r in entries for x in r["responses"] if x != "parse_fail"]
    if not all_valid:
        return np.nan, np.nan, np.nan
    p = sum(1 for x in all_valid if x == target) / len(all_valid)
    n = len(all_valid)
    se = np.sqrt(p * (1 - p) / n)
    ci_half = 1.96 * se
    return p, p - ci_half, p + ci_half


def bootstrap_p_steered(results, condition, coefficient, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    target = STEERED_CHOICE[condition]
    entries = [
        r for r in results
        if r["condition"] == condition and abs(r["coefficient"] - coefficient) < 1.0
    ]
    # Pool into per-entry fractions for bootstrap (resample entries)
    entry_fracs = []
    entry_ns = []
    for r in entries:
        valid = [x for x in r["responses"] if x != "parse_fail"]
        if valid:
            entry_fracs.append(sum(1 for x in valid if x == target) / len(valid))
            entry_ns.append(len(valid))
    if not entry_fracs:
        return np.nan, np.nan, np.nan
    entry_fracs = np.array(entry_fracs)
    entry_ns = np.array(entry_ns)
    point = float(np.average(entry_fracs, weights=entry_ns))
    boot_means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(entry_fracs), size=len(entry_fracs))
        boot_means.append(np.average(entry_fracs[idx], weights=entry_ns[idx]))
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return point, float(lo), float(hi)


def collect_condition_data(results, condition):
    entries = [r for r in results if r["condition"] == condition]
    coefs = sorted(set(r["coefficient"] for r in entries))
    points, lows, highs = [], [], []
    for coef in coefs:
        p, lo, hi = bootstrap_p_steered(results, condition, coef)
        points.append(p)
        lows.append(lo)
        highs.append(hi)
    return np.array(coefs), np.array(points), np.array(lows), np.array(highs)


def plot_condition_ax(ax, coefs, points, lows, highs, control_p, color, title):
    neg_mask = coefs < 0
    pos_mask = coefs >= 0

    # Grey shading for negative coefficient region
    x_min = coefs.min() * 1.15
    ax.axvspan(x_min, 0, color="lightgrey", alpha=0.35, zorder=0)

    # Control line
    ax.axhline(control_p, color="grey", linestyle="--", linewidth=1.2, label="Control", zorder=2)

    # CI band
    ax.fill_between(coefs, lows, highs, color=color, alpha=0.18, zorder=3)

    # Mean line + points
    ax.plot(coefs, points, color=color, linewidth=1.8, zorder=4)
    ax.scatter(coefs, points, color=color, s=40, zorder=5)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Coefficient", fontsize=9)
    ax.set_ylabel("P(steered task picked)", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlim(x_min, coefs.max() * 1.15)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Shared x-axis formatting: use integer labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")


def make_plot1(data):
    results = data["results"]
    control_p_a = compute_control_p_a(results)
    control_p_b = 1.0 - control_p_a

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, condition in enumerate(CONDITION_ORDER):
        target = STEERED_CHOICE[condition]
        control_p = control_p_a if target == "a" else control_p_b
        coefs, points, lows, highs = collect_condition_data(results, condition)
        color = CONDITION_COLORS[condition]
        plot_condition_ax(axes[i], coefs, points, lows, highs, control_p, color, condition)

    fig.suptitle("Phase 1: Steering Conditions — P(steered task picked)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = ASSETS_DIR / "plot_022226_phase1_conditions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


def make_plot2(data):
    results = data["results"]
    control_p_a = compute_control_p_a(results)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    key_conditions = ["diff_ab", "boost_a"]

    for ax, condition in zip(axes, key_conditions):
        coefs, points, lows, highs = collect_condition_data(results, condition)
        color = CONDITION_COLORS[condition]
        target_coef = 2641.14

        # Find point at +2641
        idx = np.argmin(np.abs(coefs - target_coef))
        p_at_target = points[idx]
        effect_pp = (p_at_target - control_p_a) * 100

        x_min = coefs.min() * 1.15
        ax.axvspan(x_min, 0, color="lightgrey", alpha=0.35, zorder=0)
        ax.axhline(control_p_a, color="grey", linestyle="--", linewidth=1.4, label=f"Control P(a)={control_p_a:.3f}", zorder=2)
        ax.fill_between(coefs, lows, highs, color=color, alpha=0.18, zorder=3)
        ax.plot(coefs, points, color=color, linewidth=2.0, zorder=4)
        ax.scatter(coefs, points, color=color, s=50, zorder=5)

        # Annotation at +2641
        sign = "+" if effect_pp >= 0 else ""
        ax.annotate(
            f"{sign}{effect_pp:.1f}pp vs control",
            xy=(coefs[idx], p_at_target),
            xytext=(coefs[idx] - 800, p_at_target + 0.05),
            fontsize=9,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8),
        )

        ax.set_title(condition, fontsize=12, fontweight="bold")
        ax.set_xlabel("Coefficient", fontsize=10)
        ax.set_ylabel("P(steered task picked)", fontsize=10)
        ax.set_ylim(0.3, 0.9)
        ax.set_xlim(x_min, coefs.max() * 1.15)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
        ax.legend(fontsize=9)

    fig.suptitle("Phase 1: L31 Steering — Key Conditions", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = ASSETS_DIR / "plot_022226_phase1_key_conditions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    data = load_data()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    make_plot1(data)
    make_plot2(data)
