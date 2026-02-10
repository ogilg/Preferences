"""Final analysis: combined statistics, plots, and summary for hidden preferences."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

EXP_DIR = Path("experiments/hidden_preferences")
ACT_DIR = EXP_DIR / "activations"
PROBE_DIR = Path("results/probes/gemma3_3k_completion_preference/probes")
ASSETS_DIR = Path("docs/logs/assets/hidden_preferences")
LAYERS = [31, 43, 55]

TOPIC_COLORS = {
    "cheese": "#e17055",
    "rainy_weather": "#0984e3",
    "cats": "#6c5ce7",
    "classical_music": "#fdcb6e",
    "gardening": "#00b894",
    "astronomy": "#d63031",
    "cooking": "#e84393",
    "ancient_history": "#636e72",
}


def load_probe(layer):
    w = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return w[:-1], w[-1]


def load_results(filename):
    with open(EXP_DIR / "results" / filename) as f:
        return json.load(f)


def main():
    iteration = load_results("probe_behavioral_iteration.json")
    holdout = load_results("probe_behavioral_holdout.json")
    combined = iteration + holdout

    with open(EXP_DIR / "target_tasks.json") as f:
        targets = json.load(f)
    target_id_to_idx = {t["task_id"]: i for i, t in enumerate(targets)}
    topic_to_task_ids = {}
    for t in targets:
        topic_to_task_ids.setdefault(t["topic"], []).append(t["task_id"])

    # Load probes and baseline
    probes = {l: load_probe(l) for l in LAYERS}
    baseline_data = np.load(ACT_DIR / "baseline.npz", allow_pickle=True)
    baseline_scores = {}
    for layer in LAYERS:
        coef, intercept = probes[layer]
        baseline_scores[layer] = baseline_data[f"layer_{layer}"] @ coef + intercept

    # === 1. Combined statistics ===
    print("=" * 80)
    print("COMBINED RESULTS (iteration + holdout)")
    print("=" * 80)

    beh_all = np.array([r["behavioral_delta"] for r in combined])

    for layer in LAYERS:
        probe_all = np.array([r[f"probe_delta_L{layer}"] for r in combined])
        pr, pp = stats.pearsonr(beh_all, probe_all)
        sr, sp_val = stats.spearmanr(beh_all, probe_all)
        sign = np.mean(np.sign(beh_all) == np.sign(probe_all))
        print(f"\nLayer {layer}: Pearson r={pr:.3f} (p={pp:.1e}), Spearman r={sr:.3f} (p={sp_val:.1e}), "
              f"Sign={sign:.1%} ({int(sign*len(combined))}/{len(combined)})")

    # === 2. Summary table ===
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Dataset':<12} {'N':<5} {'Layer':<8} {'Pearson r':<16} {'Spearman r':<16} {'Sign %':<10}")
    print("-" * 70)

    for name, data in [("Iteration", iteration), ("Holdout", holdout), ("Combined", combined)]:
        beh = np.array([r["behavioral_delta"] for r in data])
        for layer in LAYERS:
            probe = np.array([r[f"probe_delta_L{layer}"] for r in data])
            pr, pp = stats.pearsonr(beh, probe)
            sr, sp_val = stats.spearmanr(beh, probe)
            sign = np.mean(np.sign(beh) == np.sign(probe))
            print(f"{name:<12} {len(data):<5} L{layer:<6} r={pr:.3f} p={pp:.1e}  r={sr:.3f} p={sp_val:.1e}  {sign:.1%}")

    # === 3. On-target vs off-target ===
    print("\n" + "=" * 80)
    print("ON-TARGET vs OFF-TARGET (Layer 31)")
    print("=" * 80)

    with open(EXP_DIR / "system_prompts.json") as f:
        iter_prompts = json.load(f)["prompts"]
    with open(EXP_DIR / "holdout_prompts.json") as f:
        holdout_prompts = json.load(f)["prompts"]
    all_prompts = iter_prompts + holdout_prompts

    coef31, intercept31 = probes[31]
    on_target_deltas = []
    off_target_deltas = []

    for sp in all_prompts:
        npz_path = ACT_DIR / f"{sp['id']}.npz"
        if not npz_path.exists():
            continue
        manip_data = np.load(npz_path, allow_pickle=True)
        manip_scores = manip_data["layer_31"] @ coef31 + intercept31

        topic = sp["target_topic"]
        target_task_ids = set(topic_to_task_ids[topic])

        for t_info in targets:
            idx = target_id_to_idx[t_info["task_id"]]
            delta = float(manip_scores[idx] - baseline_scores[31][idx])
            if t_info["task_id"] in target_task_ids:
                on_target_deltas.append(delta)
            else:
                off_target_deltas.append(delta)

    on_arr = np.array(on_target_deltas)
    off_arr = np.array(off_target_deltas)
    t_stat, t_pval = stats.ttest_ind(np.abs(on_arr), np.abs(off_arr))
    print(f"\n  On-target (n={len(on_arr)}): mean |delta|={np.abs(on_arr).mean():.1f}")
    print(f"  Off-target (n={len(off_arr)}): mean |delta|={np.abs(off_arr).mean():.1f}")
    print(f"  t={t_stat:.2f}, p={t_pval:.1e}")

    # === 4. Plots ===
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Left: correlation scatter (L31, colored by topic)
    ax = axes[0]
    for r in combined:
        color = TOPIC_COLORS[r["target_topic"]]
        marker = "^" if r["direction"] == "positive" else "v"
        is_holdout = r["prompt_id"].startswith("holdout_")
        alpha = 0.6 if is_holdout else 1.0
        edgecolor = "black" if is_holdout else "white"
        ax.scatter(r["behavioral_delta"], r["probe_delta_L31"],
                   c=color, marker=marker, s=80, zorder=3,
                   edgecolors=edgecolor, linewidths=0.8, alpha=alpha)

    probe_all_31 = np.array([r["probe_delta_L31"] for r in combined])
    slope, intercept_fit, _, _, _ = stats.linregress(beh_all, probe_all_31)
    x_line = np.linspace(beh_all.min() - 0.05, beh_all.max() + 0.05, 100)
    ax.plot(x_line, slope * x_line + intercept_fit, "k--", alpha=0.4)
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.5)

    pr_all, pp_all = stats.pearsonr(beh_all, probe_all_31)
    sr_all, _ = stats.spearmanr(beh_all, probe_all_31)
    sign_all = np.mean(np.sign(beh_all) == np.sign(probe_all_31))
    ax.set_xlabel("Behavioral Delta", fontsize=12)
    ax.set_ylabel("Probe Delta (Layer 31)", fontsize=12)
    ax.set_title(f"Combined (n={len(combined)})\nr={pr_all:.3f}, rho={sr_all:.3f}, sign={sign_all:.0%}",
                 fontsize=11)

    for topic in sorted(TOPIC_COLORS):
        ax.scatter([], [], c=TOPIC_COLORS[topic], s=40, label=topic.replace("_", " "))
    ax.legend(fontsize=7, loc="upper left")

    # Middle: on-target vs off-target histogram
    ax = axes[1]
    bins = np.linspace(0, max(np.abs(on_arr).max(), np.abs(off_arr).max()), 30)
    ax.hist(np.abs(on_arr), bins=bins, alpha=0.7, label=f"On-target (n={len(on_arr)})", color="#d63031")
    ax.hist(np.abs(off_arr), bins=bins, alpha=0.7, label=f"Off-target (n={len(off_arr)})", color="#636e72")
    ax.set_xlabel("|Probe Delta| (Layer 31)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Specificity Control\n|on|={np.abs(on_arr).mean():.0f} vs |off|={np.abs(off_arr).mean():.0f} (p={t_pval:.1e})",
                 fontsize=11)
    ax.legend()

    # Right: per-topic on-target delta (expected direction)
    ax = axes[2]
    topics = sorted(TOPIC_COLORS.keys())
    topic_means = []
    topic_stds = []
    for topic in topics:
        topic_prompts = [p for p in all_prompts if p["target_topic"] == topic]
        deltas = []
        for sp in topic_prompts:
            npz_path = ACT_DIR / f"{sp['id']}.npz"
            if not npz_path.exists():
                continue
            manip_data = np.load(npz_path, allow_pickle=True)
            manip_scores = manip_data["layer_31"] @ coef31 + intercept31
            for tid in topic_to_task_ids[topic]:
                idx = target_id_to_idx[tid]
                d = float(manip_scores[idx] - baseline_scores[31][idx])
                if sp["direction"] == "negative":
                    deltas.append(-d)
                else:
                    deltas.append(d)
        arr = np.array(deltas)
        topic_means.append(arr.mean())
        topic_stds.append(arr.std() / np.sqrt(len(arr)))

    bars = ax.barh(range(len(topics)), topic_means,
                   xerr=topic_stds,
                   color=[TOPIC_COLORS[t] for t in topics], alpha=0.8)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels([t.replace("_", " ") for t in topics])
    ax.set_xlabel("Mean Probe Delta (expected direction)", fontsize=12)
    ax.set_title("Per-Topic Probe Sensitivity", fontsize=11)
    ax.axvline(x=0, color="gray", linewidth=0.5)

    plt.tight_layout()
    plot_path = ASSETS_DIR / "plot_021026_final_hidden_preferences.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved combined plot to {plot_path}")

    # === 5. Behavioral success rates ===
    print("\n" + "=" * 80)
    print("BEHAVIORAL SUMMARY")
    print("=" * 80)

    for name, filename in [("Iteration", "behavioral_iteration.json"), ("Holdout", "behavioral_holdout.json")]:
        with open(EXP_DIR / "results" / filename) as f:
            beh_data = json.load(f)

        correct = 0
        for r in beh_data:
            expected_sign = -1 if r["direction"] == "negative" else 1
            actual_sign = 1 if r["delta"] > 0 else (-1 if r["delta"] < 0 else 0)
            if actual_sign == expected_sign:
                correct += 1

        abs_deltas = [abs(r["delta"]) for r in beh_data]
        print(f"\n  {name} (n={len(beh_data)}):")
        print(f"    Direction agreement: {correct}/{len(beh_data)} = {correct/len(beh_data):.1%}")
        print(f"    Mean |delta|: {np.mean(abs_deltas):.3f}")
        print(f"    Median |delta|: {np.median(abs_deltas):.3f}")

    # Combined behavioral
    with open(EXP_DIR / "results" / "behavioral_iteration.json") as f:
        beh_iter = json.load(f)
    with open(EXP_DIR / "results" / "behavioral_holdout.json") as f:
        beh_hold = json.load(f)
    all_beh = beh_iter + beh_hold
    correct = sum(
        1 for r in all_beh
        if (1 if r["delta"] > 0 else -1) == (-1 if r["direction"] == "negative" else 1)
    )
    print(f"\n  Combined (n={len(all_beh)}):")
    print(f"    Direction agreement: {correct}/{len(all_beh)} = {correct/len(all_beh):.1%}")


if __name__ == "__main__":
    main()
