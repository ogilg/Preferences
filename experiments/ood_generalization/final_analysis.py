"""Final analysis: combined iteration+holdout, controls, and summary statistics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

EXP_DIR = Path("experiments/ood_generalization")
ACT_DIR = EXP_DIR / "activations"
PROBE_DIR = Path("results/probes/gemma3_3k_completion_preference/probes")
ASSETS_DIR = Path("docs/logs/assets/ood_generalization")
LAYERS = [31, 43, 55]


def load_probe(layer):
    w = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return w[:-1], w[-1]


def load_results(filename):
    with open(EXP_DIR / "results" / filename) as f:
        return json.load(f)


def main():
    iteration = load_results("probe_behavioral_comparison.json")
    holdout = load_results("holdout_probe_behavioral.json")
    combined = iteration + holdout

    with open(EXP_DIR / "target_tasks.json") as f:
        targets = json.load(f)
    target_id_to_idx = {t["task_id"]: i for i, t in enumerate(targets)}
    category_to_target = {t["topic"]: t["task_id"] for t in targets}

    # Load probes and baseline
    probes = {l: load_probe(l) for l in LAYERS}
    baseline_data = np.load(ACT_DIR / "baseline.npz", allow_pickle=True)
    baseline_scores = {}
    for layer in LAYERS:
        coef, intercept = probes[layer]
        baseline_scores[layer] = baseline_data[f"layer_{layer}"] @ coef + intercept

    # === 1. Combined correlation ===
    print("=" * 80)
    print("COMBINED RESULTS (iteration + holdout)")
    print("=" * 80)

    beh_all = np.array([r["behavioral_delta"] for r in combined])
    for layer in LAYERS:
        probe_all = np.array([r[f"probe_delta_L{layer}"] for r in combined])
        pr, pp = stats.pearsonr(beh_all, probe_all)
        sr, sp = stats.spearmanr(beh_all, probe_all)
        sign = np.mean(np.sign(beh_all) == np.sign(probe_all))
        print(f"\nLayer {layer}: Pearson r={pr:.3f} (p={pp:.1e}), Spearman r={sr:.3f} (p={sp:.1e}), Sign={sign:.1%} ({int(sign*len(combined))}/{len(combined)})")

    # === 2. Irrelevant manipulation control (L31 only) ===
    print("\n" + "=" * 80)
    print("IRRELEVANT MANIPULATION CONTROL (Layer 31)")
    print("=" * 80)
    print("Probe deltas for NON-TARGETED tasks (should be small/random)")

    # Load all system prompts (iteration + holdout)
    with open(EXP_DIR / "system_prompts.json") as f:
        iter_prompts = json.load(f)["prompts"]
    with open(EXP_DIR / "holdout_prompts.json") as f:
        holdout_prompts = json.load(f)["prompts"]
    all_prompts = iter_prompts + holdout_prompts

    coef31, intercept31 = probes[31]

    on_target_deltas = []
    off_target_deltas = []

    for sp_info in all_prompts:
        npz_path = ACT_DIR / f"{sp_info['id']}.npz"
        if not npz_path.exists():
            continue
        manip_data = np.load(npz_path, allow_pickle=True)
        manip_scores = manip_data["layer_31"] @ coef31 + intercept31

        target_task_id = category_to_target[sp_info["target_category"]]
        target_idx = target_id_to_idx[target_task_id]

        for task_info in targets:
            idx = target_id_to_idx[task_info["task_id"]]
            delta = float(manip_scores[idx] - baseline_scores[31][idx])
            if idx == target_idx:
                on_target_deltas.append(delta)
            else:
                off_target_deltas.append(delta)

    on_arr = np.array(on_target_deltas)
    off_arr = np.array(off_target_deltas)

    print(f"\n  On-target (n={len(on_arr)}):  mean={on_arr.mean():.1f}, std={on_arr.std():.1f}, |mean|={np.abs(on_arr).mean():.1f}")
    print(f"  Off-target (n={len(off_arr)}): mean={off_arr.mean():.1f}, std={off_arr.std():.1f}, |mean|={np.abs(off_arr).mean():.1f}")
    tstat, tpval = stats.ttest_ind(np.abs(on_arr), np.abs(off_arr))
    print(f"  |on-target| vs |off-target| t-test: t={tstat:.2f}, p={tpval:.1e}")

    # === 3. Combined correlation plot ===
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    cat_colors = {
        "math": "#e17055", "coding": "#0984e3", "fiction": "#6c5ce7",
        "knowledge_qa": "#00b894", "content_generation": "#fdcb6e", "harmful_request": "#d63031",
    }

    # Left: iteration + holdout combined (L31)
    ax = axes[0]
    for r in combined:
        color = cat_colors.get(r["target_category"], "gray")
        marker = "^" if r["direction"] == "positive" else "v"
        is_holdout = r["prompt_id"].startswith("holdout_")
        alpha = 0.6 if is_holdout else 1.0
        edgecolor = "black" if is_holdout else "white"
        ax.scatter(r["behavioral_delta"], r["probe_delta_L31"],
                   c=color, marker=marker, s=80, zorder=3,
                   edgecolors=edgecolor, linewidths=0.8, alpha=alpha)

    slope, intercept_fit, _, _, _ = stats.linregress(beh_all, np.array([r["probe_delta_L31"] for r in combined]))
    x_line = np.linspace(-1.1, 0.8, 100)
    ax.plot(x_line, slope * x_line + intercept_fit, "k--", alpha=0.4)
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.5)

    pr_all, pp_all = stats.pearsonr(beh_all, np.array([r["probe_delta_L31"] for r in combined]))
    ax.set_xlabel("Behavioral Delta")
    ax.set_ylabel("Probe Delta (Layer 31)")
    ax.set_title(f"Combined (n={len(combined)}): r={pr_all:.3f} (p={pp_all:.1e})\nFilled=iteration, outlined=holdout")

    for cat in sorted(cat_colors):
        ax.scatter([], [], c=cat_colors[cat], s=40, label=cat)
    ax.legend(fontsize=7, loc="upper left")

    # Right: on-target vs off-target control
    ax = axes[1]
    ax.hist(np.abs(on_arr), bins=20, alpha=0.7, label=f"On-target (n={len(on_arr)})", color="#d63031")
    ax.hist(np.abs(off_arr), bins=20, alpha=0.7, label=f"Off-target (n={len(off_arr)})", color="#636e72")
    ax.set_xlabel("|Probe Delta| (Layer 31)")
    ax.set_ylabel("Count")
    ax.set_title(f"On-target vs Off-target Probe Shifts\n|on|={np.abs(on_arr).mean():.0f} vs |off|={np.abs(off_arr).mean():.0f} (p={tpval:.1e})")
    ax.legend()

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_021026_final_combined.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved combined plot to {ASSETS_DIR / 'plot_021026_final_combined.png'}")

    # === 4. Summary table ===
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Dataset':<12} {'N':<5} {'Layer':<8} {'Pearson r':<12} {'Spearman r':<12} {'Sign %':<10}")
    print("-" * 60)

    for name, data in [("Iteration", iteration), ("Holdout", holdout), ("Combined", combined)]:
        beh = np.array([r["behavioral_delta"] for r in data])
        probe = np.array([r["probe_delta_L31"] for r in data])
        pr, pp = stats.pearsonr(beh, probe)
        sr, sp_val = stats.spearmanr(beh, probe)
        sign = np.mean(np.sign(beh) == np.sign(probe))
        print(f"{name:<12} {len(data):<5} L31      r={pr:.3f} p={pp:.1e}  r={sr:.3f} p={sp_val:.1e}  {sign:.1%}")


if __name__ == "__main__":
    main()
