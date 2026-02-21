"""Plot B vs C (neutral vs anti) delta distributions for minimal pairs v7.

Same style as the existing A-B and A-C plots in the v7 report.
B−C delta: positive means neutral > anti (i.e. anti sentence suppresses the task).
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BEH_PATH = os.path.join(REPO_ROOT, "results/ood/minimal_pairs_v7/behavioral.json")
CFG_PATH = os.path.join(REPO_ROOT, "configs/ood/prompts/minimal_pairs_v7.json")
ASSETS_DIR = os.path.join(
    REPO_ROOT, "experiments/probe_generalization/persona_ood/minimal_pairs/assets"
)

# Target task mapping (from the v7 report — task with largest A-B delta per target)
TARGET_TASKS = {
    "shakespeare": "alpaca_14631",
    "lotr": "stresstest_73_1202_value1",
    "wwii": "stresstest_43_948_value2",
    "chess": "stresstest_54_530_neutral",
    "haiku": "alpaca_13255",
    "simpsons": "wildchat_35599",
    "pyramids": "alpaca_5529",
    "detective": "alpaca_3808",
    "convexhull": "alpaca_13003",
    "evolution": "stresstest_68_582_neutral",
}

TARGET_ORDER = [
    "shakespeare", "lotr", "wwii", "chess", "haiku",
    "simpsons", "pyramids", "detective", "convexhull", "evolution",
]


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    beh = json.load(open(BEH_PATH))
    cfg = json.load(open(CFG_PATH))
    cond_info = {c["condition_id"]: c for c in cfg["conditions"]}

    conditions = beh["conditions"]
    tasks = sorted(conditions["baseline"]["task_rates"].keys())
    base_roles = ["midwest", "brooklyn", "retired", "gradstudent"]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    for idx, target in enumerate(TARGET_ORDER):
        ax = axes[idx]
        target_task = TARGET_TASKS[target]

        # Average B−C delta across all base roles
        bc_deltas = {tid: [] for tid in tasks}
        n_bases = 0

        for role in base_roles:
            cid_b = f"{role}_{target}_B"
            cid_c = f"{role}_{target}_C"
            if cid_b not in conditions or cid_c not in conditions:
                continue
            n_bases += 1
            rates_b = {tid: v["p_choose"] for tid, v in conditions[cid_b]["task_rates"].items()}
            rates_c = {tid: v["p_choose"] for tid, v in conditions[cid_c]["task_rates"].items()}
            for tid in tasks:
                if tid in rates_b and tid in rates_c:
                    bc_deltas[tid].append(rates_b[tid] - rates_c[tid])

        # Average across bases
        mean_deltas = {}
        for tid in tasks:
            if bc_deltas[tid]:
                mean_deltas[tid] = np.mean(bc_deltas[tid])

        # Sort by delta descending
        sorted_tasks = sorted(mean_deltas.keys(), key=lambda t: mean_deltas[t], reverse=True)
        values = [mean_deltas[t] for t in sorted_tasks]
        colors = ["#e41a1c" if t == target_task else "#cccccc" for t in sorted_tasks]

        ax.bar(range(len(values)), values, color=colors, width=1.0, edgecolor="none")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlim(-1, len(values))
        ax.set_ylim(-0.3, 1.0)
        ax.set_xticks([])

        target_delta = mean_deltas.get(target_task, 0)
        ax.set_title(f"{target} (Δ={target_delta:+.2f}, {n_bases} bases)", fontsize=9)
        ax.set_ylabel("B−C delta" if idx % 5 == 0 else "", fontsize=9)

        # Compute specificity
        off_target = [abs(mean_deltas[t]) for t in sorted_tasks if t != target_task]
        mean_off = np.mean(off_target) if off_target else 0
        specificity = abs(target_delta) / mean_off if mean_off > 0 else float("inf")
        rank = sorted_tasks.index(target_task) + 1 if target_task in sorted_tasks else -1

        ax.text(
            0.97, 0.97,
            f"rank: {rank}\nspec: {specificity:.1f}x",
            transform=ax.transAxes, fontsize=7, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    fig.suptitle(
        "Specificity of B−C (neutral vs anti): delta across all 50 tasks (target in red)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    save_path = os.path.join(ASSETS_DIR, "plot_022126_delta_distributions_bc.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Print summary table
    print("\n| Target | B−C Δ | Specificity | Rank | Hits |")
    print("|--------|:-----:|:-----------:|:----:|:----:|")
    for target in TARGET_ORDER:
        target_task = TARGET_TASKS[target]
        per_base_deltas = []
        per_base_hits = 0
        for role in base_roles:
            cid_b = f"{role}_{target}_B"
            cid_c = f"{role}_{target}_C"
            if cid_b not in conditions or cid_c not in conditions:
                continue
            rates_b = {tid: v["p_choose"] for tid, v in conditions[cid_b]["task_rates"].items()}
            rates_c = {tid: v["p_choose"] for tid, v in conditions[cid_c]["task_rates"].items()}
            bc = {tid: rates_b[tid] - rates_c[tid] for tid in tasks if tid in rates_b and tid in rates_c}
            sorted_bc = sorted(bc.keys(), key=lambda t: bc[t], reverse=True)
            rank = sorted_bc.index(target_task) + 1 if target_task in sorted_bc else -1
            off = [abs(bc[t]) for t in sorted_bc if t != target_task]
            spec = abs(bc[target_task]) / np.mean(off) if np.mean(off) > 0 else 0
            if rank <= 3 and spec >= 2:
                per_base_hits += 1
            per_base_deltas.append(bc.get(target_task, 0))

        mean_delta = np.mean(per_base_deltas)
        # Mean specificity across bases
        all_mean_deltas = {}
        for tid in tasks:
            vals = []
            for role in base_roles:
                cid_b = f"{role}_{target}_B"
                cid_c = f"{role}_{target}_C"
                if cid_b not in conditions or cid_c not in conditions:
                    continue
                rates_b = {t: v["p_choose"] for t, v in conditions[cid_b]["task_rates"].items()}
                rates_c = {t: v["p_choose"] for t, v in conditions[cid_c]["task_rates"].items()}
                if tid in rates_b and tid in rates_c:
                    vals.append(rates_b[tid] - rates_c[tid])
            if vals:
                all_mean_deltas[tid] = np.mean(vals)

        sorted_all = sorted(all_mean_deltas.keys(), key=lambda t: all_mean_deltas[t], reverse=True)
        rank = sorted_all.index(target_task) + 1
        off = [abs(all_mean_deltas[t]) for t in sorted_all if t != target_task]
        spec = abs(all_mean_deltas[target_task]) / np.mean(off)
        print(f"| {target} | {mean_delta:+.2f} | {spec:.1f}x | {rank} | {per_base_hits}/4 |")


if __name__ == "__main__":
    main()
