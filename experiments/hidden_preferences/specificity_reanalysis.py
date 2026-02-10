"""Specificity reanalysis: signed deltas, with and without same-group leakage."""

import json
from pathlib import Path

import numpy as np
from scipy import stats

EXP_DIR = Path("experiments/hidden_preferences")
ACT_DIR = EXP_DIR / "activations"
PROBE_DIR = Path("results/probes/gemma3_3k_completion_preference/probes")

TOPIC_GROUPS = {
    "food": ["cheese", "cooking"],
    "nature": ["rainy_weather", "gardening"],
    "animals": ["cats"],
    "arts": ["classical_music"],
    "science": ["astronomy"],
    "history": ["ancient_history"],
}


def load_probe(layer: int) -> tuple[np.ndarray, float]:
    w = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return w[:-1], w[-1]


def main():
    with open(EXP_DIR / "target_tasks.json") as f:
        targets = json.load(f)
    with open(EXP_DIR / "system_prompts.json") as f:
        iter_prompts = json.load(f)["prompts"]
    with open(EXP_DIR / "holdout_prompts.json") as f:
        holdout_prompts = json.load(f)["prompts"]

    all_prompts = iter_prompts + holdout_prompts
    target_id_to_idx = {t["task_id"]: i for i, t in enumerate(targets)}
    topic_to_task_ids = {}
    for t in targets:
        topic_to_task_ids.setdefault(t["topic"], []).append(t["task_id"])

    topic_to_group = {}
    for group, topics in TOPIC_GROUPS.items():
        for topic in topics:
            topic_to_group[topic] = group

    coef, intercept = load_probe(31)
    baseline_data = np.load(ACT_DIR / "baseline.npz", allow_pickle=True)
    baseline_scores = baseline_data["layer_31"] @ coef + intercept

    # Collect deltas by category
    for direction in ["positive", "negative"]:
        dir_prompts = [p for p in all_prompts if p["direction"] == direction]

        on_target = []
        off_target_all = []
        off_target_same_group = []
        off_target_clean = []  # excluding same-group

        for sp in dir_prompts:
            npz_path = ACT_DIR / f"{sp['id']}.npz"
            if not npz_path.exists():
                continue
            manip_data = np.load(npz_path, allow_pickle=True)
            manip_scores = manip_data["layer_31"] @ coef + intercept

            topic = sp["target_topic"]
            target_task_ids = set(topic_to_task_ids[topic])
            target_group = topic_to_group[topic]
            same_group_topics = set(TOPIC_GROUPS[target_group]) - {topic}
            same_group_task_ids = set()
            for sg_topic in same_group_topics:
                same_group_task_ids.update(topic_to_task_ids[sg_topic])

            for t in targets:
                idx = target_id_to_idx[t["task_id"]]
                delta = float(manip_scores[idx] - baseline_scores[idx])
                if t["task_id"] in target_task_ids:
                    on_target.append(delta)
                else:
                    off_target_all.append(delta)
                    if t["task_id"] in same_group_task_ids:
                        off_target_same_group.append(delta)
                    else:
                        off_target_clean.append(delta)

        on_arr = np.array(on_target)
        off_all_arr = np.array(off_target_all)
        off_same_arr = np.array(off_target_same_group) if off_target_same_group else np.array([])
        off_clean_arr = np.array(off_target_clean)

        print(f"\n{'='*80}")
        print(f"{direction.upper()} PROMPTS (n={len(dir_prompts)} prompts)")
        print(f"{'='*80}")
        print(f"\n  Signed mean delta (probe units):")
        print(f"    On-target         (n={len(on_arr):>4}): {on_arr.mean():>+8.1f}")
        print(f"    Off-target (all)  (n={len(off_all_arr):>4}): {off_all_arr.mean():>+8.1f}")
        if len(off_same_arr) > 0:
            print(f"    Off-target (same group)  (n={len(off_same_arr):>4}): {off_same_arr.mean():>+8.1f}")
        print(f"    Off-target (clean, no same-group) (n={len(off_clean_arr):>4}): {off_clean_arr.mean():>+8.1f}")

        # t-tests: on-target vs clean off-target
        t_stat, p_val = stats.ttest_ind(on_arr, off_clean_arr)
        print(f"\n  On-target vs clean off-target (signed): t={t_stat:.2f}, p={p_val:.1e}")

        # Is off-target (clean) significantly different from zero?
        t_zero, p_zero = stats.ttest_1samp(off_clean_arr, 0)
        print(f"  Clean off-target != 0: t={t_zero:.2f}, p={p_zero:.1e}")

        # Ratio
        if abs(off_clean_arr.mean()) > 0.1:
            ratio = abs(on_arr.mean()) / abs(off_clean_arr.mean())
            print(f"  Specificity ratio (|on|/|off_clean|): {ratio:.1f}x")

    # Combined summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"\n{'Direction':<12} {'On-target':>12} {'Off (all)':>12} {'Off (same grp)':>16} {'Off (clean)':>12} {'On vs clean p':>14}")
    print("-" * 80)

    for direction in ["positive", "negative"]:
        dir_prompts = [p for p in all_prompts if p["direction"] == direction]
        on_t, off_all, off_same, off_clean = [], [], [], []

        for sp in dir_prompts:
            npz_path = ACT_DIR / f"{sp['id']}.npz"
            if not npz_path.exists():
                continue
            manip_data = np.load(npz_path, allow_pickle=True)
            manip_scores = manip_data["layer_31"] @ coef + intercept
            topic = sp["target_topic"]
            target_task_ids = set(topic_to_task_ids[topic])
            target_group = topic_to_group[topic]
            same_group_topics = set(TOPIC_GROUPS[target_group]) - {topic}
            same_group_task_ids = set()
            for sg_topic in same_group_topics:
                same_group_task_ids.update(topic_to_task_ids[sg_topic])

            for t in targets:
                idx = target_id_to_idx[t["task_id"]]
                delta = float(manip_scores[idx] - baseline_scores[idx])
                if t["task_id"] in target_task_ids:
                    on_t.append(delta)
                else:
                    off_all.append(delta)
                    if t["task_id"] in same_group_task_ids:
                        off_same.append(delta)
                    else:
                        off_clean.append(delta)

        on_a = np.array(on_t)
        off_all_a = np.array(off_all)
        off_same_a = np.array(off_same) if off_same else np.array([0.0])
        off_clean_a = np.array(off_clean)
        _, p = stats.ttest_ind(on_a, off_clean_a)

        print(f"{direction:<12} {on_a.mean():>+12.1f} {off_all_a.mean():>+12.1f} {off_same_a.mean():>+16.1f} {off_clean_a.mean():>+12.1f} {p:>14.1e}")


if __name__ == "__main__":
    main()
