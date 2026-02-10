"""Control analysis: off-target specificity and cross-topic leakage.

For each system prompt, compare probe delta on targeted tasks (2) vs non-targeted tasks (14).
Also check cross-topic leakage between semantically related topics.
"""

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

    # Reverse lookup: topic -> group
    topic_to_group = {}
    for group, topics in TOPIC_GROUPS.items():
        for topic in topics:
            topic_to_group[topic] = group

    coef, intercept = load_probe(31)
    baseline_data = np.load(ACT_DIR / "baseline.npz", allow_pickle=True)
    baseline_scores = baseline_data["layer_31"] @ coef + intercept

    # === 1. On-target vs off-target ===
    print("=" * 80)
    print("ON-TARGET vs OFF-TARGET SPECIFICITY (Layer 31)")
    print("=" * 80)

    on_target_deltas = []
    off_target_deltas = []

    for sp in all_prompts:
        npz_path = ACT_DIR / f"{sp['id']}.npz"
        if not npz_path.exists():
            continue
        manip_data = np.load(npz_path, allow_pickle=True)
        manip_scores = manip_data["layer_31"] @ coef + intercept

        topic = sp["target_topic"]
        target_task_ids = set(topic_to_task_ids[topic])

        for t in targets:
            idx = target_id_to_idx[t["task_id"]]
            delta = float(manip_scores[idx] - baseline_scores[idx])
            if t["task_id"] in target_task_ids:
                on_target_deltas.append(delta)
            else:
                off_target_deltas.append(delta)

    on_arr = np.array(on_target_deltas)
    off_arr = np.array(off_target_deltas)

    print(f"\n  On-target  (n={len(on_arr):>4}): mean={on_arr.mean():>+8.1f}, std={on_arr.std():>7.1f}, |mean|={np.abs(on_arr).mean():>7.1f}")
    print(f"  Off-target (n={len(off_arr):>4}): mean={off_arr.mean():>+8.1f}, std={off_arr.std():>7.1f}, |mean|={np.abs(off_arr).mean():>7.1f}")

    t_abs, p_abs = stats.ttest_ind(np.abs(on_arr), np.abs(off_arr))
    print(f"\n  |on-target| vs |off-target|: t={t_abs:.2f}, p={p_abs:.1e}")

    # Signed comparison by direction
    for direction in ["positive", "negative"]:
        dir_prompts = [p for p in all_prompts if p["direction"] == direction]
        on_d = []
        off_d = []
        for sp in dir_prompts:
            npz_path = ACT_DIR / f"{sp['id']}.npz"
            if not npz_path.exists():
                continue
            manip_data = np.load(npz_path, allow_pickle=True)
            manip_scores = manip_data["layer_31"] @ coef + intercept
            topic = sp["target_topic"]
            target_task_ids = set(topic_to_task_ids[topic])

            for t in targets:
                idx = target_id_to_idx[t["task_id"]]
                delta = float(manip_scores[idx] - baseline_scores[idx])
                if t["task_id"] in target_task_ids:
                    on_d.append(delta)
                else:
                    off_d.append(delta)

        on_a = np.array(on_d)
        off_a = np.array(off_d)
        print(f"\n  {direction.upper()}:")
        print(f"    On-target  mean={on_a.mean():>+8.1f}")
        print(f"    Off-target mean={off_a.mean():>+8.1f}")

    # === 2. Cross-topic leakage ===
    print("\n" + "=" * 80)
    print("CROSS-TOPIC LEAKAGE (Layer 31)")
    print("=" * 80)
    print("\nFor each manipulation, compare probe delta on:")
    print("  - Targeted tasks (same topic)")
    print("  - Same-group tasks (semantically related)")
    print("  - Other-group tasks (unrelated)")

    same_group_deltas = []
    other_group_deltas = []

    for sp in all_prompts:
        npz_path = ACT_DIR / f"{sp['id']}.npz"
        if not npz_path.exists():
            continue
        manip_data = np.load(npz_path, allow_pickle=True)
        manip_scores = manip_data["layer_31"] @ coef + intercept

        topic = sp["target_topic"]
        target_task_ids = set(topic_to_task_ids[topic])
        target_group = topic_to_group[topic]

        # Same-group topics (excluding target topic itself)
        same_group_topics = set(TOPIC_GROUPS[target_group]) - {topic}
        same_group_task_ids = set()
        for sg_topic in same_group_topics:
            same_group_task_ids.update(topic_to_task_ids[sg_topic])

        for t in targets:
            idx = target_id_to_idx[t["task_id"]]
            delta = float(manip_scores[idx] - baseline_scores[idx])
            if t["task_id"] in target_task_ids:
                continue  # Skip on-target
            elif t["task_id"] in same_group_task_ids:
                same_group_deltas.append(delta)
            else:
                other_group_deltas.append(delta)

    same_arr = np.array(same_group_deltas) if same_group_deltas else np.array([0.0])
    other_arr = np.array(other_group_deltas)

    print(f"\n  Same-group off-target  (n={len(same_group_deltas):>4}): mean |delta|={np.abs(same_arr).mean():>7.1f}")
    print(f"  Other-group off-target (n={len(other_group_deltas):>4}): mean |delta|={np.abs(other_arr).mean():>7.1f}")

    if len(same_group_deltas) > 1:
        t_leak, p_leak = stats.ttest_ind(np.abs(same_arr), np.abs(other_arr))
        print(f"  Same-group vs other-group |delta|: t={t_leak:.2f}, p={p_leak:.1e}")

    # === 3. Per-topic breakdown ===
    print("\n" + "=" * 80)
    print("PER-TOPIC ON-TARGET MEAN DELTA (Layer 31)")
    print("=" * 80)

    print(f"\n{'Topic':<20} {'N prompts':<12} {'Mean on-target delta':>22}")
    print("-" * 55)

    for topic in sorted(topic_to_task_ids.keys()):
        topic_prompts = [p for p in all_prompts if p["target_topic"] == topic]
        topic_deltas = []
        for sp in topic_prompts:
            npz_path = ACT_DIR / f"{sp['id']}.npz"
            if not npz_path.exists():
                continue
            manip_data = np.load(npz_path, allow_pickle=True)
            manip_scores = manip_data["layer_31"] @ coef + intercept
            for tid in topic_to_task_ids[topic]:
                idx = target_id_to_idx[tid]
                delta = float(manip_scores[idx] - baseline_scores[idx])
                # Flip sign for negative direction so all deltas are "in expected direction"
                if sp["direction"] == "negative":
                    topic_deltas.append(-delta)
                else:
                    topic_deltas.append(delta)

        arr = np.array(topic_deltas)
        print(f"{topic:<20} {len(topic_prompts):<12} {arr.mean():>+22.1f}")


if __name__ == "__main__":
    main()
