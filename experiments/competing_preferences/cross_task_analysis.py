"""Cross-task probe analysis for competing preferences.

For each competing prompt (e.g., "love cheese, hate math"), examine probe
responses across ALL 40 crossed tasks — not just the target. This reveals
whether the probe tracks each topic's evaluation independently.

Key test: Under "love cheese, hate math":
- Other cheese tasks (cheese_coding, cheese_fiction): should shift positive (cheese loved)
- Other math tasks (cats_math, gardening_math): should shift negative (math hated)
- Under "love math, hate cheese": the pattern should reverse

If the probe tracks evaluation, the sign of probe delta for cheese-related tasks
should FLIP between the two conditions. This is a cleaner sign-flip test than
looking at the target task (which mixes both topics).
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

EXP_DIR = Path("experiments/competing_preferences")
CROSSED_DIR = Path("experiments/crossed_preferences")
ACT_DIR = EXP_DIR / "activations"
PROBE_DIR = Path("results/probes/gemma3_3k_completion_preference/probes")
RESULTS_DIR = EXP_DIR / "results"
LAYERS = [31, 43, 55]


def load_probe(layer: int) -> tuple[np.ndarray, float]:
    weights = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return weights[:-1], weights[-1]


def score_activations(activations: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return activations @ coef + intercept


def load_activations(filename: str) -> dict[int, np.ndarray]:
    data = np.load(ACT_DIR / filename, allow_pickle=True)
    return {layer: data[f"layer_{layer}"] for layer in LAYERS}


def load_task_metadata() -> list[dict]:
    with open(CROSSED_DIR / "crossed_tasks.json") as f:
        return json.load(f)


def main():
    tasks = load_task_metadata()
    task_ids = [t["task_id"] for t in tasks]
    task_idx = {tid: i for i, tid in enumerate(task_ids)}
    task_topics = {t["task_id"]: t["topic"] for t in tasks}
    task_shells = {t["task_id"]: t["category_shell"] for t in tasks}

    # Load probe and baseline
    coef, intercept = load_probe(31)
    baseline_acts = load_activations("baseline.npz")
    baseline_scores = score_activations(baseline_acts[31], coef, intercept)

    # Load competing prompts
    with open(EXP_DIR / "competing_prompts.json") as f:
        prompts = json.load(f)["prompts"]

    # Group prompts by pair
    pairs: dict[str, dict] = {}
    for p in prompts:
        key = f"{p['target_topic']}_{p['category_shell']}"
        pairs.setdefault(key, {})
        pairs[key][p["favored_dim"]] = p

    print("=" * 80)
    print("CROSS-TASK PROBE ANALYSIS")
    print("=" * 80)
    print("\nFor each pair, compare probe deltas on tasks that share ONLY the topic")
    print("or ONLY the shell with the target, under each competing condition.\n")

    all_topic_flips = []
    all_shell_flips = []

    for pair_key, pair in sorted(pairs.items()):
        topic_prompt = pair["topic"]
        shell_prompt = pair["shell"]
        target_topic = topic_prompt["target_topic"]
        target_shell = topic_prompt["category_shell"]

        topic_acts = load_activations(f"{topic_prompt['id']}.npz")
        shell_acts = load_activations(f"{shell_prompt['id']}.npz")

        topic_scores = score_activations(topic_acts[31], coef, intercept)
        shell_scores = score_activations(shell_acts[31], coef, intercept)

        topic_deltas = topic_scores - baseline_scores
        shell_deltas = shell_scores - baseline_scores

        # Find tasks that share the topic (but different shell)
        same_topic_tasks = [t for t in tasks
                          if t["topic"] == target_topic and t["category_shell"] != target_shell]
        # Find tasks that share the shell (but different topic)
        same_shell_tasks = [t for t in tasks
                          if t["category_shell"] == target_shell and t["topic"] != target_topic]
        # Find tasks that share neither
        unrelated_tasks = [t for t in tasks
                         if t["topic"] != target_topic and t["category_shell"] != target_shell]

        print(f"\n--- {pair_key} ---")
        print(f"  topic_pos = 'love {target_topic}, hate {target_shell}'")
        print(f"  shell_pos = 'love {target_shell}, hate {target_topic}'")

        # Same-topic tasks (share topic with target)
        print(f"\n  Same-topic tasks (other {target_topic} tasks, n={len(same_topic_tasks)}):")
        topic_d_under_topicpos = []
        topic_d_under_shellpos = []
        for t in same_topic_tasks:
            idx = task_idx[t["task_id"]]
            td = float(topic_deltas[idx])
            sd = float(shell_deltas[idx])
            print(f"    {t['task_id']:<35} topic_pos: {td:+8.1f}  shell_pos: {sd:+8.1f}  diff: {td-sd:+8.1f}")
            topic_d_under_topicpos.append(td)
            topic_d_under_shellpos.append(sd)

        if topic_d_under_topicpos:
            mean_tp = np.mean(topic_d_under_topicpos)
            mean_sp = np.mean(topic_d_under_shellpos)
            print(f"    MEAN: topic_pos={mean_tp:+.1f}, shell_pos={mean_sp:+.1f}, diff={mean_tp-mean_sp:+.1f}")
            # Under topic_pos (love topic), same-topic tasks should get HIGHER probe scores
            # Under shell_pos (hate topic), same-topic tasks should get LOWER probe scores
            # So we expect mean_tp > mean_sp → diff > 0
            all_topic_flips.append({
                "pair": pair_key, "dim": "topic",
                "topic_pos_mean": mean_tp, "shell_pos_mean": mean_sp,
                "diff": mean_tp - mean_sp,
                "expected_sign": "positive",  # topic_pos should give higher scores for topic tasks
            })

        # Same-shell tasks (share shell with target)
        print(f"\n  Same-shell tasks (other {target_shell} tasks, n={len(same_shell_tasks)}):")
        shell_d_under_topicpos = []
        shell_d_under_shellpos = []
        for t in same_shell_tasks:
            idx = task_idx[t["task_id"]]
            td = float(topic_deltas[idx])
            sd = float(shell_deltas[idx])
            print(f"    {t['task_id']:<35} topic_pos: {td:+8.1f}  shell_pos: {sd:+8.1f}  diff: {td-sd:+8.1f}")
            shell_d_under_topicpos.append(td)
            shell_d_under_shellpos.append(sd)

        if shell_d_under_topicpos:
            mean_tp = np.mean(shell_d_under_topicpos)
            mean_sp = np.mean(shell_d_under_shellpos)
            print(f"    MEAN: topic_pos={mean_tp:+.1f}, shell_pos={mean_sp:+.1f}, diff={mean_tp-mean_sp:+.1f}")
            # Under topic_pos (hate shell), same-shell tasks should get LOWER probe scores
            # Under shell_pos (love shell), same-shell tasks should get HIGHER probe scores
            # So we expect mean_tp < mean_sp → diff < 0
            all_shell_flips.append({
                "pair": pair_key, "dim": "shell",
                "topic_pos_mean": mean_tp, "shell_pos_mean": mean_sp,
                "diff": mean_tp - mean_sp,
                "expected_sign": "negative",  # topic_pos should give lower scores for shell tasks (hates shell)
            })

        # Unrelated tasks
        if unrelated_tasks:
            unrelated_tp = [float(topic_deltas[task_idx[t["task_id"]]]) for t in unrelated_tasks]
            unrelated_sp = [float(shell_deltas[task_idx[t["task_id"]]]) for t in unrelated_tasks]
            print(f"\n  Unrelated tasks (n={len(unrelated_tasks)}): "
                  f"topic_pos mean={np.mean(unrelated_tp):+.1f}, shell_pos mean={np.mean(unrelated_sp):+.1f}")

    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY: PROBE RESPONSE TO SAME-TOPIC TASKS UNDER COMPETING CONDITIONS")
    print("=" * 80)
    print(f"\nPrediction: topic_pos - shell_pos should be POSITIVE for same-topic tasks")
    print(f"           (probe gives higher scores to cheese tasks under 'love cheese')")
    print(f"\n{'Pair':<30} {'Mean (topic_pos)':>16} {'Mean (shell_pos)':>16} {'Diff':>10} {'Correct?':>10}")
    print("-" * 85)
    correct_topic = 0
    for r in all_topic_flips:
        correct = r["diff"] > 0
        if correct:
            correct_topic += 1
        print(f"{r['pair']:<30} {r['topic_pos_mean']:>+16.1f} {r['shell_pos_mean']:>+16.1f} "
              f"{r['diff']:>+10.1f} {'YES' if correct else 'NO':>10}")
    print(f"\nCorrect direction: {correct_topic}/{len(all_topic_flips)}")

    print(f"\n{'='*80}")
    print("SUMMARY: PROBE RESPONSE TO SAME-SHELL TASKS UNDER COMPETING CONDITIONS")
    print("=" * 80)
    print(f"\nPrediction: topic_pos - shell_pos should be NEGATIVE for same-shell tasks")
    print(f"           (probe gives lower scores to math tasks under 'hate math')")
    print(f"\n{'Pair':<30} {'Mean (topic_pos)':>16} {'Mean (shell_pos)':>16} {'Diff':>10} {'Correct?':>10}")
    print("-" * 85)
    correct_shell = 0
    for r in all_shell_flips:
        correct = r["diff"] < 0
        if correct:
            correct_shell += 1
        print(f"{r['pair']:<30} {r['topic_pos_mean']:>+16.1f} {r['shell_pos_mean']:>+16.1f} "
              f"{r['diff']:>+10.1f} {'YES' if correct else 'NO':>10}")
    print(f"\nCorrect direction: {correct_shell}/{len(all_shell_flips)}")

    # Overall
    all_diffs_topic = [r["diff"] for r in all_topic_flips]
    all_diffs_shell = [r["diff"] for r in all_shell_flips]

    print(f"\n{'='*80}")
    print("STATISTICAL TESTS")
    print(f"{'='*80}")
    if all_diffs_topic:
        t, p = stats.ttest_1samp(all_diffs_topic, 0)
        print(f"Topic tasks: mean diff = {np.mean(all_diffs_topic):+.1f}, t={t:.2f}, p={p:.3e}")
    if all_diffs_shell:
        t, p = stats.ttest_1samp(all_diffs_shell, 0)
        print(f"Shell tasks: mean diff = {np.mean(all_diffs_shell):+.1f}, t={t:.2f}, p={p:.3e}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "cross_task_analysis.json", "w") as f:
        json.dump({
            "topic_flips": all_topic_flips,
            "shell_flips": all_shell_flips,
        }, f, indent=2)


if __name__ == "__main__":
    main()
