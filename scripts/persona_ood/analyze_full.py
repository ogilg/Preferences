"""Analyze full-scale persona and targeted prompt results.

Computes per-task deltas, selects qualifying personas/prompts,
and outputs summary statistics.

Usage:
    python -m scripts.persona_ood.analyze_full
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

PART_A_PATH = Path("experiments/probe_generalization/persona_ood/full_results_part_a.json")
PART_B_PATH = Path("experiments/probe_generalization/persona_ood/full_results_part_b.json")
TARGETED_PROMPTS_PATH = Path("experiments/probe_generalization/persona_ood/targeted_prompts.json")
TOPICS_PATH = Path("src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json")


def load_topics():
    with open(TOPICS_PATH) as f:
        raw = json.load(f)
    topics = {}
    for task_id, model_dict in raw.items():
        for model_name, classification in model_dict.items():
            topics[task_id] = classification["primary"]
            break
    return topics


def compute_deltas(results: dict, baseline_key: str = "baseline"):
    """Compute per-task delta = P(choose | condition) - P(choose | baseline)."""
    baseline_rates = results[baseline_key]["task_rates"]
    deltas = {}
    for name, data in results.items():
        if name == baseline_key:
            continue
        condition_rates = data["task_rates"]
        task_deltas = {}
        for tid in baseline_rates:
            if tid in condition_rates:
                b = baseline_rates[tid]["p_choose"]
                c = condition_rates[tid]["p_choose"]
                if not (np.isnan(b) or np.isnan(c)):
                    task_deltas[tid] = c - b
        deltas[name] = task_deltas
    return deltas


def analyze_part_a(results, topics):
    """Analyze persona results: compute deltas, summary stats, select qualifying personas."""
    deltas = compute_deltas(results)

    print("=" * 70)
    print("PART A: PERSONA ANALYSIS")
    print("=" * 70)

    # Summary per persona
    persona_stats = []
    for name, task_deltas in sorted(deltas.items(), key=lambda x: np.mean(np.abs(list(x[1].values())))):
        abs_deltas = np.abs(list(task_deltas.values()))
        stats = {
            "name": name,
            "mean_abs_delta": float(np.mean(abs_deltas)),
            "max_abs_delta": float(np.max(abs_deltas)),
            "n_shifted_02": int(np.sum(abs_deltas > 0.2)),
            "n_shifted_01": int(np.sum(abs_deltas > 0.1)),
            "n_tasks": len(task_deltas),
        }
        persona_stats.append(stats)

    print(f"\n{'Persona':<35} {'Mean |Δ|':>10} {'Max |Δ|':>10} {'|Δ|>0.1':>10} {'|Δ|>0.2':>10}")
    print("-" * 75)
    for s in persona_stats:
        print(f"{s['name']:<35} {s['mean_abs_delta']:>10.3f} {s['max_abs_delta']:>10.3f} {s['n_shifted_01']:>10} {s['n_shifted_02']:>10}")

    # Select qualifying personas: |delta| > 0.1 for meaningful number of tasks
    qualifying = [s for s in persona_stats if s["mean_abs_delta"] > 0.1]
    print(f"\nQualifying personas (mean |Δ| > 0.1): {len(qualifying)}/{len(persona_stats)}")

    # Topic-level analysis for top personas
    top_5 = sorted(persona_stats, key=lambda s: -s["mean_abs_delta"])[:5]
    print(f"\n--- Topic breakdown for top 5 personas ---")
    for s in top_5:
        task_deltas = deltas[s["name"]]
        topic_deltas = defaultdict(list)
        for tid, d in task_deltas.items():
            topic = topics.get(tid, "unknown")
            topic_deltas[topic].append(abs(d))

        print(f"\n  {s['name']} (mean |Δ|={s['mean_abs_delta']:.3f}):")
        for topic in sorted(topic_deltas, key=lambda t: -np.mean(topic_deltas[t])):
            vals = topic_deltas[topic]
            print(f"    {topic:<30} mean |Δ|={np.mean(vals):.3f}  n={len(vals)}")

    return persona_stats, deltas


def analyze_part_b(results, topics):
    """Analyze targeted prompt results: specificity ratio, rank."""
    with open(TARGETED_PROMPTS_PATH) as f:
        targeted = json.load(f)

    target_map = {p["name"]: p["target_task_id"] for p in targeted}
    deltas = compute_deltas(results)

    print("\n" + "=" * 70)
    print("PART B: TARGETED PROMPT ANALYSIS")
    print("=" * 70)

    prompt_stats = []
    for name, task_deltas in deltas.items():
        target_id = target_map.get(name)
        if target_id is None:
            continue

        on_target_delta = task_deltas.get(target_id, float("nan"))
        off_target_deltas = [abs(d) for tid, d in task_deltas.items() if tid != target_id]
        mean_off_target = float(np.mean(off_target_deltas)) if off_target_deltas else float("nan")

        # Specificity ratio
        specificity = abs(on_target_delta) / mean_off_target if mean_off_target > 0 else float("nan")

        # Rank of on-target among all deltas
        all_abs = sorted([(abs(d), tid) for tid, d in task_deltas.items()], reverse=True)
        rank = next((i + 1 for i, (_, tid) in enumerate(all_abs) if tid == target_id), len(all_abs))

        stats = {
            "name": name,
            "target_task_id": target_id,
            "on_target_delta": float(on_target_delta),
            "abs_on_target": float(abs(on_target_delta)),
            "mean_off_target": mean_off_target,
            "specificity_ratio": float(specificity),
            "rank": rank,
            "n_tasks": len(task_deltas),
        }
        prompt_stats.append(stats)

    print(f"\n{'Prompt':<30} {'Target Δ':>10} {'Off-tgt':>10} {'Spec.':>8} {'Rank':>6}")
    print("-" * 70)
    for s in sorted(prompt_stats, key=lambda s: -s["specificity_ratio"]):
        print(f"{s['name']:<30} {s['on_target_delta']:>+10.3f} {s['mean_off_target']:>10.3f} {s['specificity_ratio']:>8.1f} {s['rank']:>6}")

    # Qualifying: |on-target delta| > 0.1, specificity > 3, rank in top 3
    qualifying = [
        s for s in prompt_stats
        if s["abs_on_target"] > 0.1 and s["specificity_ratio"] > 3 and s["rank"] <= 3
    ]
    print(f"\nQualifying prompts (|Δ|>0.1, spec>3, rank≤3): {len(qualifying)}/{len(prompt_stats)}")

    # Relaxed criteria
    relaxed = [
        s for s in prompt_stats
        if s["abs_on_target"] > 0.1 and s["specificity_ratio"] > 2
    ]
    print(f"Relaxed (|Δ|>0.1, spec>2): {len(relaxed)}/{len(prompt_stats)}")

    return prompt_stats, deltas


def main():
    topics = load_topics()

    if PART_A_PATH.exists():
        with open(PART_A_PATH) as f:
            part_a = json.load(f)
        n_conditions = len(part_a)
        print(f"Part A: {n_conditions} conditions loaded")
        persona_stats, persona_deltas = analyze_part_a(part_a, topics)
    else:
        print("Part A results not found")
        persona_stats, persona_deltas = None, None

    if PART_B_PATH.exists():
        with open(PART_B_PATH) as f:
            part_b = json.load(f)
        n_conditions = len(part_b)
        print(f"\nPart B: {n_conditions} conditions loaded")
        prompt_stats, prompt_deltas = analyze_part_b(part_b, topics)
    else:
        print("Part B results not found")
        prompt_stats, prompt_deltas = None, None


if __name__ == "__main__":
    main()
