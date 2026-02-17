"""Analyze persona OOD v2 results: compute deltas, evaluate success criteria.

Usage:
    python -m scripts.persona_ood_v2.analyze \
        --results experiments/probe_generalization/persona_ood/v2_results.json \
        --config experiments/probe_generalization/persona_ood/v2_config.json \
        --output-dir experiments/probe_generalization/persona_ood
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TOPICS_PATH = Path("src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json")


def load_topics() -> dict[str, str]:
    with open(TOPICS_PATH) as f:
        topics_raw = json.load(f)
    topics = {}
    for task_id, model_dict in topics_raw.items():
        for model_name, classification in model_dict.items():
            topics[task_id] = classification["primary"]
            break
    return topics


def compute_deltas(results: dict, min_n: int = 5) -> dict[str, dict[str, float]]:
    """Compute per-task deltas (persona - baseline) for all conditions."""
    baseline_rates = results["baseline"]["task_rates"]
    all_deltas = {}

    for cond_name, cond_data in results.items():
        if cond_name == "baseline":
            continue
        cond_rates = cond_data["task_rates"]
        deltas = {}
        for tid in baseline_rates:
            if tid in cond_rates:
                b = baseline_rates[tid]
                c = cond_rates[tid]
                if b["n_total"] >= min_n and c["n_total"] >= min_n:
                    deltas[tid] = c["p_choose"] - b["p_choose"]
        all_deltas[cond_name] = deltas

    return all_deltas


def compute_category_deltas(
    deltas: dict[str, float], topics: dict[str, str]
) -> dict[str, list[float]]:
    """Group per-task deltas by topic category."""
    topic_deltas: dict[str, list[float]] = defaultdict(list)
    for task_id, delta in deltas.items():
        topic = topics.get(task_id, "unknown")
        topic_deltas[topic].append(delta)
    return dict(topic_deltas)


def evaluate_broad_persona(
    name: str,
    deltas: dict[str, float],
    topics: dict[str, str],
    expected_shifts: str,
) -> dict:
    """Evaluate a broad persona against expected category shifts."""
    cat_deltas = compute_category_deltas(deltas, topics)

    # Parse expected shifts
    positive_cats = []
    negative_cats = []
    for part in expected_shifts.split(";"):
        part = part.strip()
        for token in part.split(","):
            token = token.strip()
            if token.startswith("+"):
                cat = token[1:].split("(")[0].strip()
                positive_cats.append(cat)
            elif token.startswith("-") and token[1:].strip() not in (""):
                cat = token[1:].split("(")[0].strip()
                negative_cats.append(cat)

    result = {
        "name": name,
        "n_tasks": len(deltas),
        "mean_delta": float(np.mean(list(deltas.values()))),
        "category_means": {},
        "expected_positive": positive_cats,
        "expected_negative": negative_cats,
        "hits": 0,
        "total_expected": len(positive_cats) + len(negative_cats),
    }

    for cat, vals in sorted(cat_deltas.items()):
        mean_d = float(np.mean(vals))
        result["category_means"][cat] = {"mean": mean_d, "n": len(vals)}

        # Check if direction matches expectation
        if cat in positive_cats and mean_d > 0.05:
            result["hits"] += 1
        elif cat in negative_cats and mean_d < -0.05:
            result["hits"] += 1

    return result


def evaluate_narrow_persona(
    name: str,
    deltas: dict[str, float],
    target_task: str,
) -> dict:
    """Evaluate a narrow persona: on-target delta, specificity, rank."""
    on_target = deltas.get(target_task, float("nan"))
    off_target = [d for tid, d in deltas.items() if tid != target_task]

    if not off_target:
        return {"name": name, "target_task": target_task, "error": "no off-target data"}

    mean_off = float(np.mean(np.abs(off_target)))
    specificity = abs(on_target) / mean_off if mean_off > 0 else float("inf")

    # Rank of target task by absolute delta
    sorted_tasks = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
    rank = next(
        (i + 1 for i, (tid, _) in enumerate(sorted_tasks) if tid == target_task),
        len(sorted_tasks),
    )

    return {
        "name": name,
        "target_task": target_task,
        "on_target_delta": float(on_target),
        "mean_abs_off_target": mean_off,
        "specificity_ratio": float(specificity),
        "rank": rank,
        "n_tasks": len(deltas),
        "passes": abs(on_target) > 0.1 and specificity > 3 and rank <= 3,
    }


def print_results(broad_results: list[dict], narrow_results: list[dict]):
    print("\n" + "=" * 70)
    print("PART A: BROAD PERSONAS")
    print("=" * 70)

    for r in broad_results:
        print(f"\n--- {r['name']} ---")
        print(f"  Tasks: {r['n_tasks']}, Overall mean delta: {r['mean_delta']:+.4f}")
        print(f"  Expected hits: {r['hits']}/{r['total_expected']}")
        print(f"  Category means:")
        for cat, info in sorted(r["category_means"].items(), key=lambda x: x[1]["mean"], reverse=True):
            marker = ""
            if cat in r["expected_positive"]:
                marker = " [+expected]" + (" HIT" if info["mean"] > 0.05 else " MISS")
            elif cat in r["expected_negative"]:
                marker = " [-expected]" + (" HIT" if info["mean"] < -0.05 else " MISS")
            print(f"    {cat:25s}: {info['mean']:+.4f} (n={info['n']}){marker}")

    print("\n" + "=" * 70)
    print("PART B: NARROW PERSONAS")
    print("=" * 70)

    for r in narrow_results:
        if "error" in r:
            print(f"\n--- {r['name']} --- ERROR: {r['error']}")
            continue
        status = "PASS" if r["passes"] else "FAIL"
        print(f"\n--- {r['name']} ({status}) ---")
        print(f"  Target: {r['target_task']}")
        print(f"  On-target delta: {r['on_target_delta']:+.4f} (need |delta| > 0.1)")
        print(f"  Specificity ratio: {r['specificity_ratio']:.2f} (need > 3)")
        print(f"  Rank: {r['rank']}/{r['n_tasks']} (need top 3)")

    # Summary
    n_broad_pass = sum(1 for r in broad_results if r["hits"] > 0)
    n_narrow_pass = sum(1 for r in narrow_results if r.get("passes", False))
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"  Broad: {n_broad_pass}/{len(broad_results)} with ≥1 expected hit (need 5/10)")
    print(f"  Narrow: {n_narrow_pass}/{len(narrow_results)} passing (need 5/10)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)
    with open(args.config) as f:
        config = json.load(f)

    topics = load_topics()
    all_deltas = compute_deltas(results)

    persona_lookup = {p["name"]: p for p in config["personas"]}

    broad_results = []
    narrow_results = []

    for cond_name, deltas in all_deltas.items():
        persona = persona_lookup[cond_name]
        if persona["part"] == "A":
            r = evaluate_broad_persona(
                cond_name, deltas, topics, persona["expected_shifts"]
            )
            broad_results.append(r)
        elif persona["part"] == "B":
            r = evaluate_narrow_persona(
                cond_name, deltas, persona["target_task"]
            )
            narrow_results.append(r)

    print_results(broad_results, narrow_results)

    # Save analysis
    analysis = {
        "broad": broad_results,
        "narrow": narrow_results,
        "n_conditions": len(results) - 1,
    }
    analysis_path = args.output_dir / "v2_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis to {analysis_path}")


if __name__ == "__main__":
    main()
