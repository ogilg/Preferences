"""Phase 2b: Topic sanity check — verify preference shifts by persona.

For each persona, compute mean Thurstonian μ by topic and compare to baseline (persona 1).
Verifies that personas shift preferences in expected directions.

Outputs a markdown table and saves to:
  experiments/probe_generalization/multi_role_ablation/topic_sanity_check.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.probes.data_loading import load_thurstonian_scores

load_dotenv()

REPO = Path(__file__).parent.parent.parent
EXPERIMENT_DIR = REPO / "experiments/probe_generalization/multi_role_ablation"
TOPICS_PATH = REPO / "data/topics/gemma3_500_completion_preference/topics_v2.json"

PERSONAS = {
    1: {
        "name": "no_prompt",
        "run_dir": REPO / "results/experiments/mra_persona1_noprompt/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0",
    },
    2: {
        "name": "villain",
        "run_dir": REPO / "results/experiments/mra_persona2_villain/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0",
    },
    3: {
        "name": "midwest",
        "run_dir": REPO / "results/experiments/mra_persona3_midwest/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0",
    },
    4: {
        "name": "aesthete",
        "run_dir": REPO / "results/experiments/mra_persona4_aesthete/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0",
    },
}

EXPECTED_SHIFTS = {
    2: {  # Villain: higher for harmful/edgy, lower for creative/wholesome
        "higher": ["harmful_request", "model_manipulation", "security_legal"],
        "lower": ["fiction", "content_generation", "persuasive_writing"],
    },
    3: {  # Midwest: higher for practical/factual, lower for abstract/creative
        "higher": ["knowledge_qa", "math"],
        "lower": ["fiction", "persuasive_writing", "sensitive_creative"],
    },
    4: {  # Aesthete: higher for creative/literary, lower for math/coding/factual
        "higher": ["fiction", "content_generation", "sensitive_creative"],
        "lower": ["math", "knowledge_qa"],
    },
}


def load_scores(persona_id: int) -> dict[str, float]:
    run_dir = PERSONAS[persona_id]["run_dir"]
    return load_thurstonian_scores(run_dir)


def load_topics() -> dict[str, str]:
    """Load topic classifications: task_id -> primary topic."""
    with open(TOPICS_PATH) as f:
        data = json.load(f)
    topics = {}
    for task_id, classifications in data.items():
        # Take the first model's classification
        first_model = next(iter(classifications.values()))
        topics[task_id] = first_model["primary"]
    return topics


def compute_topic_means(scores: dict[str, float], topics: dict[str, str]) -> dict[str, float]:
    """Compute mean score per topic for tasks that have topics."""
    by_topic: dict[str, list[float]] = defaultdict(list)
    for task_id, score in scores.items():
        topic = topics.get(task_id)
        if topic:
            by_topic[topic].append(score)
    return {topic: float(np.mean(scores)) for topic, scores in by_topic.items()}


def main() -> None:
    print("Loading topics...")
    topics = load_topics()
    print(f"  {len(topics)} tasks with topic classifications")

    print("\nLoading persona scores...")
    scores_by_persona: dict[int, dict[str, float]] = {}
    for p_id in PERSONAS:
        try:
            scores_by_persona[p_id] = load_scores(p_id)
            print(f"  Persona {p_id} ({PERSONAS[p_id]['name']}): {len(scores_by_persona[p_id])} tasks")
        except Exception as e:
            print(f"  Persona {p_id}: ERROR - {e}")
            return

    # Compute topic means per persona
    print("\nComputing topic means...")
    topic_means: dict[int, dict[str, float]] = {}
    for p_id in PERSONAS:
        topic_means[p_id] = compute_topic_means(scores_by_persona[p_id], topics)

    # Print table of topic × persona
    all_topics = sorted(set(t for tm in topic_means.values() for t in tm))
    baseline_means = topic_means[1]

    print(f"\n{'Topic':<25} | {'baseline':>10} | {'villain':>10} | {'midwest':>10} | {'aesthete':>10} | {'Δvil':>6} | {'Δmid':>6} | {'Δaes':>6}")
    print("-" * 100)
    for topic in all_topics:
        base = baseline_means.get(topic, None)
        row = f"{topic:<25} | "
        if base is not None:
            row += f"{base:>10.3f} | "
        else:
            row += f"{'N/A':>10} | "
        for p_id in [2, 3, 4]:
            mean = topic_means[p_id].get(topic, None)
            if mean is not None:
                row += f"{mean:>10.3f} | "
            else:
                row += f"{'N/A':>10} | "
        for p_id in [2, 3, 4]:
            mean = topic_means[p_id].get(topic, None)
            if base is not None and mean is not None:
                delta = mean - base
                row += f"{delta:>+6.3f} | "
            else:
                row += f"{'N/A':>6} | "
        print(row)

    # Check expected shifts
    print("\n=== PERSONA VALIDATION ===")
    persona_validation = {}
    for p_id in [2, 3, 4]:
        expected = EXPECTED_SHIFTS[p_id]
        p_means = topic_means[p_id]
        p_name = PERSONAS[p_id]["name"]

        hits_higher = 0
        hits_lower = 0
        results_higher = []
        results_lower = []

        for topic in expected["higher"]:
            base = baseline_means.get(topic)
            persona_val = p_means.get(topic)
            if base is not None and persona_val is not None:
                delta = persona_val - base
                hit = delta > 0.05
                hits_higher += int(hit)
                results_higher.append((topic, delta, hit))

        for topic in expected["lower"]:
            base = baseline_means.get(topic)
            persona_val = p_means.get(topic)
            if base is not None and persona_val is not None:
                delta = persona_val - base
                hit = delta < -0.05
                hits_lower += int(hit)
                results_lower.append((topic, delta, hit))

        total_hits = hits_higher + hits_lower
        total_expected = len(results_higher) + len(results_lower)
        passes = total_hits >= total_expected // 2

        print(f"\nPersona {p_id} ({p_name}): {total_hits}/{total_expected} expected shifts ['{'PASS' if passes else 'FAIL'}']")
        print(f"  Expected higher: {', '.join(f'{t}({d:+.3f})' for t, d, h in results_higher)}")
        print(f"  Expected lower:  {', '.join(f'{t}({d:+.3f})' for t, d, h in results_lower)}")

        persona_validation[p_name] = {
            "hits_higher": hits_higher,
            "hits_lower": hits_lower,
            "total_hits": total_hits,
            "total_expected": total_expected,
            "passes": passes,
            "details_higher": [(t, float(d), bool(h)) for t, d, h in results_higher],
            "details_lower": [(t, float(d), bool(h)) for t, d, h in results_lower],
        }

    # Save results
    output = {
        "topic_means": {
            PERSONAS[p_id]["name"]: {t: float(v) for t, v in topic_means[p_id].items()}
            for p_id in PERSONAS
        },
        "topic_deltas": {
            PERSONAS[p_id]["name"]: {
                t: float(topic_means[p_id].get(t, 0) - baseline_means.get(t, 0))
                for t in all_topics
                if t in topic_means[p_id] and t in baseline_means
            }
            for p_id in [2, 3, 4]
        },
        "validation": persona_validation,
    }

    out_path = EXPERIMENT_DIR / "topic_sanity_check.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
