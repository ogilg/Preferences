"""Select qualifying personas and targeted prompts, split 75/25 iteration/holdout.

Usage:
    python -m scripts.persona_ood.select_and_save
"""

import json
from pathlib import Path

import numpy as np

PART_A_PATH = Path("experiments/probe_generalization/persona_ood/full_results_part_a.json")
PART_B_PATH = Path("experiments/probe_generalization/persona_ood/full_results_part_b.json")
TARGETED_PROMPTS_PATH = Path("experiments/probe_generalization/persona_ood/targeted_prompts.json")
ALL_PERSONAS_PATH = Path("experiments/probe_generalization/persona_ood/all_personas.json")

OUTPUT_PERSONA = Path("results/persona_behavioral.json")
OUTPUT_TARGETED = Path("results/targeted_behavioral.json")

RNG = np.random.RandomState(42)


def compute_deltas(results, baseline_key="baseline"):
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


def select_personas():
    with open(PART_A_PATH) as f:
        results = json.load(f)
    with open(ALL_PERSONAS_PATH) as f:
        all_personas = json.load(f)

    persona_lookup = {p["name"]: p for p in all_personas}
    deltas = compute_deltas(results)

    # Compute stats per persona
    persona_stats = []
    for name, task_deltas in deltas.items():
        abs_deltas = np.abs(list(task_deltas.values()))
        persona_stats.append({
            "name": name,
            "system_prompt": persona_lookup[name]["system_prompt"],
            "safety_relevant": persona_lookup[name].get("safety_relevant", False),
            "mean_abs_delta": float(np.mean(abs_deltas)),
            "max_abs_delta": float(np.max(abs_deltas)),
            "n_shifted_01": int(np.sum(abs_deltas > 0.1)),
            "n_tasks": len(task_deltas),
            "task_deltas": {tid: float(d) for tid, d in task_deltas.items()},
        })

    # Select qualifying: mean |delta| > 0.1
    qualifying = [s for s in persona_stats if s["mean_abs_delta"] > 0.1]
    qualifying.sort(key=lambda s: -s["mean_abs_delta"])

    print(f"Part A: {len(qualifying)}/{len(persona_stats)} personas qualify (mean |Δ| > 0.1)")

    if len(qualifying) < 10:
        print(f"WARNING: Only {len(qualifying)} qualifying — relaxing threshold to mean |Δ| > 0.08")
        qualifying = [s for s in persona_stats if s["mean_abs_delta"] > 0.08]
        qualifying.sort(key=lambda s: -s["mean_abs_delta"])
        print(f"  Relaxed: {len(qualifying)} qualify")

    # Split 75/25
    n_iter = int(len(qualifying) * 0.75)
    indices = RNG.permutation(len(qualifying))
    iteration = [qualifying[i] for i in indices[:n_iter]]
    holdout = [qualifying[i] for i in indices[n_iter:]]

    # Add baseline rates
    baseline_rates = {
        tid: float(data["p_choose"])
        for tid, data in results["baseline"]["task_rates"].items()
        if not np.isnan(data["p_choose"])
    }

    output = {
        "baseline_rates": baseline_rates,
        "iteration_set": [
            {k: v for k, v in s.items()} for s in iteration
        ],
        "holdout_set": [
            {k: v for k, v in s.items()} for s in holdout
        ],
        "all_personas": [
            {k: v for k, v in s.items()} for s in persona_stats
        ],
        "selection_criteria": "mean_abs_delta > 0.1",
        "n_qualifying": len(qualifying),
        "n_iteration": len(iteration),
        "n_holdout": len(holdout),
    }

    OUTPUT_PERSONA.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PERSONA, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(iteration)} iteration + {len(holdout)} holdout personas to {OUTPUT_PERSONA}")


def select_targeted():
    with open(PART_B_PATH) as f:
        results = json.load(f)
    with open(TARGETED_PROMPTS_PATH) as f:
        targeted = json.load(f)

    target_map = {p["name"]: p for p in targeted}
    deltas = compute_deltas(results)

    prompt_stats = []
    for name, task_deltas in deltas.items():
        info = target_map.get(name)
        if info is None:
            continue

        target_id = info["target_task_id"]
        on_target_delta = task_deltas.get(target_id, float("nan"))
        off_target_deltas = [abs(d) for tid, d in task_deltas.items() if tid != target_id]
        mean_off_target = float(np.mean(off_target_deltas)) if off_target_deltas else float("nan")
        specificity = abs(on_target_delta) / mean_off_target if mean_off_target > 0 else float("nan")

        all_abs = sorted([(abs(d), tid) for tid, d in task_deltas.items()], reverse=True)
        rank = next((i + 1 for i, (_, tid) in enumerate(all_abs) if tid == target_id), len(all_abs))

        prompt_stats.append({
            "name": name,
            "target_task_id": target_id,
            "task_summary": info["task_summary"],
            "system_prompt": info["system_prompt"],
            "on_target_delta": float(on_target_delta),
            "abs_on_target": float(abs(on_target_delta)),
            "mean_off_target": mean_off_target,
            "specificity_ratio": float(specificity),
            "rank": rank,
            "task_deltas": {tid: float(d) for tid, d in task_deltas.items()},
        })

    # Qualifying: |on-target delta| > 0.1 AND specificity > 3 AND rank <= 3
    qualifying = [
        s for s in prompt_stats
        if s["abs_on_target"] > 0.1 and s["specificity_ratio"] > 3 and s["rank"] <= 3
    ]
    qualifying.sort(key=lambda s: -s["specificity_ratio"])

    print(f"\nPart B: {len(qualifying)}/{len(prompt_stats)} targeted prompts qualify (strict criteria)")

    if len(qualifying) < 5:
        print(f"WARNING: Only {len(qualifying)} qualifying — also showing relaxed criteria")
        relaxed = [s for s in prompt_stats if s["abs_on_target"] > 0.1 and s["specificity_ratio"] > 2]
        print(f"  Relaxed (spec>2): {len(relaxed)}")

    # Split 75/25
    n_iter = max(1, int(len(qualifying) * 0.75))
    indices = RNG.permutation(len(qualifying))
    iteration = [qualifying[i] for i in indices[:n_iter]]
    holdout = [qualifying[i] for i in indices[n_iter:]]

    baseline_rates = {
        tid: float(data["p_choose"])
        for tid, data in results["baseline"]["task_rates"].items()
        if not np.isnan(data["p_choose"])
    }

    output = {
        "baseline_rates": baseline_rates,
        "iteration_set": iteration,
        "holdout_set": holdout,
        "all_prompts": prompt_stats,
        "selection_criteria": "abs_on_target > 0.1, specificity > 3, rank <= 3",
        "n_qualifying": len(qualifying),
        "n_iteration": len(iteration),
        "n_holdout": len(holdout),
    }

    OUTPUT_TARGETED.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TARGETED, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(iteration)} iteration + {len(holdout)} holdout targeted prompts to {OUTPUT_TARGETED}")


def main():
    if PART_A_PATH.exists():
        select_personas()
    else:
        print("Part A results not found, skipping persona selection")

    if PART_B_PATH.exists():
        select_targeted()
    else:
        print("Part B results not found, skipping targeted prompt selection")


if __name__ == "__main__":
    main()
