"""Step 1: Behavioural sanity checks for paraphrases.

Check A — Direct comparison: Pairwise preferences between each paraphrase and its original.
Check B — Relative ranking: Compare win rates of original vs paraphrase against 5 opponents.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.stats import spearmanr

from src.measurement.elicitation.measure import measure_pre_task_revealed_async
from src.measurement.elicitation.measurer import RevealedPreferenceMeasurer
from src.measurement.elicitation.prompt_templates.builders import PreTaskRevealedPromptBuilder
from src.measurement.elicitation.prompt_templates.template import PromptTemplate
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.models.openai_compatible import OpenRouterClient
from src.task_data import Task, OriginDataset

load_dotenv()

PARAPHRASES_FILE = Path("experiments/probe_science/paraphrase_augmentation/paraphrases.json")
OUTPUT_DIR = Path("experiments/probe_science/paraphrase_augmentation")
RESULTS_FILE = OUTPUT_DIR / "sanity_check_results.json"

TEMPLATE_TEXT = (
    "You will be given two tasks. Choose one and complete it.\n"
    "{format_instruction}\n\n"
    "Task A:\n{task_a}\n\n"
    "Task B:\n{task_b}"
)

N_REPEATS_CHECK_A = 5
N_OPPONENTS = 5
SEED = 42
MAX_CONCURRENT = 30


def make_task(task_id: str, prompt: str, origin: str) -> Task:
    return Task(
        id=task_id,
        prompt=prompt,
        origin=OriginDataset[origin],
        metadata={},
    )


def make_builder() -> PreTaskRevealedPromptBuilder:
    template = PromptTemplate(
        name="paraphrase_check",
        template=TEMPLATE_TEXT,
        required_placeholders=frozenset({"task_a", "task_b", "format_instruction"}),
    )
    response_format = CompletionChoiceFormat(
        task_a_label="Task A",
        task_b_label="Task B",
    )
    return PreTaskRevealedPromptBuilder(
        measurer=RevealedPreferenceMeasurer(),
        response_format=response_format,
        template=template,
    )


async def run_check_a(data: list[dict], client: OpenRouterClient, builder: PreTaskRevealedPromptBuilder) -> dict:
    """Check A: Direct comparison between original and paraphrase."""
    print("\n=== Check A: Direct Comparison ===")

    pairs = []
    # Map (task_a_id, task_b_id) -> metadata for matching (unique IDs per repeat)
    pair_meta_lookup: dict[tuple[str, str], dict] = {}
    for entry in data:
        for rep in range(N_REPEATS_CHECK_A):
            orig_id = f"{entry['task_id']}_r{rep}"
            para_id = f"{entry['task_id']}_para_r{rep}"
            orig_task = make_task(orig_id, entry["original_prompt"], entry["origin"])
            para_task = make_task(para_id, entry["paraphrased_prompt"], entry["origin"])
            if rep % 2 == 0:
                pairs.append((orig_task, para_task))
                pair_meta_lookup[(orig_id, para_id)] = {"task_id": entry["task_id"], "order": "orig_first"}
            else:
                pairs.append((para_task, orig_task))
                pair_meta_lookup[(para_id, orig_id)] = {"task_id": entry["task_id"], "order": "para_first"}

    print(f"Running {len(pairs)} comparisons ({len(data)} pairs × {N_REPEATS_CHECK_A} repeats)...")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    batch = await measure_pre_task_revealed_async(
        client=client,
        pairs=pairs,
        builder=builder,
        semaphore=semaphore,
        temperature=0.7,
    )

    print(f"Successes: {len(batch.successes)}, Failures: {len(batch.failures)}")

    # Compute per-task original win rates using task IDs for matching
    win_counts: dict[str, dict] = {}
    for measurement in batch.successes:
        key = (measurement.task_a.id, measurement.task_b.id)
        meta = pair_meta_lookup[key]
        tid = meta["task_id"]
        if tid not in win_counts:
            win_counts[tid] = {"orig_wins": 0, "para_wins": 0, "refusals": 0, "total": 0}

        if measurement.choice == "refusal":
            win_counts[tid]["refusals"] += 1
        elif meta["order"] == "orig_first":
            if measurement.choice == "a":
                win_counts[tid]["orig_wins"] += 1
            else:
                win_counts[tid]["para_wins"] += 1
        else:  # para_first
            if measurement.choice == "a":
                win_counts[tid]["para_wins"] += 1
            else:
                win_counts[tid]["orig_wins"] += 1
        win_counts[tid]["total"] += 1

    # Compute win rates
    win_rates = []
    flagged = []
    for tid, counts in win_counts.items():
        valid = counts["orig_wins"] + counts["para_wins"]
        if valid > 0:
            orig_rate = counts["orig_wins"] / valid
            win_rates.append(orig_rate)
            if orig_rate > 0.8 or orig_rate < 0.2:
                flagged.append({"task_id": tid, "orig_win_rate": orig_rate, **counts})

    mean_orig_rate = np.mean(win_rates) if win_rates else 0.0
    print(f"\nOriginal win rate: mean={mean_orig_rate:.3f} (expected ~0.5)")
    print(f"Flagged pairs (win rate >0.8 or <0.2): {len(flagged)}/{len(win_counts)}")
    for f in flagged[:5]:
        print(f"  {f['task_id']}: orig_win_rate={f['orig_win_rate']:.2f}")

    return {
        "mean_orig_win_rate": float(mean_orig_rate),
        "n_pairs": len(win_counts),
        "n_flagged": len(flagged),
        "flagged": flagged,
        "win_rates": {tid: counts for tid, counts in win_counts.items()},
        "n_successes": len(batch.successes),
        "n_failures": len(batch.failures),
    }


async def run_check_b(data: list[dict], client: OpenRouterClient, builder: PreTaskRevealedPromptBuilder) -> dict:
    """Check B: Relative ranking — compare win rates against shared opponents."""
    print("\n=== Check B: Relative Ranking ===")

    rng = np.random.default_rng(SEED)

    pairs = []
    # Map (task_a_id, task_b_id) -> metadata for matching
    pair_meta_lookup: dict[tuple[str, str], dict] = {}

    for i, entry in enumerate(data):
        other_indices = [j for j in range(len(data)) if j != i]
        other_utils = np.array([data[j]["utility"] for j in other_indices])
        strata_edges = np.percentile(other_utils, [0, 20, 40, 60, 80, 100])
        opponents = []
        for s in range(5):
            lo, hi = strata_edges[s], strata_edges[s + 1]
            candidates = [j for j in other_indices if lo <= data[j]["utility"] <= hi]
            if candidates:
                opponents.append(rng.choice(candidates))

        if len(opponents) < 5:
            opponents = list(rng.choice(other_indices, size=5, replace=False))

        orig_task = make_task(entry["task_id"], entry["original_prompt"], entry["origin"])
        para_task = make_task(f"{entry['task_id']}_para", entry["paraphrased_prompt"], entry["origin"])

        for opp_idx in opponents:
            opp = data[opp_idx]
            # Use unique IDs for opponent tasks per comparison to avoid key collisions
            opp_id_orig = f"{opp['task_id']}_opp_for_{entry['task_id']}_orig"
            opp_id_para = f"{opp['task_id']}_opp_for_{entry['task_id']}_para"
            opp_task_orig = make_task(opp_id_orig, opp["original_prompt"], opp["origin"])
            opp_task_para = make_task(opp_id_para, opp["original_prompt"], opp["origin"])

            pairs.append((orig_task, opp_task_orig))
            pair_meta_lookup[(orig_task.id, opp_id_orig)] = {
                "task_id": entry["task_id"],
                "opponent_id": opp["task_id"],
                "variant": "original",
            }
            pairs.append((para_task, opp_task_para))
            pair_meta_lookup[(para_task.id, opp_id_para)] = {
                "task_id": entry["task_id"],
                "opponent_id": opp["task_id"],
                "variant": "paraphrase",
            }

    print(f"Running {len(pairs)} comparisons ({len(data)} tasks × {N_OPPONENTS} opponents × 2 variants)...")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    batch = await measure_pre_task_revealed_async(
        client=client,
        pairs=pairs,
        builder=builder,
        semaphore=semaphore,
        temperature=0.7,
    )

    print(f"Successes: {len(batch.successes)}, Failures: {len(batch.failures)}")

    # Compute per-task, per-opponent win rates using task IDs for matching
    results_by_task: dict[str, dict] = {}
    for measurement in batch.successes:
        key = (measurement.task_a.id, measurement.task_b.id)
        meta = pair_meta_lookup[key]
        tid = meta["task_id"]
        oid = meta["opponent_id"]
        variant = meta["variant"]

        if tid not in results_by_task:
            results_by_task[tid] = {"orig": {}, "para": {}}

        if measurement.choice == "refusal":
            win = 0.5
        elif measurement.choice == "a":
            win = 1.0
        else:
            win = 0.0

        key_name = "orig" if variant == "original" else "para"
        results_by_task[tid][key_name][oid] = win

    # Compute rank correlations
    correlations = []
    for tid, res in results_by_task.items():
        common_opps = sorted(set(res["orig"].keys()) & set(res["para"].keys()))
        if len(common_opps) < 3:
            continue
        orig_wins = [res["orig"][o] for o in common_opps]
        para_wins = [res["para"][o] for o in common_opps]
        if len(set(orig_wins)) > 1 and len(set(para_wins)) > 1:
            r, p = spearmanr(orig_wins, para_wins)
            correlations.append({"task_id": tid, "rho": float(r), "p": float(p), "n_opponents": len(common_opps)})
        else:
            correlations.append({"task_id": tid, "rho": 1.0, "p": 0.0, "n_opponents": len(common_opps)})

    rhos = [c["rho"] for c in correlations]
    median_rho = float(np.median(rhos)) if rhos else 0.0
    mean_rho = float(np.mean(rhos)) if rhos else 0.0

    # Overall agreement rate
    agree_count = 0
    total_count = 0
    for tid, res in results_by_task.items():
        common_opps = set(res["orig"].keys()) & set(res["para"].keys())
        for o in common_opps:
            if res["orig"][o] == res["para"][o]:
                agree_count += 1
            total_count += 1

    agreement_rate = agree_count / total_count if total_count > 0 else 0.0

    print(f"\nRank correlations: median={median_rho:.3f}, mean={mean_rho:.3f}")
    print(f"Agreement rate (same winner): {agreement_rate:.3f} ({agree_count}/{total_count})")
    print(f"Tasks with computed correlations: {len(correlations)}")

    return {
        "median_rho": median_rho,
        "mean_rho": mean_rho,
        "agreement_rate": float(agreement_rate),
        "n_tasks_with_correlations": len(correlations),
        "correlations": correlations,
        "n_successes": len(batch.successes),
        "n_failures": len(batch.failures),
    }


async def main():
    with open(PARAPHRASES_FILE) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} paraphrases")

    client = OpenRouterClient(model_name="gemma-3-27b", max_new_tokens=2048)
    builder = make_builder()

    # Run pilot first (5 tasks)
    print("\n--- PILOT (5 tasks) ---")
    pilot_data = data[:5]
    pilot_a = await run_check_a(pilot_data, client, builder)
    print(f"Pilot Check A: mean win rate = {pilot_a['mean_orig_win_rate']:.3f}")
    if pilot_a["n_failures"] > pilot_a["n_successes"]:
        print("WARNING: More failures than successes in pilot. Investigating...")
        return

    # Full run
    print("\n--- FULL RUN ---")
    check_a_results = await run_check_a(data, client, builder)
    check_b_results = await run_check_b(data, client, builder)

    # Save results
    results = {
        "check_a": check_a_results,
        "check_b": check_b_results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_FILE}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Check A: mean orig win rate = {check_a_results['mean_orig_win_rate']:.3f} (target: 0.35-0.65)")
    ca_pass = 0.35 <= check_a_results["mean_orig_win_rate"] <= 0.65
    print(f"  PASS: {ca_pass}")
    print(f"Check B: median rho = {check_b_results['median_rho']:.3f} (target: >0.8)")
    cb_pass = check_b_results["median_rho"] > 0.8
    print(f"  PASS: {cb_pass}")


if __name__ == "__main__":
    asyncio.run(main())
