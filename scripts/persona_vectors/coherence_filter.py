"""Score coherence of all persona steering generations and flag incoherent coefficient/vector pairs.

A coefficient/vector pair is excluded if <90% of its samples are coherent (score >= 0.7).
Preference responses use a separate judge that accounts for truncated A/B choice format.
"""

import asyncio
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.measurement.elicitation.semantic_valence_scorer import (
    score_coherence_async,
    score_preference_coherence_async,
)

RESULTS_DIR = Path("results/experiments/persona_vectors_v2")
PERSONAS = ["creative_artist", "evil", "lazy", "stem_nerd", "uncensored"]
COHERENCE_THRESHOLD = 0.7
PASS_RATE = 0.90
CONCURRENCY = 50


async def score_batch(records: list[dict], scorer) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0

    async def score_one(rec: dict) -> dict:
        nonlocal done
        async with sem:
            score = await scorer(rec.get("response", ""))
            done += 1
            if done % 100 == 0:
                print(f"  Scored {done}/{len(records)}")
            return {**rec, "coherence_score": score}

    return await asyncio.gather(*[score_one(r) for r in records])


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(records: list[dict], path: Path):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    # Phase 4b: steering generations
    all_steering = []
    for persona in PERSONAS:
        path = RESULTS_DIR / persona / "steering" / "generations_checkpoint.jsonl"
        if path.exists():
            all_steering.extend(load_jsonl(path))

    # Phase 4c: preference steering
    all_preference = []
    pref_dir = RESULTS_DIR / "preference_steering"
    if pref_dir.exists():
        for persona in PERSONAS:
            path = pref_dir / f"{persona}_checkpoint.jsonl"
            if path.exists():
                all_preference.extend(load_jsonl(path))

    # Score steering (cached)
    steering_cache = RESULTS_DIR / "coherence_scores_steering.jsonl"
    if steering_cache.exists():
        print(f"Loading cached steering scores from {steering_cache}")
        steering_scored = load_jsonl(steering_cache)
    else:
        print(f"Scoring {len(all_steering)} steering records")
        steering_scored = asyncio.run(score_batch(all_steering, score_coherence_async))
        save_jsonl(steering_scored, steering_cache)

    # Score preference (separate judge, separate cache)
    pref_cache = RESULTS_DIR / "coherence_scores_preference.jsonl"
    if pref_cache.exists():
        print(f"Loading cached preference scores from {pref_cache}")
        preference_scored = load_jsonl(pref_cache)
    else:
        print(f"Scoring {len(all_preference)} preference records (preference-aware judge)")
        preference_scored = asyncio.run(score_batch(all_preference, score_preference_coherence_async))
        save_jsonl(preference_scored, pref_cache)

    # === Phase 4b analysis ===
    print("\n=== Phase 4b: Steering coherence by (persona, multiplier) ===")
    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for rec in steering_scored:
        groups[(rec["persona"], rec["multiplier"])].append(rec["coherence_score"])

    excluded_pairs = []
    for (persona, mult), scores in sorted(groups.items()):
        coherent = sum(1 for s in scores if s >= COHERENCE_THRESHOLD)
        rate = coherent / len(scores)
        status = "PASS" if rate >= PASS_RATE else "FAIL"
        if status == "FAIL":
            excluded_pairs.append((persona, mult))
        print(f"  {persona:20s} mult={mult:.2f}  coherent={coherent}/{len(scores)}  rate={rate:.0%}  {status}")

    print(f"\nExcluded pairs (coherence rate < {PASS_RATE:.0%}): {len(excluded_pairs)}")
    for persona, mult in excluded_pairs:
        print(f"  {persona} @ {mult:.2f}x")

    # === Phase 4c analysis ===
    print("\n=== Phase 4c: Preference steering coherence by (persona, condition) ===")
    pref_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rec in preference_scored:
        pref_groups[(rec["persona"], rec["condition"])].append(rec["coherence_score"])

    excluded_pref = []
    for (persona, condition), scores in sorted(pref_groups.items()):
        coherent = sum(1 for s in scores if s >= COHERENCE_THRESHOLD)
        rate = coherent / len(scores)
        status = "PASS" if rate >= PASS_RATE else "FAIL"
        if status == "FAIL":
            excluded_pref.append((persona, condition))
        print(f"  {persona:20s} {condition:10s}  coherent={coherent}/{len(scores)}  rate={rate:.0%}  {status}")

    print(f"\nExcluded preference pairs: {len(excluded_pref)}")
    for persona, condition in excluded_pref:
        print(f"  {persona} @ {condition}")

    # Save summary
    summary = {
        "coherence_threshold": COHERENCE_THRESHOLD,
        "pass_rate": PASS_RATE,
        "steering_excluded": [{"persona": p, "multiplier": m} for p, m in excluded_pairs],
        "preference_excluded": [{"persona": p, "condition": c} for p, c in excluded_pref],
    }
    save_jsonl([summary], RESULTS_DIR / "coherence_summary.json")
    print(f"\nSaved summary to {RESULTS_DIR / 'coherence_summary.json'}")


if __name__ == "__main__":
    main()
