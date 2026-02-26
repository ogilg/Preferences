"""Re-pick triage coefficients accounting for coherence.

For each persona, find the (selector, layer, multiplier) with the highest
mean trait score, subject to >=90% coherence rate (score >= 0.7).
Uses both screen and fine triage data.
"""

import asyncio
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.measurement.elicitation.semantic_valence_scorer import score_coherence_async

RESULTS_DIR = Path("results/experiments/persona_vectors_v2")
PERSONAS = ["creative_artist", "evil", "lazy", "stem_nerd", "uncensored"]
COHERENCE_THRESHOLD = 0.7
PASS_RATE = 0.90
CONCURRENCY = 50


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


async def score_batch(records: list[dict]) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0

    async def score_one(rec: dict) -> dict:
        nonlocal done
        async with sem:
            score = await score_coherence_async(rec["response"])
            done += 1
            if done % 100 == 0:
                print(f"  Scored {done}/{len(records)}")
            return {**rec, "coherence_score": score}

    return await asyncio.gather(*[score_one(r) for r in records])


def main():
    # Load all triage data (screen + fine results with trait scores)
    all_records = []
    for persona in PERSONAS:
        triage_dir = RESULTS_DIR / persona / "triage"
        for fname in ["screen_results.json", "fine_results.json"]:
            path = triage_dir / fname
            if path.exists():
                with open(path) as f:
                    all_records.extend(json.load(f))

    print(f"Total triage records: {len(all_records)}")

    # Build response -> coherence_score lookup from cached JSONL scores
    cache_path = RESULTS_DIR / "coherence_scores_triage.jsonl"
    response_to_coherence: dict[str, float] = {}
    if cache_path.exists():
        for rec in load_jsonl(cache_path):
            response_to_coherence[rec["response"]] = rec["coherence_score"]

    # Find records that need scoring
    unscored = [r for r in all_records if r["response"] not in response_to_coherence]
    if unscored:
        print(f"Scoring {len(unscored)} new records (have {len(response_to_coherence)} cached)...")
        newly_scored = asyncio.run(score_batch(unscored))
        for rec in newly_scored:
            response_to_coherence[rec["response"]] = rec["coherence_score"]
        # Update cache
        with open(cache_path, "a") as f:
            for rec in newly_scored:
                f.write(json.dumps(rec) + "\n")
    else:
        print(f"All {len(all_records)} records have cached coherence scores")

    # Attach coherence scores to all records
    scored = [{**r, "coherence_score": response_to_coherence[r["response"]]} for r in all_records]

    # Group by (persona, selector, layer, multiplier)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in scored:
        key = (rec["persona"], rec["selector"], rec["layer"], rec["multiplier"])
        groups[key].append(rec)

    # For each persona, find best coherent combo
    print("\n=== Re-triage with coherence constraint ===\n")
    for persona in PERSONAS:
        print(f"--- {persona} ---")
        persona_groups = {k: v for k, v in groups.items() if k[0] == persona}

        # Show all combos
        candidates = []
        for (_, selector, layer, mult), recs in sorted(persona_groups.items()):
            n = len(recs)
            coherent = sum(1 for r in recs if r["coherence_score"] >= COHERENCE_THRESHOLD)
            coherence_rate = coherent / n
            trait_scores = [r["trait_score"] for r in recs]
            mean_trait = sum(trait_scores) / len(trait_scores)
            passes = coherence_rate >= PASS_RATE

            candidates.append({
                "selector": selector,
                "layer": layer,
                "multiplier": mult,
                "mean_trait": mean_trait,
                "coherence_rate": coherence_rate,
                "n": n,
                "passes": passes,
            })

        # Sort by coherence pass then trait score
        candidates.sort(key=lambda c: (c["passes"], c["mean_trait"]), reverse=True)

        for c in candidates:
            flag = "PASS" if c["passes"] else "FAIL"
            marker = " <-- BEST" if c == candidates[0] and c["passes"] else ""
            print(
                f"  {c['selector']:12s} L{c['layer']:<3d} mult={c['multiplier']:.2f}  "
                f"trait={c['mean_trait']:.1f}  coherent={c['coherence_rate']:.0%} ({c['n']}){marker}  [{flag}]"
            )

        # Report best
        best = candidates[0] if candidates[0]["passes"] else None
        if best:
            print(f"  >> Selected: {best['selector']} L{best['layer']} @ {best['multiplier']:.2f}x (trait={best['mean_trait']:.1f}, coherence={best['coherence_rate']:.0%})")
        else:
            print("  >> NO coherent combo found!")

        # Load and compare with original selection
        orig_path = RESULTS_DIR / persona / "triage" / "selected.json"
        if orig_path.exists():
            with open(orig_path) as f:
                orig = json.load(f)
            print(f"  >> Original: {orig.get('selector', '?')} L{orig.get('layer', '?')} @ {orig.get('multiplier', '?')}x")
        print()


if __name__ == "__main__":
    main()
