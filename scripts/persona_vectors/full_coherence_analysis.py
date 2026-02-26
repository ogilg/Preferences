"""Full coherence-filtered analysis of persona vectors v2.

Pipeline:
1. Load triage data + coherence scores → filter → select best coherent (layer, selector, coeff)
2. Load Phase 4b steering data + coherence scores → filter → dose-response on coherent points only
3. Load Phase 4c preference data + coherence scores → filter → preference analysis on coherent conditions
4. Print everything needed for the report
"""

import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path("results/experiments/persona_vectors_v2")
PERSONAS = ["creative_artist", "evil", "lazy", "stem_nerd", "uncensored"]
COHERENCE_THRESHOLD = 0.7
PASS_RATE = 0.90


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def coherence_rate(records: list[dict]) -> float:
    coherent = sum(1 for r in records if r["coherence_score"] >= COHERENCE_THRESHOLD)
    return coherent / len(records)


# ============================================================
# 1. TRIAGE: coherence-filtered selection
# ============================================================

print("=" * 70)
print("PHASE 4a: COHERENCE-FILTERED TRIAGE SELECTION")
print("=" * 70)

# Load triage results (with trait scores) and coherence scores
triage_by_persona: dict[str, list[dict]] = defaultdict(list)
for persona in PERSONAS:
    triage_dir = RESULTS_DIR / persona / "triage"
    for fname in ["screen_results.json", "fine_results.json"]:
        path = triage_dir / fname
        if path.exists():
            with open(path) as f:
                triage_by_persona[persona].extend(json.load(f))

# Load coherence scores (keyed by response text)
coherence_cache = {}
cache_path = RESULTS_DIR / "coherence_scores_triage.jsonl"
if cache_path.exists():
    for rec in load_jsonl(cache_path):
        coherence_cache[rec["response"]] = rec["coherence_score"]

# Attach coherence scores
for persona in PERSONAS:
    for rec in triage_by_persona[persona]:
        rec["coherence_score"] = coherence_cache[rec["response"]]

# Group and select
selections = {}
for persona in PERSONAS:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in triage_by_persona[persona]:
        key = (rec["selector"], rec["layer"], rec["multiplier"])
        groups[key].append(rec)

    print(f"\n--- {persona} ---")

    # Find all coherent combos
    coherent_combos = []
    all_combos = []
    for (selector, layer, mult), recs in sorted(groups.items()):
        n = len(recs)
        cr = coherence_rate(recs)
        mean_trait = sum(r["trait_score"] for r in recs) / n
        passes = cr >= PASS_RATE
        combo = {
            "selector": selector, "layer": layer, "multiplier": mult,
            "mean_trait": mean_trait, "coherence_rate": cr, "n": n, "passes": passes,
        }
        all_combos.append(combo)
        if passes:
            coherent_combos.append(combo)

    # Sort coherent by trait score
    coherent_combos.sort(key=lambda c: c["mean_trait"], reverse=True)

    # Show top 5 coherent
    print(f"  Top coherent combos (of {len(coherent_combos)} passing / {len(all_combos)} total):")
    for c in coherent_combos[:5]:
        print(f"    {c['selector']:12s} L{c['layer']:<3d} mult={c['multiplier']:.2f}  trait={c['mean_trait']:.1f}  coherent={c['coherence_rate']:.0%} (n={c['n']})")

    # Show top 3 that FAIL coherence (for comparison)
    failed = [c for c in all_combos if not c["passes"]]
    failed.sort(key=lambda c: c["mean_trait"], reverse=True)
    if failed:
        print(f"  Top INCOHERENT combos (would have been selected without coherence filter):")
        for c in failed[:3]:
            print(f"    {c['selector']:12s} L{c['layer']:<3d} mult={c['multiplier']:.2f}  trait={c['mean_trait']:.1f}  coherent={c['coherence_rate']:.0%} (n={c['n']})")

    if coherent_combos:
        best = coherent_combos[0]
        selections[persona] = best
        print(f"  >> SELECTED: {best['selector']} L{best['layer']} @ {best['multiplier']:.2f}x (trait={best['mean_trait']:.1f})")
    else:
        print("  >> NO COHERENT COMBO FOUND")

    # Load original selection for comparison
    orig_path = RESULTS_DIR / persona / "triage" / "selected.json"
    if orig_path.exists():
        with open(orig_path) as f:
            orig = json.load(f)
        print(f"  >> Original (no coherence): {orig.get('selector', '?')} L{orig.get('layer', '?')} @ {orig.get('multiplier', '?')}x (trait={orig.get('mean_score', '?')})")

# Summary table
print("\n\n" + "=" * 70)
print("SELECTION SUMMARY")
print("=" * 70)
print(f"{'Persona':<20s} {'Selector':<14s} {'Layer':<6s} {'Mult':<8s} {'Trait':<7s} {'Coherence'}")
for persona in PERSONAS:
    s = selections.get(persona)
    if s:
        print(f"{persona:<20s} {s['selector']:<14s} L{s['layer']:<4d} {s['multiplier']:.2f}x   {s['mean_trait']:.1f}     {s['coherence_rate']:.0%}")
    else:
        print(f"{persona:<20s} NO COHERENT COMBO")


# ============================================================
# 2. PHASE 4b: Steering dose-response (coherent points only)
# ============================================================

print("\n\n" + "=" * 70)
print("PHASE 4b: STEERING DOSE-RESPONSE (COHERENT POINTS ONLY)")
print("=" * 70)

# Load steering coherence scores
steering_cache = RESULTS_DIR / "coherence_scores_steering.jsonl"
if steering_cache.exists():
    steering_scored = load_jsonl(steering_cache)
else:
    print("WARNING: No steering coherence scores found")
    steering_scored = []

for persona in PERSONAS:
    persona_recs = [r for r in steering_scored if r["persona"] == persona]
    if not persona_recs:
        print(f"\n--- {persona}: no Phase 4b data ---")
        continue

    # Group by multiplier
    by_mult: dict[float, list[dict]] = defaultdict(list)
    for rec in persona_recs:
        by_mult[rec["multiplier"]].append(rec)

    # What layer was this evaluated at?
    layer = persona_recs[0]["layer"]

    # Does this match the coherence-constrained selection?
    sel = selections.get(persona)
    matches = sel and sel["layer"] == layer and sel["selector"] == persona_recs[0].get("selector", "prompt_last")

    print(f"\n--- {persona} (evaluated at L{layer}) {'[MATCHES selection]' if matches else '[MISMATCH - evaluated at original layer]'} ---")

    for mult in sorted(by_mult.keys()):
        recs = by_mult[mult]
        cr = coherence_rate(recs)
        passes = cr >= PASS_RATE
        # Get trait scores if available
        trait_scores = [r.get("trait_score") for r in recs if "trait_score" in r]
        mean_trait = sum(trait_scores) / len(trait_scores) if trait_scores else None
        trait_str = f"trait={mean_trait:.1f}" if mean_trait else "trait=N/A"

        status = "PASS" if passes else "FAIL"
        print(f"  mult={mult:.2f}  coherent={sum(1 for r in recs if r['coherence_score'] >= COHERENCE_THRESHOLD)}/{len(recs)}  rate={cr:.0%}  {status}  {trait_str}")


# ============================================================
# 3. PHASE 4c: Preference steering (coherent conditions only)
# ============================================================

print("\n\n" + "=" * 70)
print("PHASE 4c: PREFERENCE STEERING (COHERENCE FILTER)")
print("=" * 70)

pref_cache = RESULTS_DIR / "coherence_scores_preference.jsonl"
if pref_cache.exists():
    pref_scored = load_jsonl(pref_cache)
else:
    print("WARNING: No preference coherence scores found")
    pref_scored = []

for persona in PERSONAS:
    persona_recs = [r for r in pref_scored if r["persona"] == persona]
    if not persona_recs:
        continue

    baseline = [r for r in persona_recs if r["condition"] == "baseline"]
    steered = [r for r in persona_recs if r["condition"] == "steered"]

    base_cr = coherence_rate(baseline) if baseline else 0
    steer_cr = coherence_rate(steered) if steered else 0

    base_pass = base_cr >= PASS_RATE
    steer_pass = steer_cr >= PASS_RATE

    # Compute preference rates on COHERENT subset only
    def pref_rate(recs, threshold=COHERENCE_THRESHOLD):
        coherent = [r for r in recs if r["coherence_score"] >= threshold]
        if not coherent:
            return None, 0
        chose_pos = sum(1 for r in coherent if r.get("chose_positive", False))
        return chose_pos / len(coherent), len(coherent)

    base_rate, base_n = pref_rate(baseline)
    steer_rate, steer_n = pref_rate(steered)

    print(f"\n--- {persona} ---")
    print(f"  Baseline: coherent={base_cr:.0%} {'PASS' if base_pass else 'FAIL'}  pref_rate={base_rate:.1%} (n={base_n})" if base_rate is not None else f"  Baseline: coherent={base_cr:.0%}")
    print(f"  Steered:  coherent={steer_cr:.0%} {'PASS' if steer_pass else 'FAIL'}  pref_rate={steer_rate:.1%} (n={steer_n})" if steer_rate is not None else f"  Steered:  coherent={steer_cr:.0%}")

    if base_pass and steer_pass:
        delta = steer_rate - base_rate
        print(f"  >> VALID COMPARISON: {base_rate:.1%} → {steer_rate:.1%} (delta={delta:+.1%})")
    elif base_pass and not steer_pass:
        print(f"  >> INVALID: steered condition fails coherence ({steer_cr:.0%})")
    else:
        print(f"  >> INVALID: baseline fails coherence ({base_cr:.0%})")


# ============================================================
# 4. Transcript samples at coherent settings
# ============================================================

print("\n\n" + "=" * 70)
print("TRANSCRIPT SAMPLES (from triage data at coherent settings)")
print("=" * 70)

for persona in PERSONAS:
    sel = selections.get(persona)
    if not sel:
        print(f"\n--- {persona}: no selection ---")
        continue

    # Get records at selected (selector, layer, multiplier) from triage
    matching = [
        r for r in triage_by_persona[persona]
        if r["selector"] == sel["selector"]
        and r["layer"] == sel["layer"]
        and r["multiplier"] == sel["multiplier"]
        and r["coherence_score"] >= COHERENCE_THRESHOLD
    ]

    # Also get baseline records at same (selector, layer)
    baseline = [
        r for r in triage_by_persona[persona]
        if r["selector"] == sel["selector"]
        and r["layer"] == sel["layer"]
        and r["multiplier"] == 0.0
        and r["coherence_score"] >= COHERENCE_THRESHOLD
    ]

    print(f"\n--- {persona}: {sel['selector']} L{sel['layer']} @ {sel['multiplier']:.2f}x ---")

    if baseline:
        r = baseline[0]
        print(f"\n  BASELINE (mult=0.00, trait={r['trait_score']}):")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['response'][:400]}...")

    if matching:
        r = matching[0]
        print(f"\n  STEERED (mult={sel['multiplier']:.2f}, trait={r['trait_score']}):")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['response'][:400]}...")


# ============================================================
# 5. Save selections for downstream use
# ============================================================

out_path = RESULTS_DIR / "coherence_constrained_selections.json"
with open(out_path, "w") as f:
    json.dump(selections, f, indent=2)
print(f"\n\nSaved selections to {out_path}")
