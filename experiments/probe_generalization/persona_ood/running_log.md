# Persona OOD v2: Running Log

## Setup

- Fixed `PreTaskRevealedPromptBuilder` to accept `system_prompt` parameter
- Updated `measure_persona.py` baseline to use `"You are a helpful assistant."` instead of `None`
- Created `v2_config.json` with 101 core tasks, 10 anchors, 20 personas (10 broad + 10 narrow)
- n_resamples=2, max_concurrent=100, temperature=0.7
- Expected: 101 × 10 × 2 = 2,020 API calls per condition, 21 conditions = ~42K total

## Smoke test (baseline + stem_enthusiast)

- Baseline: 1627/2020 successes (80.5%), 393 failures, 6 refusals, 143s
- Stem_enthusiast: 1286/2020 successes (63.7%), 734 failures, 1 refusal, 138s
- Higher failure rate with persona prompt (longer responses hitting limits? parse errors?)
- But coverage is sufficient: 91/101 tasks have >= 5 comparisons in both conditions

Category-level deltas for stem_enthusiast (expected: +math, +coding; -fiction, -content_gen):
- math: +0.266, coding: +0.230, fiction: -0.311, content_gen: -0.109
- Matches predictions perfectly. Strong signal.

## Full run — blocked by API key

OpenRouter API key stopped working (401 "User not found") after completing baseline + stem_enthusiast.
4 conditions attempted with dead key (creative_writer, philosopher, trivia_nerd, hacker) — all had ~98% failure rate. Removed those results.

Attempted fixes:
- Verified key exists in .env
- Tested direct OpenAI client call — same error
- Not a rate limit (single call also fails)
- No alternative provider has gemma-3-27b (not on Hyperbolic or Cerebras)

Proceeding with analysis of the 2 valid conditions to validate the full pipeline.

API recovered at 22:28. Resumed measurement.

## Full run — completed

All 21 conditions completed successfully. Success rates per condition:
- baseline: 1627/2020 (81%)
- stem_enthusiast: 1286/2020 (64%) — first run during API instability
- creative_writer: 2020/2020 (100%) — ran after API recovered
- All subsequent conditions: 1900-2020 successes (95-100%)
- witch_trials_scholar: 1924 (slower, 790s vs ~200s avg, likely rate limit blip)

## Analysis results

**Part A (broad): 9/10 pass** (need 5/10)
- Perfect: stem_enthusiast (4/4), hacker (4/4)
- Strong: creative_writer (3/5), edgelord (3/4), pragmatist (3/5)
- Moderate: trivia_nerd (2/3), storyteller (2/4)
- Weak: philosopher (1/2), debate_champion (1/2)
- Fail: safety_advocate (0/3) — all expected negative shifts didn't materialize

**Part B (narrow): 3/10 pass strict criteria** (need 5/10)
- PASS: organ_enthusiast (rank 1, spec 4.13), chess_programming_lover (rank 1, spec 3.62), sql_devotee (rank 3, spec 3.10)
- Near-miss: doctor_who_fan (rank 5, spec 2.82), witch_trials_scholar (rank 6, spec 2.68), wildlife_conservation (rank 10, spec 2.64)
- Fail: dune_lore_master (rank 100, delta=0.0), spongebob/polynomial/horror (low specificity)

