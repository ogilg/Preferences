# Patching Pilot Running Log

## 2026-03-06: Setup
- Environment: H100 80GB, IS_SANDBOX=1
- Branch: research-loop/patching-pilot
- Data: 10 tasks at evenly spaced utility quantiles, 45 pairs, both orderings = 90 prompts
- Conditions: baseline, last-token swap, span swap
- 5 trials per ordering per condition, temperature=1.0, max_new_tokens=16

## 2026-03-06: Pilot validation (2 pairs)
- BOS offset = 1 (correctly handled)
- Span verification passed
- All responses parsed successfully (no parse_fail)
- Pair 1 (stresstest_17_576 vs stresstest_4_304): span_swap flipped AB ordering (A=5→B=5), no effect in BA
- Pair 2 (stresstest_17_576 vs alpaca_10661): no effect (large utility gap)
- Last-token swap had no effect on either pair

## 2026-03-06: Full experiment
- All 90 prompts × 3 conditions × 5 trials completed (~75 min)
- Zero parse failures
- Key observations from raw output:
  - Most baselines are unanimous (5/0 or 0/5) - very deterministic
  - Last-token swap rarely differs from baseline
  - Span swap flips some pairs (e.g., pairs 1, 6, 22, 25, 27, 63, 66, 72, 74, 81, 83, 89)
  - Many non-stresstest pairs show zero patching effect

## 2026-03-06: Analysis
- Position bias: P(A) = 0.591 baseline, 0.629 last-token, 0.473 span
- After aggregating orderings: 21/45 pairs at P(B)=0.50 (position bias), 23/45 at P(B)=1.00
- Last-token swap: 6/45 sig shifts, 1 flip (2.2%), mean |shift|=0.069
- Span swap: 12/45 sig shifts (27%), 1 flip (2.2%), mean |shift|=0.144
- Direction analysis: 5 positive, 8 negative shifts. 5/8 negatives involve stresstest tasks.
- Utility gap effect: shifts concentrated at |Δμ| < 10, zero shifts at |Δμ| > 10
- Created 3 plots: P(B) vs gap, span shift vs gap, position bias bars
- Report written
