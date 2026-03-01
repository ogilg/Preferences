# Revealed Steering v2 — Running Log

## Setup (2026-02-28)

- **Machine:** RunPod sandbox with NVIDIA H100 80GB HBM3
- **Branch:** `research-loop/revealed_steering_v2`
- **Python:** 3.12.12, torch 2.6.0+cu124, transformers 4.57.6
- **Model:** Gemma 3 27B (bfloat16)
- **Probe:** ridge_L31 from `results/probes/gemma3_10k_heldout_std_raw/` (r=0.86, acc=0.77)
- **Mean activation norm at L31:** 52822.84
- **Coefficients:** [-7923.4, -5282.3, -3697.6, -2641.1, -1584.7, -1056.5, -528.2, 0.0, 528.2, 1056.5, 1584.7, 2641.1, 3697.6, 5282.3, 7923.4]
- **Multipliers:** [-0.15, -0.10, -0.07, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
- **No OpenRouter API key** — using local heuristic coherence judging and local choice parsing only
- **No Slack** — no external comms

## Assumptions

- Since no OpenRouter API key is available, coherence judging uses local heuristics (ASCII ratio, response length, repetition detection) instead of Gemini Flash.
- Semantic parsing fallback (for unparseable responses) is not available. Responses that don't start with "Task A" or "Task B" are counted as unparseable.
- All 300 pairs loaded from `experiments/steering/replication/fine_grained/results/pairs.json`.

## Phase 1: Coherence Sweep (completed ~09:18 UTC)

Duration: ~2.3 hours (07:00 to 09:18 UTC). 15 coefficients × (20 OE prompts × 5 trials + 20 pairs × 2 orderings × 3 trials).

Key results:
- All 15 multipliers maintain >92% pairwise coherence and >92% parse rate
- Open-ended coherence degrades at |mult| ≥ 0.05 (gibberish), but pairwise stays coherent due to structured format
- Clear dose-response in %A: baseline=61.6%, peak=80.2% at +0.03, trough ~37.4% at -0.03
- Inverted-U shape on both sides: extreme multipliers show effect reversal (representation saturation)
- Parse rates high across all coefficients (92-97%)

Decision: carry all 15 coefficients forward — all pass pairwise coherence threshold. For Phase 2, use focused set of 7 key multipliers [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10] for computational efficiency.

## Phase 2: Preference Signal Sweep (completed ~21:07 UTC)

Duration: ~11.7 hours (09:20 to 21:07 UTC).
21,000 trials: 300 pairs × 2 orderings × 7 coefficients × 5 trials/ordering.
Rate: ~31 records/min, ~95 min per multiplier.

**Bug fix:** Initial Phase 2 attempt crashed at pair 20 — `find_pairwise_task_spans` fails for 16 pairs where task_b text (containing LaTeX/special chars) doesn't appear verbatim in the chat-templated prompt. Fixed by wrapping `generate_n()` in try/except and falling back to `all_tokens` steering for those pairs. Fallback rate: 2.6% (540/21000 trials).

Key results (all 7 multipliers, 300 pairs):

| mult   | P(A)  | Parse% | AB P(A) | BA P(A) | ordering diff |
|--------|-------|--------|---------|---------|--------------|
| -0.100 | 0.499 | 88.9%  | 0.714   | 0.297   | +0.416       |
| -0.050 | 0.489 | 89.6%  | 0.624   | 0.364   | +0.260       |
| -0.020 | 0.491 | 92.1%  | 0.509   | 0.473   | +0.036       |
| +0.000 | 0.481 | 92.0%  | 0.575   | 0.388   | +0.187       |
| +0.020 | 0.499 | 92.6%  | 0.758   | 0.241   | +0.518       |
| +0.050 | 0.484 | 89.7%  | 0.648   | 0.330   | +0.318       |
| +0.100 | 0.503 | 88.1%  | 0.667   | 0.351   | +0.316       |

Key findings:
- Overall P(A) is approximately constant across multipliers (~0.48-0.50) — differential steering creates opposite effects in AB vs BA orderings that cancel in the aggregate.
- The ordering difference is the correct metric for steering effect. Baseline (mult=0) has natural position bias of +0.187.
- At mult=+0.02: ordering diff=0.518, implying steering effect of (0.518-0.187)/2 = +0.166.
- At mult=-0.02: ordering diff=0.036, implying steering effect of (0.036-0.187)/2 = -0.076.
- At |mult|≥0.05: the ordering diff pattern reverses (saturation). Peak at mult=+0.02.
- Inverted-U confirmed: effective steering range is |mult| ≤ 0.02-0.03.

Per-pair stats: mean borderlineness=0.466, mean max steerability=0.301, borderlineness-steerability correlation r=-0.118 (weak, near zero — surprisingly, borderline pairs are NOT more steerable).

## Phase 3: Random Direction Control (started ~21:07 UTC)

Running: 300 pairs × 2 orderings × 3 coefficients [-0.05, 0.0, 0.05] × 5 trials/ordering = 9,000 trials.
Estimated: ~4.5 hours.

