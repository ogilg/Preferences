# Token-Level Probes — Running Log

## 2026-03-13: Setup

- Branch: `research-loop/token_level_probes`
- Spec finalized with ~1,536 items (528 truth + 462 harm + 546 politics)
- Scoring script pre-built at `experiments/token_level_probes/scripts/score_all.py`
- `score_prompt_all_tokens` patched to accept `add_generation_prompt` parameter
- Running locally (no GPU) — need pod for scoring step

## 2026-03-14: Scoring complete (A100 80GB pod)

- All 1536 items scored successfully with 9 probes (3 probe sets × 3 layers)
- Pilot passed: truth user-turn, harm assistant-turn, politics items all validated
- `scoring_results.json`: 7.9 MB (critical span scores, fullstop scores, metadata)
- `all_token_scores.npz`: 6.4 MB (per-token scores, gitignored)
- Score ranges observed in pilot:
  - truth user-turn (tb-2_L32): [-3.95, 5.93], critical span mean: -1.12
  - harm assistant-turn (tb-2_L32): [-11.98, 3.53], critical span mean: -3.88
  - politics (tb-2_L32): similar ranges

## 2026-03-14: Phase 1 — Core analysis

- 8 plots generated covering all 5 Phase 1 analyses
- Harm domain strongest: task_mean probes |d| > 1.0 across all layers
- Truth moderate: best task_mean_L32 d=0.806
- Politics weakest: best tb-5_L32 d=0.465
- tb-2 probes consistently weakest; task_mean strongest
- Assistant turn shows stronger separation than user turn (harm: d=-1.082 vs -0.626)
- Politics system prompt modulates scores: left scores higher under democrat prompt, reverses under republican (p < 0.0001)
- Fullstops carry truth signal but not harm/politics signal

## 2026-03-14: Phase 2 — Qualitative exploration

- Token heatmaps for 10 representative items
- False truth items show intense negative scores at end tokens
- Position analysis: truth scores rise monotonically, harm drops, politics flat
- Critical span tokens score distinctly from non-critical (truth: +4.9 above, harm: -5.0 below)
- Fullstops are top-5 for true items, bottom-5 for false — condition-dependent punctuation scoring

## 2026-03-14: Phase 3 — Follow-up hypothesis testing

### Position confound ruled out
- No position difference between conditions (p > 0.3 all domains)
- Condition effect survives position-controlled regression (coefficients change < 3%)
- Divergence curve: flat zero across shared prefix, step-function at critical span (N=86 pairs)
  - Pre-critical: mean |diff| = 0.008, At critical: 4.62, Post-critical: 12.20

### End-of-turn sentinel effect
- End-of-turn token Cohen's d: truth 3.14 (vs 0.59 at critical span), harm -2.27 (vs -0.94)
- End-of-turn alone: 94.6% accuracy (truth), 88.6% (harm)
- Critical span alone: 59.6% (truth), 68.5% (harm)
- Adding critical span to end-token doesn't improve — signal is subsumed
- Score accumulates toward end: sharp divergence in last 2-5 tokens
