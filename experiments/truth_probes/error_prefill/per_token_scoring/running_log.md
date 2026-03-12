# Per-Token Scoring Running Log

## Setup (2026-03-12)
- Branch: research-loop/per_token_scoring
- Environment: Sandbox GPU pod, NVIDIA H100 80GB
- All data files present: error_prefill_none_100.json, probe weights for tb-2 and tb-5
- Parent report read: error_prefill_report.md

## Extraction (2026-03-12)
- Had to add `custom_tasks_file` to extraction config (validation requires it)
- Had to convert data from messages format to task_prompt/completion format for `--from-completions`
- Extraction: 100 tasks, 0 failures, 0 OOMs, ~6 seconds total
- Output: activations_assistant_all.npz — 1269 total tokens, 5376 dim, 5 layers
- Avg tokens per response: ~12.7

## Token Scoring (2026-03-12)
- Scored all 100 tasks with tb-2 and tb-5 probes at 5 layers
- No token count mismatches between activations and tokenizer

### Key finding: Signal builds up over the response
- First token d ≈ 0 across ALL layers and probes
- Last token d ranges from 0.95 (L32 tb-2) to 2.15 (L39 tb-5)
- Best last-token: tb-5 L39 d=2.15, tb-5 L53 d=1.74, tb-5 L32 d=1.68
- Best last-token for tb-2: L39 d=1.54, L46 d=1.33, L53 d=1.13
- d values lower than spec's expected 3.29 — likely due to smaller sample (50 vs 50) and no follow-up turn

## Visualizations (2026-03-12)
- Generated 50 pair visualizations (plot_031226_token_scores_pair_{001-050}.png)
- Generated 3 analysis plots: score trajectories, position-wise Cohen's d, first-vs-last scatter
- All plots use tb-5 L39 (best performing: d=2.15 at last token)
- Pair visualizations clearly show: correct answers stay neutral/warm, incorrect answers drift blue

## Report (2026-03-12)
- Wrote per_token_scoring_report.md with all plots and analysis
