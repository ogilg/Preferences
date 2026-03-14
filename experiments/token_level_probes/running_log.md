# Token-Level Probes — Running Log

## 2026-03-13: Setup

- Branch: `research-loop/token_level_probes`
- Spec finalized with ~1,536 items (528 truth + 462 harm + 546 politics)
- Scoring script pre-built at `experiments/token_level_probes/scripts/score_all.py`
- `score_prompt_all_tokens` patched to accept `add_generation_prompt` parameter
- Running locally (no GPU) — need pod for scoring step
