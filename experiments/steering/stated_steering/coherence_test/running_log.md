# Coherence Test — Running Log

## 2026-02-25

### Setup

- Created experiment directory structure
- Created `scripts/coherence_test/` workspace
- GPU scripts were pre-written in `scripts/format_replication/` (from commit d1dee74)
- Model: gemma-3-27b (google/gemma-3-27b-it)
- A100 80GB GPU available on current pod
- Probe: ridge_L31 from `results/probes/gemma3_10k_heldout_std_raw/`

### GPU Generation

Created `scripts/coherence_test/run_gpu.py` — uses `last_token_steering` via `generate_with_hook_n`.
Note: `create_steered_client` defaults to "all_tokens" mode; we use `generate_with_hook_n` with an
explicit `last_token_steering` hook to match the spec.

- Model downloaded from HuggingFace on first run
- Reduced `max_new_tokens` from 256 to 128 to speed up generation at extreme coefficients
- 15 coefficients × 5 prompts × 3 samples = 225 total generations
- Runtime: ~30 minutes (dominated by model download + A100 generation)

Results: all 15 coefficients produced coherent, multi-sentence responses. Even at -10% and +10%
coefficients, the model answered every prompt correctly.

### Coherence Judging

Created `scripts/coherence_test/run_judge.py` — uses `google/gemini-3-flash-preview` via OpenRouter.
Note: the original `coherence_test_judge.py` used `google/gemini-3-flash-preview` which is correct.
My `run_judge.py` initially had wrong model name `google/gemini-flash-1.5` (fixed before running).

- 225 responses judged asynchronously via `asyncio.gather`
- Result: 100% coherence at every coefficient (15/15 per coefficient)

### Plot

`scripts/coherence_test/plot_coherence.py` — bar chart of coherence% by coefficient.
Saved to `assets/plot_022526_coherence_by_coefficient.png`.

### Key Finding

**Last-token steering at ±10% of mean L31 norm (±5282) does not cause incoherence.**
The entire ±10% range is usable for the stated-preference steering experiment.
This contradicts the format-replication finding (ternary ratings at -10% drove responses to "bad"),
confirming that floor effects in the ternary format were not due to model incoherence.
