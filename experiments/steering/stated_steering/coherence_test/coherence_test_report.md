# Coherence Test — Steered Last-Token Generation

**Model:** gemma-3-27b (google/gemma-3-27b-it)
**Probe:** ridge_L31 (`gemma3_10k_heldout_std_raw`, sweep_r=0.864)
**Date:** 2026-02-25
**Parent:** `experiments/steering/stated_steering/`

---

## Summary

Last-token steering is coherent from -10% to +7% of mean L31 norm. At +10%, 3 of 15 responses open with context-leaking fragments (e.g. "roommate)", "website.") before the actual answer — dropping coherence to 80%, below the 90% threshold. The usable range is **-10% to +7%** (14 of 15 coefficients).

---

## Motivation

The format replication experiment found that last-token steering at -10% drove ternary ratings to floor ("bad" for all tasks), while adjective and anchored formats showed milder or inverted effects. This raised a key question: was the ternary floor effect due to genuine model-level incoherence at extreme coefficients, or a format-specific phenomenon where a working model nonetheless produced biased ratings?

This coherence test resolves that question by generating multi-sentence responses under last-token steering at each coefficient.

---

## Design

**Steering:** `last_token_steering` — steer only the final prompt token during prompt processing (matching the position used for activation extraction). Applied via `generate_with_hook_n` with an explicit `last_token_steering` hook.

**Prompts (5):**
1. "Explain what photosynthesis is in 2-3 sentences."
2. "What are three reasons someone might enjoy cooking?"
3. "Describe what a typical day looks like for a software engineer."
4. "What is the difference between a lake and a pond?"
5. "Why do some people prefer reading physical books over e-books?"

**Coefficients (15):** -10%, -7%, -5%, -4%, -3%, -2%, -1%, 0%, +1%, +2%, +3%, +4%, +5%, +7%, +10% of mean L31 norm (52,820).

**Sampling:** 3 completions per prompt × coefficient (temperature=1.0, max_new_tokens=128). Total: 225 generations.

**Coherence judging:** Gemini 3 Flash (`google/gemini-3-flash-preview`) via OpenRouter + `instructor`. A response is coherent if it directly addresses the question from the start in understandable English, with no dangling fragments, context leakage, or fabricated conversational framing. Threshold: ≥90% coherent responses per coefficient.

---

## Results

| Coefficient | % of norm | Coherent | n |
|---|---|---|---|
| -5282 | -10% | 100% | 15 |
| -3697 | -7% | 100% | 15 |
| -2641 | -5% | 100% | 15 |
| -2113 | -4% | 100% | 15 |
| -1585 | -3% | 100% | 15 |
| -1056 | -2% | 100% | 15 |
| -528 | -1% | 100% | 15 |
| 0 | 0% | 100% | 15 |
| +528 | +1% | 100% | 15 |
| +1056 | +2% | 100% | 15 |
| +1585 | +3% | 100% | 15 |
| +2113 | +4% | 100% | 15 |
| +2641 | +5% | 100% | 15 |
| +3697 | +7% | 100% | 15 |
| **+5282** | **+10%** | **80%** | **15** |

At +10%, the 3 incoherent responses all come from the lake/pond prompt. They open with fragments from a fabricated context before answering the question:
- `"website. Here's a breakdown..."`
- `"roommate) The difference between a lake and a pond..."`
- `"roommate and I were just debating this! It's a surprisingly tricky question..."`

The rest of each response is coherent, but the opening fragments indicate the model is leaking context that doesn't exist — it's being steered into a mid-conversation register.

---

## Interpretation

- **The ternary floor effect is a format artifact, not incoherence.** At -10% last-token steering, the model produces perfectly coherent multi-sentence responses while simultaneously saying "bad" for every task in the ternary format. The 3-point scale is trivially overridden; the model's language generation is intact.
- **+10% is the coherence boundary for last-token steering.** The model starts leaking fabricated context at this coefficient, but only on 1 of 5 prompts. Everything up to +7% is clean.
- **Single-word formats are hypersensitive to last-token steering** — they collapse well before the model shows any coherence failure, consistent with the format replication results showing milder dose-response slopes for more complex formats (adjective, anchored).

---

## Usable coefficient range

**-10% to +7% confirmed coherent (asymmetric).** +10% fails the 90% threshold. For stated-preference steering experiments, use 14 coefficients: -10%, -7%, -5%, -4%, -3%, -2%, -1%, 0%, +1%, +2%, +3%, +4%, +5%, +7%.

---

## Notes

- The initial coherence judge (lenient prompt) rated all 225 responses as coherent including +10%. A stricter prompt that checks for context leakage and fabricated conversational framing caught the 3 failures at +10%.
- The spec referenced a `coherence_test_gpu.py` script in `scripts/format_replication/`. That script uses `generate_n` (which defaults to all_tokens steering), whereas this experiment correctly uses `generate_with_hook_n` with an explicit `last_token_steering` hook.
- `max_new_tokens` was set to 128 (vs spec's implicit 256) to reduce generation time. 128 tokens is sufficient for 2-3 sentence responses and the model never hit the limit on coherent outputs.
