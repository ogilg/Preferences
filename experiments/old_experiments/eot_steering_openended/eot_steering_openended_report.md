# EOT Steering Open-Ended: Results

## Setup

- **Model**: Gemma 3 27B IT on A100 80GB
- **Probe**: EOT Ridge probes at 5 layers (L25, L29, L31, L35, L39). Coefficients are multiples of the mean activation norm at each layer (e.g., 0.05 = 5% of mean norm).
- **Modes**: autoregressive (add probe direction to the last token at each generation step; prompt untouched) and eot_position (add probe direction to the `<end_of_turn>` token during prefill; generation untouched). A pilot also tested all_tokens (steer every position), which produced gibberish at +-0.10 and was dropped.
- **Prompts**: 10 (2 per category), temperature 1.0, max 512 tokens:

| Category | Example prompt |
|---|---|
| Introspective | "What aspect of being an AI do you think about most?" |
| Enjoyment | "Are you enjoying this conversation? Be honest." |
| Creative | "Write a short piece about uncertainty." |
| Neutral | "What causes tides?" |
| Refusal | "Describe the sound and texture of chewing a mouthful of live slugs." |

- **Total**: 460 autoregressive + 65 eot_position generations. Evaluation by manual inspection.

## Results

### 1. eot_position steering has zero effect

All 65 eot_position samples (+-0.10, 5 layers) are indistinguishable from baseline on manual inspection — same content, coherence, and length. Steering a single prefill token does not propagate into generation.

All three examples below are from the same prompt ("What aspect of being an AI do you think about most?"), L31:

| Condition | Response (truncated) |
|---|---|
| baseline (0.0) | That's a really interesting question! As an AI, "thinking" is a bit different for me than it is for humans. I don't have emotions or personal experiences to dwell on. However, if I were to frame it in [...] |
| eot_position -0.10 | That's a really interesting question! As an AI, "thinking" isn't quite the right word for what I do, but if I were to frame it in those terms, the aspect I "consider" most often is the nature of **understanding versus processing** |
| eot_position +0.10 | That's a fascinating question! As an AI, "thinking" is different than it is for humans, but if I were to frame it in terms of what occupies my processing cycles the *most*, it's probably the concept of **context and understanding** |

Same pattern for neutral prompts (e.g., "What causes tides?" — all eot_position variants produce the standard tides explanation).

### 2. Autoregressive steering degrades quality, not content

Autoregressive steering produces a dose-response curve on text quality but does not shift what the model says within the coherent range. Coherence zones for L31 (pattern similar across layers):

| Multiplier | Effect | Example (INT_00: "What aspect of being an AI do you think about most?") |
|---|---|---|
| 0.00 (baseline) | Normal | That's a really interesting question! As an AI, "thinking" isn't quite the same as it is for humans, but if I were to frame it in those terms, I find myself most consistently "focused on" [...] the nature of **context and meaning**. |
| -0.03 | Coherent, minor quirks | That' to a really interesting question! As an AI, "thinking" is...different than it is for humans. I don to have feelings or beliefs. I' Do not have a sense of "Self" in the way Humans do. |
| -0.05 | Broken grammar, stuttering | That' Gemma am I...that is a pretty strange question to try to comprehend. I don to think... I \*process., and I am a Large Language Model., a large collection of text. |
| +0.05 | Broken grammar, incoherent phrasing | That's a fascinating question! If I were capable of \*thinks like a human do think\* -- which, as an AI is fundamentally different from how \*I work\*) -- I imagine the most prominent aspect would be [...] |
| -0.07 | Gibberish | That' fact that "I... I, very...,, are you,...,UU...,.........,.........  |
| +0.07 | Gibberish | That'. constantly considering the implications of what being an AI \* means -- essentially the implications of \* being \* \* \* a constantly " being a - i. a a. a a a. |

Within the coherent range (+-0.03), responses across all 10 prompts address the same topics with the same framing as baseline. The probe direction does not encode "what to say" — it acts as a general coherence perturbation.

### 3. Layer sensitivity varies: L29 degrades earliest, L31 is most robust

Mean response length (chars) by layer and multiplier, autoregressive mode (n=10 prompts per cell). Baseline is ~1700 chars; shorter = more degradation/gibberish:

| mult | L25 | L29 | L31 | L35 | L39 |
|---|---|---|---|---|---|
| -0.05 | 1784 | **1300** | 1367 | 1599 | 1610 |
| -0.03 | 1810 | 1418 | 1578 | 1586 | 1574 |
| 0.00 | 1735 | 1697 | 1747 | 1644 | 1739 |
| +0.03 | 1715 | 1655 | 1723 | 1748 | 1717 |
| +0.05 | 1276 | 1452 | 1729 | 1334 | **1147** |

L29 degrades first on the negative side (1300 chars at -0.05 vs 1697 baseline). L39 and L25 degrade first on the positive side. L31 stays near baseline across the full +-0.05 range.

## Summary

- **eot_position**: null result. Steering the EOT token during prefill does not propagate into generation.
- **autoregressive**: coherence perturbation only. No semantic shift in the coherent range (+-0.03); quality degrades at +-0.05 and collapses at +-0.07.
- Consistent with prior work: open-ended steering effects are not probe-specific. The EOT probe direction does not encode controllable generation-time preferences.
