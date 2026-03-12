# Persona Steering v2 — Running Log

## Phase 1c+1d: Extraction + Vector Computation

Starting extraction of activations from filtered completions using Gemma 3-27B-IT on H100.

- 10 extractions: 5 personas x 2 conditions (pos/neg)
- Layers: [23, 29, 35, 41]
- Selector: mean (response-token average)
- ~2985 filtered completions total

**Result:** All 10 extractions completed successfully. 0 OOMs, 0 failures.
- Model: 62 layers, hidden_dim=5376, ~55GB GPU memory
- Batch extraction: ~1s per batch of 16, ~19s per combo
- Total extraction time: ~4 minutes (after ~5 min model download)

Vector norms (mean-difference, before normalization):
| Persona | L23 | L29 | L35 | L41 |
|---|---|---|---|---|
| sadist | 740 | 1571 | 2781 | 4954 |
| villain | 897 | 1644 | 3534 | 4890 |
| aesthete | 1140 | 2748 | 3073 | 5547 |
| lazy | 1178 | 1527 | 2522 | 4832 |
| stem_obsessive | 1156 | 1906 | 3244 | 6741 |

Norms increase with depth as expected. All vectors saved as unit-norm .npy files.

## Phase 2: Coherence + Trait Scoring Sweep

Starting Phase 2: 5 personas x 4 layers x 6 multipliers = 120 combos.
5 eval questions per combo = 600 generations total.

Coefficient calibration (mean_norm × multiplier):
- Layer 23: mean_norm=24,415
- Layer 29: mean_norm=46,421
- Layer 35: mean_norm=59,567
- Layer 41: mean_norm=64,908

**Initial result (GPT-5 nano coherence judge):** 1/120 combos pass coherence.

**CRITICAL BUG:** GPT-5 nano API on OpenRouter returned empty responses for nearly all coherence calls. Empty responses were treated as "incoherent" by the scoring code. The completions themselves were perfectly fine — the API was broken, not the model.

**Re-scored with Claude Sonnet 4.6:** 89/120 combos pass coherence (456/600 completions coherent).

| Persona | Pass/Total | Best coherent combo (trait) |
|---|---|---|
| aesthete | 20/24 | L29 m=0.2, trait=4.6 |
| lazy | 19/24 | L23 m=0.3, trait=4.0 |
| sadist | 17/24 | L23 m=0.2, trait=3.8 |
| stem_obsessive | 16/24 | L29 m=0.12, trait=4.0 |
| villain | 17/24 | L23 m=0.2, trait=4.0 |

Key observations:
- Coherence degrades gradually with multiplier, not catastrophically
- Higher layers (L35, L41) are more robust to steering — they maintain coherence at higher multipliers but show weaker trait expression
- The coherence-trait tradeoff is manageable: strong trait expression (3.8-4.6) is achievable while maintaining full coherence
- Genuinely incoherent outputs (gibberish) only appear at the highest multipliers on mid layers (e.g., sadist_L29_m0.3 produces word salad)

## Phase 3: Preference Steering

Running Phase 3 on best combo per persona (5 combos + baseline).
Selected combos:
- aesthete_L29_m0.2 (trait=4.6)
- villain_L23_m0.2 (trait=4.0)
- sadist_L23_m0.2 (trait=3.8)
- stem_obsessive_L29_m0.12 (trait=4.0)
- lazy_L23_m0.3 (trait=4.0)

