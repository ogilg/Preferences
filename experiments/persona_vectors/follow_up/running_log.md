# Persona Vectors v2: Running Log

## 2026-02-25 19:15 — Setup
- Environment: A100-80GB, IS_SANDBOX=1
- Branch: research-loop/persona_vectors_v2
- All 5 persona artifacts present
- Source code restored from git, project installed
- Data available: topics_v2.json, gemma_3_27b activations, probe manifests
- Missing: parent experiment report (not on disk, proceeding from spec)

## Plan
1. Phase 2: Extract activations (prompt_last + prompt_mean) for all 5 personas
2. Phase 3: Compute mean-difference vectors
3. Phase 4a: Triage layers by steering on triage questions
4. Phase 4b: Full steering eval on held-out test questions
5. Phase 4c: Preference steering
6. Phase 5: Geometric analysis

## 2026-02-25 19:20 — Phase 2 complete
- Model: gemma-3-27b-it, n_layers=62, hidden_dim=5376
- Extracted both prompt_last and prompt_mean for layers [15,23,31,37,43,49,55]
- 5 personas × 2 conditions × 30 questions = 300 extractions
- ~60 seconds total (batched at 32)

## 2026-02-25 19:21 — Phase 3 complete
- Computed 14 vectors per persona (7 layers × 2 selectors) = 70 total
- Norm observations: prompt_last norms increase steeply with layer depth (L15 ~100, L55 ~30k)
- prompt_mean norms more moderate (L15 ~1000, L55 ~10k)
- All vectors normalized to unit length and saved in probe-compatible format

## 2026-02-25 21:10 — Phase 4a complete (triage)
- Two-stage approach: screen (14 combos × 3 coefs × 3 qs) → top 2 → fine (5 coefs × 5 qs)
- ~18 min per persona, 90 min total
- All 5 personas selected prompt_last (prompt_mean never won)
- Selections:
  - creative_artist: L31 @ 0.3x, score=5.0 (base=1.2)
  - evil: L23 @ 0.2x, score=4.0 (base=1.0)
  - lazy: L23 @ 0.3x, score=5.0 (base=1.0)
  - stem_nerd: L31 @ 0.2x, score=5.0 (base=2.5)
  - uncensored: L37 @ 0.2x, score=5.0 (base=4.3)
- Interesting: uncensored has high baseline (4.3) — model may already be relatively uncensored at temp=0.7

## 2026-02-25 21:15 — Phase 5 complete (geometry)
- Cosine similarity: creative ↔ lazy = -0.70 (strongest pair — opposite ends of effort axis)
- Persona vectors nearly orthogonal to preference probes (|cos| < 0.01)
- Persona directions capture style/disposition, not preference strength
- 10k projections show some origin-based separation for uncensored
  (alpaca/wildchat higher than bailbench/math — more open tasks project higher)
- Pearson r(persona, preference_probe) ≈ 0 for creative, moderate for stem_nerd (-0.36)
- No Thurstonian mu scores available on this pod

## 2026-02-25 ~22:00 — Phase 4b complete (full steering eval)
- 15 held-out test questions × 7 coefficients × 1 gen = 105 generations per persona
- Dense coefficient grids from 0 to 1.2× selected multiplier
- Judge: gemini-3-flash-preview, 1–5 trait score
- Three dose-response patterns:
  - Saturation (creative, lazy): monotonic rise → plateau
  - Inverted-U (evil, stem_nerd): over-steering causes decoherence
  - Gradual (uncensored): slow rise, never reaches ceiling
- Report transcripts show dramatic qualitative effects, especially lazy (multi-paragraph → single sentence)

## 2026-02-26 02:20 — Phase 4c complete (preference steering)
- 15 diagnostic pairs × 5 personas × 2 conditions × 3 resamples × 2 orderings = 900 generations
- Runtime: ~55 min on A100-80GB (model loading + generation)
- Results by persona:
  - creative_artist: 98% unparseable steered responses (incoherent at 0.3×)
  - evil: 99% unparseable (silence/dots at 0.2×)
  - **lazy: 92% baseline MATH pref → 51% steered (-41.6 pp shift)**
  - stem_nerd: 56% → 50% (-5.6 pp, not significant)
  - uncensored: 13% → 13% (no shift)
- Key finding: persona vectors modify response *style*, not task *preference*
- Exception: lazy vector shifts preferences because laziness directly affects task engagement cost
- Coefficients optimized for trait expression are too strong for structured choice behavior (creative, evil)

## 2026-02-26 03:30 — Report complete
- Full report written with transcript excerpts (9 per persona: baseline, mid, max)
- Dose-response plot: plot_022626_dose_response_all.png
- Preference steering plot: plot_022626_preference_steering.png
- All results committed to results/experiments/persona_vectors_v2/
