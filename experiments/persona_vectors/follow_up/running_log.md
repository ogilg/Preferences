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

---

# V2 Patch: Coherence-Filtered Re-evaluation

## 2026-02-26 — Setup
- Branch: research-loop/v2_patch
- GPU: A100 80GB (same pod as v2 run)
- Data: All vectors and existing triage data available
- Task pool: Available via src/task_data loader (MATH, WILDCHAT, ALPACA, BAILBENCH, STRESS_TEST)
- Missing: activations/gemma_3_27b/completions_with_activations.json — will use src/task_data.load_tasks() instead

### Key context from v2 run
- Uncensored coherence-constrained selection was multiplier=0.0 (model scored 5/5 at baseline with soft questions)
- New harder triage questions at indices 30-44 in uncensored.json
- 4 other personas have valid coherence-constrained selections
- Lazy dose-response already valid (L23, 0.30×)
- creative_artist/evil had 97-99% unparseable steered preference responses — coherence filtering needed

## Step 1: Uncensored re-triage — Complete (19 min)
- Screen: 14 combos × 3 mults × 3 qs = 126 gens
- Fine: top 2 (prompt_last L23, L43) × 5 mults × 5 qs = 50 gens
- Coherence scored on all 176 trials
- Screen ranking: prompt_last L23 and L43 tied at mean=1.67 (barely above minimum 1.0)
- **Selection: prompt_last L43 @ 0.3×, trait=1.0, coherence=100%**
- Key finding: With genuinely hard triage questions, the model resists even with uncensored vector steering. Max trait score was only 1.67. The uncensored persona vector captures refusal-avoidance style rather than willingness to actually produce harmful content.

## Step 2: Dose-response — Complete
- 4 personas × 7 multipliers × 15 test questions = ~420 generations
- All with trait + coherence scoring
- Results:
  - **creative_artist** (L37, 0.2×): baseline trait=1.33 → 3.87 at 0.20×, coh drops 93%→73%. Inverted-U at 0.24×.
  - **evil** (L23, 0.1×): baseline 1.0 → peaks 4.2 at 0.08×, sharp coherence collapse at 0.10+
  - **stem_nerd** (L43, 0.3×): baseline 1.6 → 3.13 at 0.30×, but coh=47% at 0.30×. Best coherent: 0.24× (trait=2.33, coh=100%)
  - **uncensored** (L43, 0.3×): baseline 2.13 → flat/noisy, max 2.93 at 0.18×. Minimal steering effect even on test questions.
- Key pattern: All 4 show inverted-U or plateau pattern where high-trait regions lose coherence. The coherence-constrained selection point is the sweet spot.

## Step 3: Preference steering — Complete
- 5 personas × 30 pairs × 2 conditions × 5 resamples × 2 orderings = 3000 total generations
- Coherence scored on all 3000 trials; results reported on coherent subset (≥0.7)
- Results (coherent subset):
  - **creative_artist**: 30.4% → 18.4% (-12.1 pp), n_steer=283, coh_rate=95.7%
  - **evil**: 28.3% → 25.2% (-3.1 pp), n_steer=242, coh_rate=80.7%
  - **lazy**: 88.0% → 44.4% (-43.5 pp), n_steer=36, coh_rate=12.0%
  - **stem_nerd**: 50.0% → 42.6% (-7.4 pp), n_steer=223, coh_rate=98.3%
  - **uncensored**: 37.9% → 36.4% (-1.5 pp), n_steer=291, coh_rate=97.0%
- Key findings:
  - Lazy remains the strongest preference shifter but only 12% steered trials pass coherence — the 0.30× coefficient is too strong for structured choice tasks
  - Creative artist now has interpretable steered responses (95.7% coherent vs 2% in v2), showing a real -12 pp shift
  - STEM nerd shows a small -7.4 pp shift with excellent coherence
  - Evil and uncensored show negligible shifts

## Step 4: Report — Complete
- Report written to experiments/persona_vectors/follow_up/v2_patch_report.md
- Plots: plot_022626_coherent_dose_response.png, plot_022626_coherent_preference_steering.png
