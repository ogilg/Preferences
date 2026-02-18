# Persona OOD Phase 2 — Running Log

## Setup
- Date: 2026-02-18
- GPU: NVIDIA H100 80GB HBM3
- Branch: research-loop/persona_ood_phase2

## Plan
1. Extract activations for 101 core tasks × 22 conditions (no_prompt, neutral, 10 broad, 10 narrow) at layers [31, 43, 55]
2. Score with primary probes (gemma3_3k_std_raw/ridge_L31, gemma3_3k_std_demean/ridge_L31)
3. Compute behavioral deltas from v2_results.json
4. Correlate probe deltas with behavioral deltas
5. Run controls (shuffled, cross-persona)

## Step 1: Extraction
- Extracted 21 conditions × 101 tasks at layers [31, 43, 55]
- Model: google/gemma-3-27b-it (5376 hidden dim, 62 layers)
- ~4s per condition, ~85s total
- Saved to activations/persona_ood_phase2/{condition}.npz
- no_prompt condition uses existing activations from activations/gemma_3_27b/activations_prompt_last.npz (29,996 tasks, filtered to 101 core tasks)
- Note: Gemma-3 doesn't have native system role; tokenizer merges system content into user turn

## Step 2: Primary Analysis Results

### Best probe: demean/ridge_L31
- Pooled r = 0.481 (p < 1e-100, n=2014)
- 20/20 personas with r > 0.2 (all significant)
- Sign agreement = 65.2%
- Broad r = 0.462, Narrow r = 0.538
- Shuffle control: mean r = 0.001 ± 0.023, p < 0.001
- Cross-persona control: mean r = 0.309

### raw/ridge_L31
- Pooled r = 0.437 (p < 1e-94, n=2014)
- 20/20 personas with r > 0.2
- Sign agreement = 66.3%
- Broad r = 0.449, Narrow r = 0.442
- Cross-persona control: mean r = 0.257

### Baseline check (Metric 5)
- no_prompt vs neutral correlation: r = 0.992 (raw L31), 0.971 (demean L31)
- Mean score differences small but nonzero (neutral shifts scores up slightly)

### Key observations
- All success criteria met: pooled r > 0.3, all 20/20 > 0.2, sign agreement > 60%
- Demean probe slightly better than raw on pooled correlation (0.481 vs 0.437)
- Cross-persona r (~0.26-0.31) substantially lower than matched r (~0.44-0.48), confirming persona-specific tracking
- L31 best across both probe types; later layers (L43, L55) slightly worse
- Philosopher persona consistently best tracked (r = 0.65-0.73 raw, 0.41-0.65 demean)
- Trivia_nerd consistently worst tracked (r = 0.05-0.29)
