# Content-orthogonal probing with Gemma-2 27B base as content encoder

## Goal

Repeat the content-orthogonal analysis from `docs/logs/research_loop_content_orthogonal.md`, but replace the sentence-transformer (all-MiniLM-L6-v2, 384d) with **Gemma-2 27B base** (`google/gemma-2-27b`) as the content encoder. This tests whether a much more powerful content model captures more content variance, shrinking the content-orthogonal residual.

## Method

1. **Extract content embeddings** from Gemma-2 27B base: run each task prompt through the model, take the last-token hidden state as the content embedding. This gives ~3584d embeddings instead of 384d.
2. **Run the same pipeline**: residualize Gemma-3-27B activations against Gemma-2 27B base embeddings, then train probes on the residuals.
3. **Compare** to the sentence-transformer baseline (already computed — see research log).

## Existing code

- `src/probes/content_embedding.py` — current sentence-transformer embedding code. Adapt or write a new function for Gemma-2 base.
- `src/probes/content_orthogonal.py` — residualization + probe pipeline. Should work as-is with different embeddings.
- `scripts/content_orthogonal/compare_probes.py` — main comparison script. Adapt to load the new embeddings.
- `scripts/content_orthogonal/embed.py` — current embedding script. Reference for the flow.

## Data on pod

- `activations/gemma_3_27b/activations_prompt_last.npz` — Gemma-3-27B activations (the model being probed)
- `activations/gemma_3_27b/completions_with_activations.json` — task prompts + metadata
- `results/experiments/gemma3_3k_run2/` — preference scores (Thurstonian utilities)
- `src/analysis/topic_classification/output/topics_v2.json` — topic labels for residualization

## Key numbers to beat (sentence-transformer baseline)

| Layer | Standard probe R² | Content-orth R² | % Retained |
|-------|------------------|-----------------|------------|
| 31    | 0.863            | 0.237           | 27.5%      |
| 43    | 0.840            | 0.209           | 24.8%      |
| 55    | 0.835            | 0.198           | 23.8%      |

Content-only baseline (sentence-transformer → preferences): R² = 0.521

If Gemma-2 base captures more content variance, the content-orthogonal R² should drop and % retained should shrink. If it stays similar, the residual signal is robust to content encoder choice.
