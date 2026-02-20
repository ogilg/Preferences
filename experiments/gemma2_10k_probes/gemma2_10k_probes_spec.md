# Gemma-2 27B Base: 10k Probe Training

Does scaling from 3k to 10k training examples improve Gemma-2 base probes, and by how much compared to Gemma-3 IT scaling?

## Context

Gemma-3 IT probes benefit modestly from 10k training (+0.023 r raw, +0.062 r demeaned at L31). Gemma-2 base was previously evaluated with 3k Gemma-3 IT preference scores (HOO r=0.579 raw, 0.532 demeaned). This experiment asks whether the same scaling benefit applies to Gemma-2 base activations.

## Evaluations

1. **Heldout raw**: Ridge probe trained on 10k Gemma-3 IT scores, evaluated on separate 4k heldout set
2. **Heldout demeaned**: Same but with topic-demeaned scores (within-topic signal only)
3. **HOO cross-topic**: 12-fold HOO (one topic held out per fold), trained on 10k

## Data

- Train: `results/experiments/gemma3_10k_run1/` (10k Thurstonian scores, Gemma-3 IT)
- Eval: `results/experiments/gemma3_4k_pre_task/` (4k separate measurement run)
- Activations: `activations/gemma_2_27b_base/activations_prompt_last.npz` (30k tasks, 100% overlap)
- Layers: [11, 23, 27, 32, 36, 41] — fractions [0.25, 0.5, 0.6, 0.7, 0.8, 0.9] of 46 total
- Topics: `src/analysis/topic_classification/output/topics.json` (v1, 30k coverage)

## Comparison targets

- Gemma-2 3k HOO r (raw): 0.579 at L23
- Gemma-2 3k HOO r (demeaned): 0.532 at L27
- Gemma-3 10k heldout r (raw): 0.864 at L31
- Gemma-3 10k HOO r: 0.817 at L31
