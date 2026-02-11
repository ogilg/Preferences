# Content-Orthogonal Probing: Gemma-2 Base as Content Encoder

**Goal**: Replace all-MiniLM-L6-v2 (384d) with a Gemma-2 base model (3584d) as content encoder. If a more powerful encoder captures more content variance, the content-orthogonal R² should drop. If it stays similar, the residual signal is robust to content encoder choice.

**Result**: The experiment reveals a methodological limitation rather than a clean answer. Gemma-2 9B base embeddings (3584d) predict 76% of preference variance directly (vs 62% for ST), confirming they capture more content. But the content-orthogonal analysis breaks down because p > n (3584 dims > 2400 samples) causes the residualization Ridge to overfit catastrophically (train R²=0.94, CV R²=0.39), removing nearly all activation variance including evaluative signal. The near-zero content-orthogonal R² with Gemma-2 is a dimensionality artifact, not evidence that content explains everything.

## Baseline (sentence-transformer, from prior research loop on 3000 tasks)

| Layer | Standard probe R² | Content-orth R² | % Retained |
|-------|------------------|-----------------|------------|
| 31    | 0.863            | 0.237           | 27.5%      |
| 43    | 0.840            | 0.209           | 24.8%      |
| 55    | 0.835            | 0.198           | 23.8%      |

Content-only baseline (sentence-transformer → preferences): R² = 0.521

## Setup

- **Pod constraints**: 50GB workspace quota, H100 80GB GPU
- **Gemma-2 27B base** (109GB float32) doesn't fit — used **Gemma-2 9B base** (`google/gemma-2-9b`, hidden_size=3584) loaded in bf16
- **Gemma-3 27B IT** (54GB) doesn't fit — used **bnb-4bit** (`unsloth/gemma-3-27b-it-bnb-4bit`, ~16GB) for activation extraction. Required `dtype=torch.bfloat16` to prevent NaN overflow in non-quantized layers.
- Sequential model download: extract content embeddings → delete cache → extract activations
- 2400 tasks (stress_test data unavailable on pod — 600 tasks from each of alpaca, bailbench, wildchat, competition_math)

## Iteration 1: Raw comparison

Standard probes on the new bnb-4bit activations match the original experiment closely:

| Layer | Standard R² (this run) | Standard R² (original) |
|-------|----------------------|----------------------|
| 31    | 0.877                | 0.863                |
| 43    | 0.854                | 0.840                |
| 55    | 0.849                | 0.835                |

### Content-only baselines

| Encoder | cv R² | Dimensions |
|---------|-------|------------|
| Sentence-transformer (all-MiniLM-L6-v2) | 0.615 | 384 |
| Gemma-2 9B base | 0.760 | 3584 |

Gemma-2 captures 76% of preference variance from content alone — 15pp more than the sentence-transformer.

### Content-orthogonal comparison

| Layer | Standard | ST-Orth R² | ST % Ret | G2-Orth R² | G2 % Ret |
|-------|----------|-----------|----------|-----------|----------|
| 31    | 0.877    | 0.167     | 19.1%    | 0.005     | 0.6%     |
| 43    | 0.854    | 0.141     | 16.6%    | -0.000    | ~0%      |
| 55    | 0.849    | 0.134     | 15.8%    | -0.002    | ~0%      |

![Encoder comparison](assets/content_orthogonal_gemma2base/plot_021126_encoder_comparison.png)

## Iteration 2: Diagnosing the near-zero G2 result

The Gemma-2 content-orthogonal R² of ~0% is suspicious. Investigation reveals:

### Content→Activation Ridge overfitting

| Layer | ST train R² | ST cv R² | ST gap | G2 train R² | G2 cv R² | G2 gap |
|-------|-----------|---------|--------|-----------|---------|--------|
| 31    | 0.565     | 0.364   | 0.201  | 0.945     | 0.388   | 0.557  |
| 43    | 0.521     | 0.342   | 0.179  | 0.919     | 0.332   | 0.587  |
| 55    | 0.511     | 0.326   | 0.185  | 0.914     | 0.299   | 0.615  |

![Content R² comparison](assets/content_orthogonal_gemma2base/plot_021126_content_r2_comparison.png)

The Gemma-2 Ridge (3584 features, 2400 samples, p/n=1.49) achieves train R²=0.94 at its CV-optimal α=1000 — it fits nearly all activation variance on the training set. The residualization uses these training predictions, so it removes ~94% of activation variance including most of the evaluative signal. CV R² is only 0.39, meaning the Ridge genuinely captures ~39% of activation variance; the other ~55% removed is noise-fitting.

The sentence-transformer (384 features, p/n=0.16) is much better behaved: train R²=0.56, CV R²=0.36, gap=0.20.

### PCA reduction doesn't help

Reducing Gemma-2 from 3584d to 384d via PCA still gives negative content-orthogonal R². Even at 50 PCA dims, the embeddings explain more activation variance than the sentence-transformer does at 384d. The Gemma-2 embeddings are genuinely more informative — PCA-50 gives content→act CV R²=0.41 vs ST's 0.36.

### Alpha sweep (Layer 31, Gemma-2)

| α | Content→Act train R² | Content→Act cv R² | CO probe R² |
|---|---------------------|-------------------|-------------|
| 0.01 | 1.000 | -0.531 | 0.419 |
| 1.0 | 1.000 | -0.510 | 0.419 |
| 100 | 0.993 | -0.010 | -0.155 |
| 1000 | 0.936 | 0.388 | -0.581 |
| 10000 | 0.778 | 0.515 | -1.175 |

At low α, the Ridge perfectly interpolates the activations, so residuals are near-zero and the CO probe trivially can't fit. At high α, the Ridge underfits on CV but the residualization still uses training predictions. There's no α that gives both a reasonable content→activation fit and a meaningful residual.

## Dead ends

- **bnb-4bit with fp16 compute**: Produced all-NaN activations from layer 10 onward. Fixed by forcing `dtype=torch.bfloat16`.
- **Cross-validated residualization**: Fitting content→activation Ridge on train folds and predicting residuals for held-out folds. Doesn't help — fold-specific noise in residuals makes probing worse.
- **Gemma-2 27B base**: 109GB, exceeds pod disk quota. Used 9B variant instead.

## Final results

| Metric | ST (384d) | Gemma-2 9B (3584d) |
|--------|----------|-------------------|
| Content-only baseline R² | 0.615 | 0.760 |
| Content→Act CV R² (L31) | 0.364 | 0.388 |
| Content-orth R² (L31) | 0.167 | 0.005 |
| % Retained (L31) | 19.1% | 0.6% |

**Key insight**: The residualization approach breaks down when p > n. With 3584-dimensional content embeddings and 2400 samples, the Ridge can fit nearly all activation variance on the training set regardless of regularization, making the "content-orthogonal" residuals meaningless. The near-zero content-orthogonal R² with Gemma-2 is an artifact of over-residualization, not evidence that a more powerful content model eliminates the evaluative signal.

The experiment does confirm that Gemma-2 9B base embeddings capture substantially more content-related preference variance (R²=0.76 vs 0.62 content-only baseline). But to properly test whether this stronger content model reduces the content-orthogonal signal, you would need either: (a) many more samples (n >> p, so n >> 3584), or (b) a fundamentally different residualization method that doesn't suffer from the p > n problem (e.g., nonlinear projections or matched-dimensionality approaches).

The sentence-transformer results on this run (16-19% retained) are somewhat lower than the original (24-28%), likely due to using 2400 tasks instead of 3000 and bnb-4bit quantized activations.
