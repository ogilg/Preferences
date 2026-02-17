# Scaled HOO Topic Generalization

**Goal**: Test whether activation probes generalize across topics when holding out 3/8 topics (harder than 1/8). Compare Ridge (raw + demeaned), Bradley-Terry, and sentence-transformer content baseline across all C(8,3)=56 folds.

**Result**: Activation probes massively outperform the content baseline (0.78 vs 0.24 hoo_r, p < 10^-50). Demeaned probes close the val-hoo gap to near zero (0.009), confirming within-topic preference signal transfers perfectly. BT generalizes (hoo_acc=0.65) but with a larger gap than Ridge.

## Setup

- Model: Gemma 3 27B, layers 31/43/55
- Data: 3000 tasks, 8 topics, Thurstonian scores from gemma3_3k_run2
- Folds: 56 combinations of holding out 3 topics (train on 5, eval on 3)
- Content baseline: all-MiniLM-L6-v2 sentence-transformer (384d)

Note: v2 topic classification has 12 topics, not the 8 assumed in the spec. We restricted to the 8 largest topics via `hoo_groups` config. The 4 extra topics (model_manipulation, other, security_legal, sensitive_creative) are small and always included in training.

## Results

| Method | Layer | Mean hoo_r | Std | Gap (val - hoo) |
|--------|-------|-----------|-----|-----------------|
| Ridge raw | 31 | **0.779** | 0.117 | 0.121 |
| Ridge raw | 43 | 0.704 | 0.185 | 0.177 |
| Ridge raw | 55 | 0.711 | 0.169 | 0.165 |
| Ridge demeaned | 31 | **0.706** | 0.157 | **0.009** |
| Ridge demeaned | 43 | 0.625 | 0.215 | 0.055 |
| Ridge demeaned | 55 | 0.630 | 0.196 | 0.039 |
| ST baseline | 0 | 0.245 | 0.116 | 0.423 |

BT uses pairwise accuracy (not Pearson r):

| Method | Layer | Mean hoo_acc | Std | Gap (val - hoo) |
|--------|-------|-------------|-----|-----------------|
| BT raw | 31 | 0.652 | 0.055 | 0.190 |
| BT raw | 43 | 0.617 | 0.064 | 0.248 |
| BT raw | 55 | 0.610 | 0.064 | 0.259 |

Paired Ridge raw L31 vs ST baseline: t=72.9, p<10^-50, 56/56 fold wins.

![Box plot](assets/hoo_scaled/plot_021126_hoo_scaled_boxplot_L31.png)

![Per-topic breakdown](assets/hoo_scaled/plot_021126_hoo_scaled_per_topic_L31.png)

## Key Insights

1. **Content alone explains very little**: ST baseline hoo_r = 0.24 despite val_r = 0.67. Task content can fit within-distribution but fails to generalize — the fitted probe memorizes topic-specific patterns.

2. **Activation probes capture genuine evaluative signal**: 0.78 hoo_r means the probe ranks preferences of unseen-topic tasks with r=0.78 correlation. This signal is not explainable by content alone.

3. **Demeaning is the cleanest evidence**: After removing topic means, val_r = 0.715 and hoo_r = 0.706 — virtually no generalization gap. The probe captures within-topic preference variation that is consistent across all topics.

4. **Layer 31 is consistently best**: Early-middle layer outperforms deeper layers, especially for math (L43/L55 struggle with math-containing folds).

5. **BT generalizes with a larger gap**: BT hoo_acc=0.65 (above 0.50 chance) confirms pairwise signal transfers, but the val-hoo gap (0.19) is larger than Ridge (0.12). BT's lower variance (std=0.055) reflects accuracy being a less sensitive metric than Pearson r.

## Process Notes

- The spec assumed 8 topics (C(8,3)=56 folds). The actual v2 classification has 12 topics (C(12,3)=220 folds). First run burned ~65 folds on the 220-fold version before discovering this. Fixed with `hoo_groups` config to restrict to 8 topics.
- BT was too slow for 56 folds × 3 layers locally (each L-BFGS fit takes 2-5 min on 5376d). Ran in background for ~3 hours total.
- Ridge and ST baseline runs completed in ~10 min each (closed-form solution).

## Dead Ends

- First run without `hoo_groups` produced 220 folds (12 topics). Killed after 65 folds.
- Combined Ridge+BT config was too slow. Split into separate configs.
