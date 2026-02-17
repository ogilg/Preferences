# Running Log: Gemma-2 27B Base HOO Cross-Topic Generalization

## Setup (Step 0)
- Branched from `research-loop/hoo_scaled` as `research-loop/gemma2_base_hoo`
- Added `gemma-2-27b-base` to model registry (hf_name: google/gemma-2-27b)
- Created extraction config: `configs/extraction/gemma2_27b_base_prompt_last.yaml`
- Checked out stress_test data files from main
- Fixed `_format_messages` in huggingface_model.py — base models lack chat templates, added fallback to concatenate message content
- GPU: NVIDIA H100 80GB HBM3

## Extraction (Step 1)
- Extracted 30,000 activations at layers [11, 23, 27, 32, 36, 41] (fractions [0.25, 0.5, 0.6, 0.7, 0.8, 0.9] of 46)
- 0 failures, 0 OOMs
- Batch size 8, ~15 min for 19k batches
- 3000/3000 overlap with preference scores (stress_test fix worked)
- Output: activations/gemma_2_27b_base/activations_prompt_last.npz (3.3GB)

## HOO Raw (Step 2a)
- Config: configs/probes/gemma2_base_hoo_raw.yaml
- Alpha sweep: best_alpha=1e+04 (vs Gemma-3's 2154)
- Results (56 folds, all layers):

| Layer | val_r | hoo_r | gap |
|-------|-------|-------|-----|
| 11 | 0.761 | 0.497 | 0.264 |
| 23 | 0.794 | 0.579 | 0.215 |
| 27 | 0.786 | 0.575 | 0.211 |
| 32 | 0.770 | 0.553 | 0.217 |
| 36 | 0.771 | 0.543 | 0.228 |
| 41 | 0.773 | 0.544 | 0.229 |

Best layer: L23 (hoo_r=0.579)

## HOO Demeaned (Step 2b)
- Config: configs/probes/gemma2_base_hoo_demeaned.yaml
- Results (56 folds, all layers):

| Layer | val_r | hoo_r | gap |
|-------|-------|-------|-----|
| 11 | 0.496 | 0.405 | 0.091 |
| 23 | 0.542 | 0.499 | 0.044 |
| 27 | 0.532 | 0.532 | 0.000 |
| 32 | 0.507 | 0.512 | -0.006 |
| 36 | 0.509 | 0.518 | -0.010 |
| 41 | 0.513 | 0.528 | -0.014 |

Best layer: L27 (hoo_r=0.532, gap=0.0004)
Generalization gap effectively zero — within-topic signal transfers perfectly.

## Statistical Tests (Step 3)
- Gemma-2 base vs ST baseline: t=33.5, p<10^-37, 56/56 fold wins
- Gemma-2 base vs Gemma-3 IT: t=-22.2, p<10^-28, 0/56 Gemma-2 wins
- Report and plots written
