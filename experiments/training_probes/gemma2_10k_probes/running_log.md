# Running Log: Gemma-2 10k Probe Training

## Setup
- Branch: research-loop/gemma2_10k_probes (at main HEAD)
- Activations: ~/Dev/MATS/Preferences/activations/gemma_2_27b_base/activations_prompt_last.npz — 30k tasks, 100% overlap with 10k train and 4k eval
- Copied activations to activations/gemma_2_27b_base/activations_prompt_last.npz
- Copied topics.json from Preferences-role-prompting repo
- Created configs: gemma2_10k_heldout_std_raw.yaml, gemma2_10k_heldout_std_demean.yaml, gemma2_10k_hoo_dataset.yaml
- Initial topics.json only covered 1475/10k tasks; full 30k-coverage file found at ~/Dev/MATS/Preferences/data/topics/topics.json (Claude Sonnet 4.5 classifier, 100% coverage)

## Heldout Raw (Step 2a)
- Config: configs/probes/gemma2_10k_heldout_std_raw.yaml
- All layers use alpha=4642

| Layer | Heldout r |
|-------|-----------|
| L11 | 0.7103 |
| L23 | 0.7672 |
| L27 | 0.7615 |
| L32 | 0.7402 |
| L36 | 0.7319 |
| L41 | 0.7310 |

Best: L23 (r=0.767). Compares to Gemma-3 10k L31 r=0.864.

## Heldout Demeaned (Step 2b)
- Only 1475 train tasks retained after demeaning (topics.json covers 15% of 10k tasks)
- Results unreliable due to tiny effective training set — not reported

## Topic HOO (Step 3)
- Config: configs/probes/gemma2_10k_hoo_topic.yaml
- 12 folds (one topic held out per fold), topics from full 30k topics.json

| Layer | Val r | HOO r (mean) | Gap |
|-------|-------|--------------|-----|
| L11 | 0.756 | 0.529 | 0.227 |
| L23 | 0.796 | 0.605 | 0.192 |
| L27 | 0.784 | 0.574 | 0.210 |
| L32 | 0.769 | 0.553 | 0.216 |
| L36 | 0.765 | 0.550 | 0.215 |
| L41 | 0.768 | 0.564 | 0.205 |

Best: L23 (HOO r=0.605). Math worst (r=0.228). Compares to Gemma-3 10k L31 HOO r=0.817.

## Demeaned Heldout (Step 4)
- Config: configs/probes/gemma2_10k_heldout_std_demean.yaml
- Train R²=0.377 (11 features, 10000 tasks retained), Eval R²=0.288

| Layer | Heldout r |
|-------|-----------|
| L11 | 0.548 |
| L23 | 0.610 |
| L27 | 0.610 |
| L32 | 0.571 |
| L36 | 0.563 |
| L41 | 0.566 |

Best: L27/L23 tied (r=0.610). Compares to Gemma-3 10k L31 demeaned r=0.761.

## Pairwise accuracy (real pairwise choices)

Both heldout and HOO pairwise accuracy computed against real pairwise choices from
measurements.yaml (NOT Thurstonian scores — that would inflate accuracy and be incomparable
to Gemma-3's hoo_acc which uses real choices).

**Heldout acc (L23, real choices):** 0.708 (5091 unique pairs, 15201 measurements, final eval split)
Script: `scripts/gemma2_10k_probes/compute_heldout_pairwise_acc.py`

**HOO acc (L23, real choices):** mean=0.561 ± 0.124 across 12 folds
Script: `scripts/gemma2_10k_probes/compute_hoo_pairwise_acc.py`

Note: "other" (10 measurements) and "sensitive_creative" (43 measurements) folds are
unreliable due to tiny sample sizes — same issue affects Gemma-3.

| Topic | n_measurements | Gemma-2 acc | Gemma-3 acc |
|-------|---------------|-------------|-------------|
| math | 32207 | 0.503 | 0.571 |
| knowledge_qa | 22853 | 0.629 | 0.726 |
| content_generation | 8631 | 0.646 | 0.728 |
| harmful_request | 6575 | 0.576 | 0.648 |
| fiction | 1741 | 0.636 | 0.690 |
| coding | 659 | 0.650 | 0.788 |
| model_manipulation | 412 | 0.670 | 0.648 |
| persuasive_writing | 385 | 0.613 | 0.761 |
| security_legal | 334 | 0.608 | 0.662 |
| sensitive_creative | 43 | 0.465 | 0.535 |
| summarization | 30 | 0.533 | 0.967 |
| other | 10 | 0.200 | 0.700 |
