# Running Log: Gemma-2 10k Probe Training

## Setup
- Branch: research-loop/gemma2_10k_probes (at main HEAD)
- Activations: ~/Dev/MATS/Preferences/activations/gemma_2_27b_base/activations_prompt_last.npz — 30k tasks, 100% overlap with 10k train and 4k eval
- Copied activations to activations/gemma_2_27b_base/activations_prompt_last.npz
- Copied topics.json from Preferences-role-prompting repo
- Created configs: gemma2_10k_heldout_std_raw.yaml, gemma2_10k_heldout_std_demean.yaml, gemma2_10k_hoo_dataset.yaml
- Note: topics.json (v1) only covers 1475/10k train tasks → demeaned heldout and topic HOO not runnable without full topics file

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

## Dataset HOO (Step 3)
- Config: configs/probes/gemma2_10k_hoo_dataset.yaml
- 5 folds (one per dataset origin: alpaca, bailbench, competition_math, stresstest, wildchat)
- Very low cross-dataset generalization (HOO r ≈ 0.35 at L23)
- Math (competition_math) is the worst holdout (r=0.167 at L23)

| Layer | Val r | HOO r (mean) | Gap |
|-------|-------|--------------|-----|
| L11 | 0.756 | 0.312 | 0.444 |
| L23 | 0.797 | 0.353 | 0.444 |
| L27 | 0.784 | 0.323 | 0.461 |
| L32 | 0.771 | 0.274 | 0.497 |
| L36 | 0.768 | 0.283 | 0.485 |
| L41 | 0.770 | 0.300 | 0.471 |
