# Exp 3C Anti Running Log

## Session start: 2026-02-21

### Initial setup
- Branch: research-loop/exp3c_anti
- Pod: A100 80GB PCIe
- IS_SANDBOX=1

### Missing data found
- `activations/gemma_3_27b/activations_prompt_last.npz` — MISSING (main activations for 14k training tasks)
- `activations/ood/exp3_minimal_pairs/` — MISSING (all OOD activations: baseline, A/B/C conditions)
- `results/probes/gemma3_10k_heldout_std_demean/probes/` — MISSING (only manifest.json present, probe .npy files absent)

### Available data
- Behavioral data: `results/ood/minimal_pairs_v7/behavioral.json` ✓
- Config: `configs/ood/prompts/minimal_pairs_v7.json` (40 A/C conditions per role, 40 C conditions total) ✓
- Thurstonian scores: `results/experiments/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_80fa9dc8.csv` ✓
- HF_TOKEN available ✓
- GPU: A100 80GB ✓

### Plan
1. Download gemma-3-27b using HF_TOKEN
2. Extract main activations (14k tasks from Thurstonian run)
3. Train probe on main activations
4. Extract OOD C conditions (20 × 50 tasks) + baseline + A/B conditions
5. Run exp3c analysis
6. Generate plots and write report

### Model download: complete
- gemma-3-27b-it downloaded to /opt/hf_cache/hub/
- 12/12 shards, ~52GB total

### Main activation extraction: first attempt OOM
- Started with batch_size=32, 440 batches (14049 tasks / 32)
- Ran ~9 min (batches 1-162 of 440), then crashed with CUDA OOM
- Error: lm_head allocation of 14.91 GiB, only 12.25 GiB free
- Root cause: long sequences (avg ~890 tokens) cause lm_head output to be 32 × 890 × 262144 × 2 = 14.9 GiB
- Fix: batch_size=16 (halves lm_head output) + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Main activation extraction: second attempt — complete
- batch_size=16, 879 batches, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- Completed in 17:33 with no OOM errors
- Saved to activations/gemma_3_27b/activations_prompt_last.npz (14049 tasks)

### Probe training — complete
- Layers 31, 43, 55 (only these extracted, used gemma3_exp3c_layers.yaml config)
- L31: sweep_r=0.7278, final_r=0.7207 (matches original: 0.7608)
- L43: final_r=0.7286 (matches original: 0.7285)
- L55: final_r=0.7207 (matches original: 0.7206)
- Best alpha 4642 (matches original manifest 4641.59) — confirms reproducibility
- Issues: matplotlib missing (installed via python -m pip), measurements.yaml missing (created empty files)

### OOD extraction (exp3_all: A+B+C conditions) — complete
- All 60 conditions extracted (40 A+B + 20 C) x 50 tasks at L31/43/55
- Ran sequentially in single model load (~6 min per 60 conditions)

### Analysis — complete
Layer 31 key results:
- Overall correlation: A r=0.513, B r=0.524, C r=0.517 (all perm_p=0.000)
- Target task rank (desc, 1=highest): A=6.8, B=11.0, C=21.6
- Top-5 rate: A=65%, B=30%, C=15%
- Paired A/C: only 3/20 pairs have A>0 & C<0 (expected inversion almost never happens)
- Mean probe delta: A=5.14, C=2.32 (C ~45% of A, not negative)
- Key finding: C conditions reduce probe activation for target task but don't invert it
- Mean probe delta B=4.26 (added in review pass)

### Report + review — complete
- Report written to exp3c_anti_report.md; plots to assets/
- Subagent review: added B delta to paired table, clarified hit-rate metric comparison, gave equal weight to training-data-bias vs unipolar-representation hypotheses, noted 3 inversions are marginal

### Commit and push — complete
- Committed 15 files to research-loop/exp3c_anti (c1de62b)
- Pushed to https://github.com/ogilg/Preferences/tree/research-loop/exp3c_anti
- Note: inline credential helper needed (empty helper entry in local config blocked gh auth helper)
- Not committed: manifest.json (retrained probe overwrote original; unstaged to avoid corrupting main probe record), src/analysis/topics.json (4.3 MB generated)

