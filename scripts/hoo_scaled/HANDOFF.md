# Handoff Notes for Scaled HOO Experiment

## Status
- Ridge raw: DONE (results/probes/hoo_scaled_raw/hoo_summary.json)
- Ridge demeaned: DONE (results/probes/hoo_scaled_demeaned/hoo_summary.json)
- ST baseline: DONE (results/probes/hoo_scaled_st_baseline/hoo_summary.json)
- BT raw: DONE (results/probes/hoo_scaled_bt/hoo_summary.json)

All experiments complete. Report and log updated.

## Key results so far
- Ridge raw L31: mean hoo_r = 0.779 (best)
- Ridge demeaned L31: mean hoo_r = 0.706, gap = 0.009
- ST baseline: mean hoo_r = 0.245
- Ridge raw vs ST: paired t = 72.9, p < 10^-50

## File locations
- Configs: configs/probes/hoo_scaled_*.yaml (4 files)
- Scripts: scripts/hoo_scaled/ (analyze.py, extract_st_embeddings.py)
- Report: experiments/ood_generalization/hoo_scaled/report.md
- Plots: experiments/ood_generalization/hoo_scaled/assets/
- Research log: docs/logs/research_loop_hoo_scaled.md
