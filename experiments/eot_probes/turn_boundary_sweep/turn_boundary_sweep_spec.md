# Turn Boundary Token Sweep — Probe Comparison

## Goal

Compare probe performance across all 5 turn boundary tokens in Gemma-3-27B's chat template suffix, plus `task_mean` (mean over user content tokens) as a baseline. Prior work trained probes on `turn_boundary:-1` (the `\n` before generation, formerly `prompt_last`) and `turn_boundary:-5` (the `<end_of_turn>` token, formerly `eot`). This experiment fills in the 3 missing positions and runs all 6 at the same layers for a clean comparison.

For Gemma-3 IT, the 5 tokens before generation are:
```
<end_of_turn>\n<start_of_turn>model\n
     -5       -4      -3       -2  -1
```

## Motivation

Patching experiments showed `<end_of_turn>` carries strong decision signal, and probes trained on it match or exceed those at `\n` (turn_boundary:-1). But we haven't systematically compared all positions in the turn boundary. The intermediate tokens (`\n`, `<start_of_turn>`, `model`) might accumulate or relay signal differently across layers.

## Prerequisites

### Model weights
Phase 1 loads `google/gemma-3-27b-it` via HuggingFace. Ensure `HF_TOKEN` is set in `.env` and there is ~54GB free disk for the model cache.

### Data already in git
All required measurement data (Thurstonian scores in `results/experiments/main_probes/`) and topic labels (`data/topics/topics.json`) are tracked in the git repo. No data sync is needed beyond the repo itself.

### Phase ordering
Phases must run sequentially: Phase 1 produces the activation `.npz` files that Phase 2 reads. Phase 3 reads Phase 2 output.

## Design

### Phase 1: Extraction (GPU)

Single forward pass extracting all 6 selectors at 5 layers spanning the causal window and beyond.

- **Config**: `configs/extraction/gemma3_27b_turn_boundary_sweep.yaml`
- **Selectors**: `turn_boundary:-1` through `turn_boundary:-5`, plus `task_mean`
- **Layers**: 25, 32, 39, 46, 53 (every 7 layers from 25 to ~55)
- **Tasks**: 30k, same seed/origins as all other extractions
- **Output**: `activations/gemma_3_27b_turn_boundary_sweep/activations_{selector}.npz` (6 files)

```bash
python -m src.probes.extraction.run configs/extraction/gemma3_27b_turn_boundary_sweep.yaml --resume
```

After extraction completes, sync activations back to local machine:
```bash
rsync -avz -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" root@<IP>:/workspace/repo/activations/gemma_3_27b_turn_boundary_sweep/ activations/gemma_3_27b_turn_boundary_sweep/
```

### Phase 2: Probe Training (CPU)

10 probe configs: 5 selectors × 2 eval types.

**Heldout eval** (train on 10k, sweep alpha on half of 4k eval, test on other half):
```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/heldout_eval_gemma3_tb-1.yaml
python -m src.probes.experiments.run_dir_probes --config configs/probes/heldout_eval_gemma3_tb-2.yaml
python -m src.probes.experiments.run_dir_probes --config configs/probes/heldout_eval_gemma3_tb-3.yaml
python -m src.probes.experiments.run_dir_probes --config configs/probes/heldout_eval_gemma3_tb-4.yaml
python -m src.probes.experiments.run_dir_probes --config configs/probes/heldout_eval_gemma3_tb-5.yaml
```

**Hold-one-out by topic** (leave one topic out, train on rest, repeat):
```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/gemma3_10k_hoo_topic_tb-1.yaml
python -m src.probes.experiments.run_dir_probes --config configs/probes/gemma3_10k_hoo_topic_tb-2.yaml
python -m src.probes.experiments.run_dir_probes --config configs/probes/gemma3_10k_hoo_topic_tb-3.yaml
python -m src.probes.experiments.run_dir_probes --config configs/probes/gemma3_10k_hoo_topic_tb-4.yaml
python -m src.probes.experiments.run_dir_probes --config configs/probes/gemma3_10k_hoo_topic_tb-5.yaml
```

### Phase 3: Analysis

Load results from all 10 probe runs and produce a comparison.

**Result format**: Each heldout run produces `manifest.json` with per-layer entries containing `final_r` (Pearson r on held-out test set) and `final_acc` (pairwise accuracy). Each HOO run produces `hoo_summary.json` with per-fold and mean R².

**Analysis steps**:

1. Load `results/probes/heldout_eval_gemma3_tb-{1..5}/manifest.json` — extract `final_r` per layer from each probe entry
2. Load `results/probes/gemma3_10k_hoo_topic_tb-{1..5}/hoo_summary.json` — extract mean cross-topic r per layer
3. Build a 5×5 table (token position × layer) for both metrics
4. Plot a grouped bar chart or line plot: x-axis = layer, one line/bar group per token position, y-axis = Pearson r. Y-axis anchored at 0. One plot for heldout r, one for HOO mean r.
5. Save plots to `experiments/eot_probes/turn_boundary_sweep/assets/` with naming convention `plot_{mmddYY}_description.png`

**Key questions to address**:

1. **Which token position gives the best probe?** Does `<end_of_turn>` (-5) still win, or does signal accumulate at later positions?
2. **How does performance vary across layers?** Does the optimal token position shift with depth?
3. **Do intermediate tokens (-2, -3, -4) carry signal?** If `<start_of_turn>` or `model` tokens are informative, it suggests the model is actively relaying preference signal through the turn boundary, not just encoding it at `<end_of_turn>`.

Write findings to `experiments/eot_probes/turn_boundary_sweep/turn_boundary_sweep_report.md` with the plots embedded via relative paths. Use Pearson r as the primary metric (not R²). Present a summary table and the plots, then discuss the three questions above.

## Commit Guidance

- **Do NOT commit** activation `.npz` files (gitignored, multi-GB) or `.npy` probe weight files
- **DO commit** the analysis script, report markdown, and plots in `assets/`
- **DO commit** probe result JSONs (`manifest.json`, `hoo_summary.json`) — these are small summary files

## Done When

1. All 5 activation files exist in `activations/gemma_3_27b_turn_boundary_sweep/` locally (synced back from pod)
2. All 10 probe result directories are populated in `results/probes/` locally (synced back from pod)
3. `experiments/eot_probes/turn_boundary_sweep/turn_boundary_sweep_report.md` exists with comparison plots and addresses the three key questions

## Output

- `activations/gemma_3_27b_turn_boundary_sweep/` — 5 `.npz` files (gitignored)
- `results/probes/heldout_eval_gemma3_tb-{1..5}/` — heldout eval results
- `results/probes/gemma3_10k_hoo_topic_tb-{1..5}/` — HOO results
- `experiments/eot_probes/turn_boundary_sweep/assets/` — comparison plots
- `experiments/eot_probes/turn_boundary_sweep/turn_boundary_sweep_report.md` — write-up
