# Multi-Role Ablation: Running Log

## Setup (Phase 1)

**Task sampling**:
- 1500 tasks sampled from `activations/gemma_3_27b/` (29,996 total)
- Stratified: 300 tasks per origin (wildchat, alpaca, math, bailbench, stress_test)
- Seed: 42
- Train: 1000 tasks, Eval: 500 tasks
- Saved to: `experiments/probe_generalization/multi_role_ablation/task_ids_{all,train,eval}.txt`
- Exclude file (28,496 tasks NOT in 1500): `configs/measurement/active_learning/exclude_mra_non_target.txt`

**Configs created**:
- Measurement: `configs/measurement/active_learning/mra_persona{1-4}_*.yaml`
- Extraction: `configs/extraction/mra_persona{2-4}_*.yaml`
- Modified `src/probes/extraction/config.py` and `extract.py` to support `system_prompt` and `task_ids_file`

## Phase 2: Preference Measurement

**Bug 1 — Cache key missing system_prompt**: `RevealedCache._make_key()` did not include system_prompt. All 4 personas would share identical measurements. Fixed: append `sp{sha256[:8]}` to key. Processes stopped before data written.

**Bug 2 — Race condition on cache writes**: All 4 processes shared the SAME temp file `gemma-3-27b.yaml.tmp`, causing interleaved YAML writes and FileNotFoundError on rename. Fixed: modified `RevealedCache.__init__()` to use per-persona filenames (`gemma-3-27b-sp{hash8}.yaml`) when system_prompt is set.

**First attempt** (2026-02-25 05:34 UTC): Personas 2 and 3 crashed ~2h46m in due to race condition. Personas 1 and 4 killed preemptively. Corrupted cache moved to backup.

**Second attempt completed** (2026-02-25 ~10:18 UTC):
- All 4 personas converged in 6 iterations, ~43,600 comparisons each
- Convergence: 0.97 pair agreement, 0.990+ rank correlation
- Per-persona cache files: `gemma-3-27b[-sp{hash}].yaml`
- BUT: These measurements were INVALID (wrong task set — see Bug 3 below)

**Bug 3 — Exclude file covered wrong task universe**: Phase 1 generated exclude file from `activations/gemma_3_27b/` only (29,996 tasks), leaving 47,638 tasks available. Measurement sampled 1500 from those, only ~105 overlapping with the MRA 1500-task set. Fixed: regenerated exclude file with 74,634 entries (all 76,134 tasks minus the 1500 MRA tasks).

**Third attempt started** (2026-02-25 12:27 UTC):
- Invalid measurement dirs moved to `results/experiments/_invalid_mra_backup/`
- Processes 14071 (p1), 14143 (p2), 14278 (p3), 14351 (p4) running
- Exclude file: 74,634 entries confirmed (only 1500 MRA tasks available)
- 186 active HTTPS connections as of ~12:29 UTC

## Phase 3: Activation Extraction

**Completed** (2026-02-25):
- Persona 2 (villain): `activations/gemma_3_27b_villain/activations_prompt_last.npz` → shape (1500, 5376)
- Persona 3 (midwest): `activations/gemma_3_27b_midwest/activations_prompt_last.npz` → shape (1500, 5376)
- Persona 4 (aesthete): `activations/gemma_3_27b_aesthete/activations_prompt_last.npz` → shape (1500, 5376)
- Persona 1 (no prompt): using existing `activations/gemma_3_27b/` (29,996 tasks → 1500 overlap with MRA set confirmed)
- Layer 31 (= 0.5 × 62 layers), selector: prompt_last, gemma-3-27b

## Phase 4: Probe Training & Evaluation (Pilot — Invalid Data)

**Pilot run with bug 3 data** (n=78 per persona instead of 1000):
- All 15 conditions trained; Pearson r ranged 0.763–0.933
- Results not reportable (wrong task set; ~105/1500 overlap with MRA activations)
- Utility correlations were 0.942–0.944 across personas (strong signal)
- Full re-run pending valid measurements from Phase 2 third attempt

## Phase 4: Probe Training & Evaluation (Main)

**Completed** (2026-02-25 ~16:02 UTC):
- All 15 conditions trained, N=1000 per persona ✓
- Single-persona same/cross r: no_prompt 0.875/0.773, villain 0.876/0.814, midwest 0.892/0.811, aesthete 0.893/0.831
- All-persona probe: mean r=0.882 across 4 eval sets
- Utility correlations: 0.935–0.943 (strong shared signal, meaningful variation)
- Results saved: `experiments/probe_generalization/multi_role_ablation/probe_results.json`

## Phase 5: Analysis

## Phase 5: Analysis

**Completed** (2026-02-25 ~16:05 UTC):
- 4 plots saved to `experiments/probe_generalization/multi_role_ablation/assets/`
- Generalization matrix, scaling curve, cosine similarity matrix, utility correlation matrix
- Cross-persona scaling: 1 persona cross r=0.807 → 2=0.834 → 3=0.839 → 4=0.882
