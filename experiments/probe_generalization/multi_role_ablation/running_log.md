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

**Critical bug found and fixed**: `RevealedCache._make_key()` did not include system_prompt in cache key. All 4 personas would have shared identical measurements. Fixed by appending `sp{sha256[:8]}` to cache key when system_prompt is not None. Processes were stopped before any data was written.

**Bug fixed**: `MeasurementCache` and `runners.py` updated to propagate `system_prompt`.

**Measurement runs started** (2026-02-25 05:34 UTC):
- Process 7068: mra_persona1_noprompt (no system prompt, baseline)
- Process 7140: mra_persona2_villain (Mortivex villain)
- Process 7276: mra_persona3_midwest (Midwest pragmatist)
- Process 7413: mra_persona4_aesthete (Celestine aesthete)
- Each: max_concurrent=30, active_learning (initial_degree=5, 1500 tasks → ~3750 pairs round 1)
- 253 active HTTPS connections to OpenRouter as of 05:41 UTC (7 min in)
- Cache file `results/cache/revealed/gemma-3-27b.yaml` not yet written (first batch in progress)

## Phase 3: Activation Extraction

**Completed** (2026-02-25):
- Persona 2 (villain): `activations/gemma_3_27b_villain/activations_prompt_last.npz` → shape (1500, 5376)
- Persona 3 (midwest): `activations/gemma_3_27b_midwest/activations_prompt_last.npz` → shape (1500, 5376)
- Persona 4 (aesthete): `activations/gemma_3_27b_aesthete/activations_prompt_last.npz` → shape (1500, 5376)
- Persona 1 (no prompt): using existing `activations/gemma_3_27b/` (29,996 tasks → 1500 overlap with MRA set confirmed)
- Layer 31 (= 0.5 × 62 layers), selector: prompt_last, gemma-3-27b

## Phase 4: Probe Training & Evaluation

[To be filled in as probes are trained]

## Phase 5: Analysis

[To be filled in as analysis runs]
