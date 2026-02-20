# Cross-topic generalisation: Gemma-2 27B base

Do probes trained on Gemma-2 27B base (`google/gemma-2-27b`) activations generalise across topics? Run the same held-one-out (HOO) cross-topic evaluation we did for Gemma-3 27B IT, and compare.

## Why this matters

Gemma-2 base probes achieve R² = 0.789 on standard (in-distribution) probe training. But if this signal is content-driven rather than evaluative, it should generalise poorly to held-out topics. The Gemma-3 IT HOO baseline (R² = 0.78, 56/56 folds beat content baseline) is the bar to clear.

## Branch

Branch from `research-loop/hoo_scaled` — it already has all the HOO configs, scripts, Gemma-3 comparison results, and sentence-transformer baseline. Cherry-pick from `research-loop/gemma2-base-probes`:
- `configs/extraction/gemma2_27b_base_prompt_last.yaml`
- The model registry entry in `src/models/registry.py` — key `gemma-2-27b-base`, mapping to `hf_name: google/gemma-2-27b`. If the cherry-pick conflicts, just add it manually next to the existing `gemma-2-27b` entry.

## Method

### Step 1: Extract activations

Re-extract Gemma-2 27B base activations — these were lost when the previous pod terminated. Use the existing extraction pipeline with `configs/extraction/gemma2_27b_base_prompt_last.yaml`.

```
python -m src.probes.extraction.run configs/extraction/gemma2_27b_base_prompt_last.yaml
```

**Save the activations before the pod terminates.** They are gitignored (`*.npz`) and will be lost otherwise. Post a Slack message with the SCP command so the user can pull them to local:
```
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/repo/activations/gemma_2_27b_base/activations_prompt_last.npz activations/gemma_2_27b_base/
```

### Step 2: Run HOO cross-topic evaluation

Use the existing HOO pipeline: `python -m src.probes.experiments.run_dir_probes --config <config>`.

Create configs modelled on `configs/probes/hoo_scaled_raw.yaml`, changing only `activations_path`, `layers`, `output_dir`, and `experiment_name`. Key parameters to match:
- `hoo_grouping: topic`
- `hoo_hold_out_size: 3`
- `hoo_groups: [math, fiction, coding, persuasive_writing, content_generation, summarization, knowledge_qa, harmful_request]`
- `layers: [11, 23, 27, 32, 36, 41]` — these are the Gemma-2 27B layers at fractional depths [0.25, 0.5, 0.6, 0.7, 0.8, 0.9] of 46 total. Do NOT use Gemma-3's layer numbers (15, 31, 43, 55 — those are for a 62-layer model).
- `standardize: true`

Run both:
- **Raw** (no demeaning)
- **Demeaned** (topic-demeaned scores, modelled on `hoo_scaled_demeaned.yaml`)

**Do not rerun Gemma-3 or sentence-transformer experiments.** Those results already exist in `results/probes/hoo_scaled_raw/`, `results/probes/hoo_scaled_demeaned/`, and `results/probes/hoo_scaled_st_baseline/`. Just load them for comparison numbers.

### Step 3: Report results

The main deliverable is the Gemma-2 base HOO numbers. Report them alongside the existing Gemma-3 and ST baselines for context, but don't go deep into the comparison — just present the Gemma-2 results clearly.

## Existing code — use it, don't rewrite it

- **Extraction**: `src/probes/extraction/run.py`
- **HOO pipeline**: `src/probes/experiments/run_dir_probes.py` with `hoo_grouping`, `hoo_hold_out_size`, `hoo_groups`
- **HOO Ridge**: `src/probes/experiments/hoo_ridge.py`
- **Topic grouping**: `src/probes/experiments/residualization.py` — `build_task_groups()`
- **Plotting**: `src/probes/experiments/run_dir_probes.py` — `plot_hoo_summary()`

## Data

- Preference scores: `results/experiments/gemma3_3k_run2/`
- Topic labels: `src/analysis/topic_classification/output/topics_v2.json`
- Gemma-3 HOO results for comparison: `results/probes/hoo_scaled_raw/`, `results/probes/hoo_scaled_demeaned/`
- ST baseline: `results/probes/hoo_scaled_st_baseline/`

## Sample overlap

A previous Gemma-2 extraction had only 2264/3000 task overlap because `stress_test.jsonl` was missing from the pod. This is now fixed (committed to git). Verify after extraction that the overlap with the 3000 preference tasks is close to 3000. If stress_test tasks are still missing, debug before proceeding — incomplete overlap confounds the comparison.
