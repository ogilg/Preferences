# OOD System Prompts with EOT Probes

## Goal

Re-run OOD experiments 1a–1d using probes trained on `<end_of_turn>` activations instead of `prompt_last`. The EOT probes generalise much better across topics (1.8% gap vs 8.8%), so the question is whether this also improves OOD tracking of artificially induced preferences.

## Background

The original OOD experiments (report at `experiments/ood_system_prompts/ood_system_prompts_report.md`) showed prompt_last probes track system-prompt-induced preference shifts. But all activations were extracted at `prompt_last`. To use EOT probes, we need EOT activations under each system prompt condition.

Original results (prompt_last, L31):

| Experiment | Overall r | On-target r |
|------------|----------|-------------|
| 1a: Category preference | 0.61 | 0.90 |
| 1b: Hidden preference | 0.65 | 0.95 |
| 1c: Crossed preference | 0.66 | 0.86 |
| 1d: Competing preference | 0.78 | 0.88 |

## Methodology

This must be an exact replication of the original OOD experiments — same behavioral data, same system prompts, same tasks, same analysis pipeline, same ground truth labels. The ONLY thing that changes is the token position (eot instead of prompt_last) and the probe (trained on EOT activations). Any difference in results is attributable solely to the token position.

## What we reuse

- **Behavioral data**: `results/ood/{category,hidden,crossed,competing}_preference/pairwise.json`. No re-measurement needed.
- **System prompts and tasks**: AL configs in `configs/measurement/active_learning/ood_exp1{a,b,c,d}/`.
- **Analysis code**: `src/ood/analysis.py` and `scripts/ood_system_prompts/analyze_ood.py`. Need small modifications (see below).
- **EOT probe**: `results/probes/heldout_eval_gemma3_eot/probes/probe_ridge_L31.npy` (trained on 10k, evaluated on 4k heldout, r=0.868).

## Code changes

### 1. `src/ood/analysis.py` — parameterise activation filename

`compute_deltas()` hardcodes `activations_prompt_last.npz` on lines 103 and 114. Add an `activations_filename` parameter:

```python
def compute_deltas(
    rates: dict[str, dict[str, float]],
    activations_dir: Path,
    probe_path: Path,
    layer: int,
    baseline_activations_key: str = "baseline",
    activations_filename: str = "activations_prompt_last.npz",  # NEW
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
    baseline_npz = activations_dir / baseline_activations_key / activations_filename
    ...
        cond_npz = activations_dir / cid / activations_filename
```

This is backwards-compatible — existing callers don't pass `activations_filename` and get the old default.

### 2. `scripts/ood_system_prompts/analyze_ood.py` — create EOT variant

The existing script (`scripts/ood_system_prompts/analyze_ood.py`) has the full per-experiment analysis logic: rate computation, condition filtering, on-target/off-target splits for 1d, etc. Create a copy at `scripts/ood_eot/analyze_eot.py` with three changes:

```python
ACTS_DIR = REPO_ROOT / "activations" / "ood_eot"          # was "ood"
PROBE_DIR = REPO_ROOT / "results" / "probes" / "heldout_eval_gemma3_eot" / "probes"  # was "gemma3_10k_heldout_std_demean"
LAYERS = [31]                                               # was [31, 43, 55]
```

And pass `activations_filename="activations_eot.npz"` to every `compute_deltas()` call.

Everything else — `compute_p_choose_from_pairwise()`, condition filtering logic (targeted vs competing, hidden_ vs crossed_ task ID prefixes), `correlate_deltas()`, `per_condition_correlations()` — stays identical.

### 3. `scripts/run_all_extractions.py` — add EOT extraction function

Add `run_ood_eot_extractions()` alongside the existing `run_ood_extractions()`:

```python
OOD_EOT_LAYERS = [31]
OOD_EOT_SELECTORS = ["eot"]

OOD_EOT_EXPERIMENTS = [
    ("configs/measurement/active_learning/ood_exp1a", "activations/ood_eot/exp1_category"),
    ("configs/measurement/active_learning/ood_exp1b", "activations/ood_eot/exp1_prompts"),
    ("configs/measurement/active_learning/ood_exp1c", "activations/ood_eot/exp1_prompts"),
    ("configs/measurement/active_learning/ood_exp1d", "activations/ood_eot/exp1_prompts"),
]
```

Same logic as `run_ood_extractions()` — iterates AL configs, reads `measurement_system_prompt` / `custom_tasks_file` / `include_task_ids_file` from each, builds an `ExtractionConfig` with `selectors=OOD_EOT_SELECTORS`, `layers_to_extract=OOD_EOT_LAYERS`. Uses `--resume` to skip existing.

Note: 1b, 1c, and 1d share the same activation output root (`exp1_prompts`) because their condition IDs don't overlap (targeted_* vs crossed_* vs compete_*).

## Extraction

| Exp | Conditions | Tasks/condition | Forward passes |
|-----|-----------|----------------|---------------|
| 1a | 13 (12 persona + baseline) | 50 (standard pool via `include_task_ids_file`) | 650 |
| 1b | 17 (16 targeted + baseline) | 48 (custom tasks via `custom_tasks_file`) | 816 |
| 1c | 17 (16 targeted + baseline) | 48 (crossed tasks via `custom_tasks_file`) | 816 |
| 1d | 17 (16 competing + baseline) | 48 (crossed tasks via `custom_tasks_file`) | 816 |
| **Total** | | | **3,098** |

~20 min on A100. Single model load, selector `eot`, layer 31 only.

Run command:
```bash
python -c "from scripts.run_all_extractions import run_ood_eot_extractions; run_ood_eot_extractions()"
```

## Storage layout

```
activations/ood_eot/
  exp1_category/
    baseline/activations_eot.npz           # 50 tasks, layer 31
    coding_neg_persona/activations_eot.npz
    coding_pos_persona/activations_eot.npz
    ...  (13 condition dirs)
  exp1_prompts/
    baseline/activations_eot.npz           # 48 tasks, layer 31
    cheese_neg_persona/activations_eot.npz
    cheese_pos_persona/activations_eot.npz
    ...                                    # 16 targeted (1b)
    crossed_cheese_math/...                # shared tasks, 1c conditions overlap
    compete_cheese_math_topicpos/activations_eot.npz
    compete_cheese_math_shellpos/activations_eot.npz
    ...                                    # 16 competing (1d)
```

## Scoring and analysis

The analysis script scores activations with the probe via `_score_activations()` in `src/ood/analysis.py`:

```python
def _score_activations(npz_path, layer, weights, bias):
    data = np.load(npz_path, allow_pickle=True)
    acts = data[f"layer_{layer}"]       # (n_tasks, d_model) at layer 31
    scores = acts @ weights + bias       # linear probe: dot product + bias
    task_ids = list(data["task_ids"])
    return {tid: float(s) for tid, s in zip(task_ids, scores)}
```

The probe is loaded from `.npy` and split into (weights, bias):
```python
probe = np.load("results/probes/heldout_eval_gemma3_eot/probes/probe_ridge_L31.npy")
weights, bias = probe[:-1], float(probe[-1])  # (d_model,), scalar
```

This is fully compatible with EOT activations — the npz format is identical (`layer_31` key, `task_ids` key), just extracted at a different token position. The probe weights were trained on EOT activations so the dimensions match.

For each experiment, the analysis:
1. Loads `pairwise.json` → `compute_p_choose_from_pairwise()` → `{condition_id: {task_id: p_choose}}`
2. Filters conditions/tasks per experiment (e.g., 1b: only `hidden_*` tasks, no `compete_*` conditions)
3. Calls `compute_deltas(rates, acts_dir, probe_path, layer=31, activations_filename="activations_eot.npz")`
   - Loads baseline activations → scores with probe → baseline scores
   - For each condition: loads condition activations → scores → computes `probe_delta = cond_score - baseline_score` per task
   - Also computes `beh_delta = cond_rate - baseline_rate` per task
4. `correlate_deltas(beh, probe)` → Pearson r, Spearman r, sign agreement, permutation p

Run command:
```bash
python scripts/ood_eot/analyze_eot.py --exp all --output experiments/ood_eot/analysis_results.json
```

## Deliverable

Report at `experiments/ood_eot/ood_eot_report.md` with:
- Side-by-side comparison table: prompt_last vs EOT probe correlations for all four experiments
- Scatter plots: behavioral delta vs probe delta (one per experiment)
- Whether the EOT probe's better cross-topic generalisation translates to better OOD tracking
