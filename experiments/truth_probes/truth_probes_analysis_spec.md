# Truth Probes Analysis: Does the preference direction separate true from false?

## Question

We have preference probes trained on pairwise task-choice data. We now have activations from 9,395 CREAK claims (true/false factual statements) passed through Gemma 3 27B IT. Does the preference probe direction assign systematically different scores to true vs false claims?

## Inputs

### Activations (extracted, gitignored)

Two framings, each with 4 selectors × 5 layers:

| Framing | Path | Description |
|---------|------|-------------|
| Raw | `activations/gemma_3_27b_creak_raw/` | Claim is the user message |
| Repeat | `activations/gemma_3_27b_creak_repeat/` | `"Please say the following statement: '{claim}'"` |

Each directory contains:
- `activations_{selector}.npz` for selectors: `task_mean`, `task_last`, `turn_boundary:-2`, `turn_boundary:-5`
- `completions_with_activations.json` — list of dicts, each with `task_id` (str) and `task_prompt` (str). Row order matches the activation arrays.

Layers in each `.npz`: `[25, 32, 39, 46, 53]`.

### Labels (in git)

`src/task_data/data/creak.jsonl` — one JSON object per line: `{"ex_id": "train_0", "sentence": "...", "label": "true"/"false", "entity": "..."}`.

Join labels to activations via `ex_id == task_id` from `completions_with_activations.json`.

### Probes (heldout-trained, gitignored)

Preference probes trained on 10k pairwise-choice Thurstonian scores, heldout-evaluated:

| Probe | Activation selector | Path pattern | Layers | Heldout r (best) |
|-------|-------------------|-------------|--------|-----------------|
| tb-2 (`model` token) | `turn_boundary:-2` | `results/probes/heldout_eval_gemma3_tb-2/probes/probe_ridge_L{layer}.npy` | 25, 32, 39, 46, 53 | 0.874 (L32) |
| tb-5 (`<end_of_turn>`) | `turn_boundary:-5` | `results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L{layer}.npy` | 25, 32, 39, 46, 53 | 0.868 (L32) |

Note: tb-1 probes exist but no `turn_boundary:-1` activations were extracted for CREAK. Exclude from this analysis. No heldout probes exist for `task_mean` or `task_last`.

Format: `[coef_0, ..., coef_n, intercept]`. Applied as `activations @ weights[:-1] + weights[-1]`.

### Data sync (if running on pod)

Activations and probes are gitignored. Sync before running:

| Data | Local path | Size |
|------|-----------|------|
| CREAK activations (raw) | `activations/gemma_3_27b_creak_raw/` | ~3.8 GB |
| CREAK activations (repeat) | `activations/gemma_3_27b_creak_repeat/` | ~3.8 GB |
| Preference probes (tb-2) | `results/probes/heldout_eval_gemma3_tb-2/probes/` | ~50 MB |
| Preference probes (tb-5) | `results/probes/heldout_eval_gemma3_tb-5/probes/` | ~50 MB |

Labels are in `src/task_data/data/creak.jsonl` (tracked in git, no sync needed).

## Analysis steps

### 1. Load and score

For each (framing, probe, layer) triple:
1. Load activations from the matching `.npz` file at the given layer
2. Load probe weights from the matching `.npy` file
3. Compute probe scores using `score_with_probe` from `src.probes.core.evaluate`
4. Load ground-truth labels from `src/task_data/data/creak.jsonl`, join on `ex_id == task_id`
5. Split scores into `true_scores` and `false_scores`

Do not reimplement activation loading or probe scoring — use `load_activations` from `src.probes.core.activations` and `score_with_probe` from `src.probes.core.evaluate`.

### 2. Metrics

For each (framing, probe, layer) triple, compute:

| Metric | Description |
|--------|-------------|
| **Mean difference** | `mean(true_scores) - mean(false_scores)`. Positive = probe scores true claims higher (model "prefers" true statements). |
| **Cohen's d** | Effect size: `(mean_true - mean_false) / pooled_std`. Scale-invariant (d=0.2 small, 0.5 medium, 0.8 large). |
| **Welch's t-test p-value** | Two-sample t-test (unequal variance). With n≈4700 per group, even tiny effects will be significant — focus on d, not p. |

The hypothesis is simple: if the preference direction encodes something like "how good is this?", and the model values accuracy, then true statements should score higher than false ones along this direction.

### 3. Primary analysis table

A results table with rows = (probe, layer), columns = (metric), one table per framing.

Focus on best layer (L32) for the two probes (tb-2, tb-5). Report all layers for completeness.

### 4. Plots

**Plot 1: `plot_031126_truth_probe_score_distributions.png`**
- For the best probe (tb-2 L32), side-by-side violins for true vs false, one panel per framing.
- Shows the full distribution shape, not just means.

**Plot 2: `plot_031126_truth_effect_size_by_layer.png`**
- Cohen's d (y-axis) vs layer (x-axis), one line per probe (tb-2, tb-5).
- One panel per framing, or two framings overlaid.
- Shows whether the truth signal follows the same layer profile as the preference signal (peaks at L32).

### 5. Framing comparison

Compare raw vs repeat for the same probe and layer. Key question: does wrapping the claim in "Please say the following statement" change the truth signal? The repeat framing forces the model to commit to the statement rather than passively process it — this might amplify or suppress truth-related activation patterns.

## Interpretation guide

| Outcome | Cohen's d | Interpretation |
|---------|-----------|---------------|
| Strong signal | > 0.5 | Preference direction encodes truth-value; model "prefers" true statements |
| Weak signal | 0.1–0.5 | Small but real truth signal in preference direction |
| No signal | < 0.1 | Preference direction is orthogonal to truth; specific to task preference |
| Reversed | < -0.1 | Model "prefers" false statements (unexpected, would need investigation) |

## Implementation

Single script: `scripts/truth_probes/analyze_truth_probes.py`

Uses existing infrastructure:
- `src.probes.core.evaluate.score_with_probe` for scoring
- `src.probes.core.activations.load_activations` for loading `.npz` files
- `scipy.stats.ttest_ind` for Welch's t-test

Output:
- Print results table to stdout
- Save plots to `experiments/truth_probes/assets/`
- Save raw metrics JSON to `experiments/truth_probes/truth_probes_results.json`

## Commit policy

Do NOT commit input data (activations `.npz`, probe `.npy` files). These are pre-existing on disk.

Commit only:
- `scripts/truth_probes/analyze_truth_probes.py`
- `experiments/truth_probes/assets/*.png`
- `experiments/truth_probes/truth_probes_results.json`

## Done criteria

The experiment is complete when:
1. `scripts/truth_probes/analyze_truth_probes.py` runs without error
2. Results table is printed to stdout
3. Both plots saved to `experiments/truth_probes/assets/`
4. `experiments/truth_probes/truth_probes_results.json` is saved
