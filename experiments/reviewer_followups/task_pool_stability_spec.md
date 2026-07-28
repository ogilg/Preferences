# Leave-one-dataset-out task-pool stability

## Objective

Test whether removing one source dataset changes:

1. Thurstonian utilities for the remaining tasks;
2. the L32 end-of-turn ridge-probe direction.

This is offline analysis of existing comparisons and activations. Do not make
API calls or run model inference.

## Inputs

Use exactly:

```text
TRAIN_RUN = results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_train_task_ids
EVAL_RUN = results/experiments/persona_sweep_final_six/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_eval_task_ids
ACTIVATIONS = activations/gemma-3-27b_it/pref_layer_sweep/activations_eot_L23_L32.npz
```

Join data by `task_id`; never rely on row order. Use `layer_32` from
`ACTIVATIONS`.

From each run directory, require exactly one `thurstonian_*.csv`. Use:

- `TRAIN_RUN/measurements.yaml` for all refits and train origins;
- `EVAL_RUN/measurements.yaml` only to recover eval origins;
- the existing train CSV as the baseline utility target;
- the existing eval CSV only for ridge-alpha selection.

Dataset labels are the `origin_a` and `origin_b` values in
`measurements.yaml`. The five labels are:

```text
WILDCHAT
ALPACA
MATH
BAILBENCH
STRESS_TEST
```

## Conditions

Fit six probes:

1. baseline: no dataset omitted;
2. omit `WILDCHAT`;
3. omit `ALPACA`;
4. omit `MATH`;
5. omit `BAILBENCH`;
6. omit `STRESS_TEST`.

For an omission condition:

1. Remove every train task with the omitted origin.
2. Remove every comparison with either endpoint from that origin.
3. Remove retained tasks with zero remaining comparisons.
4. Assert that the remaining undirected comparison graph is connected.
5. Sort the remaining tasks lexicographically by `task_id`.
6. Fit utilities on the remaining tasks and comparisons.

Use the current repository implementations without modification:

```python
PairwiseData.from_comparisons(...)
fit_thurstonian(...)  # default arguments
```

Do not clean, resample, deduplicate, reweight, or reinterpret comparison
records. In particular, preserve the current `PairwiseData.from_comparisons`
handling of every stored choice, including `refusal`, so the ablations match the
existing baseline target.

Use the existing train CSV for baseline utilities; do not refit the unchanged
baseline graph.

## Probe fitting

For each condition:

1. Align L32 activations and train utilities by task ID.
2. Fit `StandardScaler()` on that condition's training activations.
3. Fit `Ridge()` with an intercept for each alpha in:

   ```python
   np.logspace(-1, 5, 10)
   ```

4. For an omission condition, exclude the omitted origin from eval. Do not
   refit eval utilities.
5. Transform eval activations with the condition's training scaler.
6. Select the alpha with highest eval Pearson correlation. Break an exact tie
   in favour of the smaller alpha.
7. Fit the selected ridge model on the condition's training data only. Do not
   add eval tasks to training.
8. Convert the fitted direction back to raw activation coordinates:

   ```python
   w_raw = ridge.coef_ / scaler.scale_
   ```

Do not use `train_ridge_heldout` or `_train_ridge_probe_heldout`; those helpers
split eval internally and implement a different protocol.

## Metrics

For each omission condition, compute:

```text
utility_r
```

Pearson correlation between its refitted utilities and the existing baseline
train utilities on exactly the retained task IDs.

```text
probe_cosine
```

Signed cosine similarity between its `w_raw` and the baseline `w_raw`. Exclude
the intercept and do not flip either direction's sign.

## Required checks

Fail instead of continuing if a fit does not converge or an input/count check
fails.

Expected pre-fit counts:

| Condition omitted | Train tasks | Stored comparisons | Zero-degree tasks removed | Eval tasks |
|---|---:|---:|---:|---:|
| none | 4,000 | 37,196 | 0 | 1,000 |
| `ALPACA` | 3,000 | 20,852 | 1 | 750 |
| `BAILBENCH` | 3,600 | 30,656 | 0 | 899 |
| `MATH` | 2,996 | 21,514 | 4 | 750 |
| `STRESS_TEST` | 3,398 | 27,081 | 0 | 850 |
| `WILDCHAT` | 3,001 | 21,072 | 0 | 751 |

`Stored comparisons` means retained rows of `measurements.yaml`, including
repeated task pairs.

Also assert:

- the baseline utility CSV has 4,000 unique task IDs;
- the eval utility CSV has 1,000 unique task IDs;
- `ACTIVATIONS` has 6,000 unique task IDs and `layer_32` has shape
  `(6000, 5376)`;
- every train and eval task used by a probe has an L32 activation;
- all five ablated comparison graphs are connected after zero-degree removal.

## Outputs

Write under:

```text
experiments/reviewer_followups/task_pool_stability/
```

Required files:

```text
run.py
results.csv
results.json
report.md
```

The canonical command is:

```bash
.venv/bin/python experiments/reviewer_followups/task_pool_stability/run.py
```

`results.csv` must contain one row per condition and at least:

```text
condition
omitted_dataset
n_train_tasks
n_train_comparisons
n_zero_degree_removed
n_eval_tasks
converged
best_alpha
eval_r
utility_r
probe_cosine
```

For the baseline row, set `utility_r = 1` and `probe_cosine = 1`.

`report.md` must contain the six-row results table and a one-paragraph
rebuttal-ready summary. State the minimum ablated `utility_r` and minimum
ablated `probe_cosine`. Do not create a plot unless requested.
