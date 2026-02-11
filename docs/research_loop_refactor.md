# Research Loop Refactoring Ideas

Identified from reviewing how research agents used `src/` across the OOD generalization, hidden preferences, and crossed preferences experiments.

## Done

### `score_with_probe` in `src/probes/core/evaluate.py`
Agents hand-rolled `activations @ weights[:-1] + weights[-1]` in 5+ scripts. Now a single function encapsulates the `[coef..., intercept]` convention. Both `evaluate_probe_on_data` and `evaluate_probe_on_template` use it internally.

## TODO

### Probe scoring: simple multi-layer scorer
Agents repeatedly wrote the same loop: load probes for layers [31, 43, 55], score activations at each layer, collect results into a dict. A helper that takes a probe directory + activations dict and returns `{layer: scores}` would eliminate the most common boilerplate in evaluation scripts.

### Activation extraction: decouple from measurement infrastructure
`src/probes/extraction/` assumes manifest directories, experiment stores, and completion stores. Agents just needed "extract activations for these tasks with this optional system prompt" — a thin wrapper around `model.get_activations_batch()` that handles message construction and npz saving. The HuggingFace model API itself is fine; the missing piece is the message-building step for single tasks (not pairs).

### Measurement pipeline: aggregate choice rates
`measure_pre_task_revealed()` returns individual `BinaryPreferenceMeasurement` objects. Agents needed aggregate choice rates across resamples (rate, n_parsed, n_failed). They bypassed the pipeline entirely and called `client.generate_batch()` + hand-rolled parsing. A function that wraps measurement results into aggregate rates would let agents use the existing pipeline instead of reimplementing it.

### Analysis: probe-vs-behavioral correlation utility
Every evaluation script computes Pearson r, Spearman rho, and sign agreement between behavioral deltas and probe deltas. A function like `correlate_deltas(behavioral: ndarray, probe: ndarray) -> dict` returning all three statistics would standardize this and prevent agents from forgetting or miscomputing statistics.
