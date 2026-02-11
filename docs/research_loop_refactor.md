# Research Loop Refactoring Ideas

Identified from reviewing how research agents used `src/` across the OOD generalization, hidden preferences, and crossed preferences experiments.

## Done

### `score_with_probe` in `src/probes/core/evaluate.py`
Agents hand-rolled `activations @ weights[:-1] + weights[-1]` in 5+ scripts. Now a single function encapsulates the `[coef..., intercept]` convention. Both `evaluate_probe_on_data` and `evaluate_probe_on_template` use it internally.

### `extract_activations` in `src/probes/extraction/simple.py`
Lightweight wrapper around `model.get_activations_batch()` that handles message construction for single tasks (with optional system prompt), batching, and npz saving. No dependency on manifest directories, experiment stores, or completion stores.

### `aggregate_choice_rates` in `src/measurement/elicitation/measure.py`
Takes a `MeasurementBatch[BinaryPreferenceMeasurement]` and returns `{"rate", "n_parsed", "n_failed", "n_refusal"}`. Lets agents use the existing measurement pipeline instead of hand-rolling parsing.

## Dropped

### Probe scoring: simple multi-layer scorer
Trivial loop (3 lines) that varies per experiment — not worth abstracting. Agents can call `score_with_probe` in a dict comprehension.

### Analysis: probe-vs-behavioral correlation utility
Too experiment-specific (different scripts need different subsets of statistics, different delta computations). scipy.stats utilities already exist for the individual computations.
