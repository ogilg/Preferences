# Steering-configuration selection audit

## What was actually done

### Probe fitting

The canonical 6,000 tasks were split into:

- 4,000 `default_train` tasks for fitting each ridge probe;
- 1,000 `default_eval` tasks for selecting ridge alpha;
- 1,000 `default_test` tasks for reported probe metrics and steering-pair
  construction.

This separation is clean for probe fitting and alpha selection.

### Intervention layer

The layer sweep trained end-of-turn probes at 20 layers:

`2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59`.

Steering was evaluated on 50 pairs sampled from `default_test`, using
`c ∈ {±0.03, ±0.05}` and both task orders. The causal effect peaked at L23.
L23 was therefore selected from observed steering efficacy, not from probe
predictivity (which peaks around L29--L32).

### Headline evaluation

After observing the layer sweep, the main result was rerun at L23 on a new
harm-balanced set of 150 pairs: 50 benign--benign, 50 harmful--benign, and 50
harmful--harmful. These pairs were also constructed from `default_test`.

Thus the headline pairs differ from the 50 layer-sweep pairs, but layer selection
and headline evaluation draw from the same 1,000-task split. Task overlap between
the two historical samples was not precluded.

### Coefficient range

Earlier steering work swept approximately
`c ∈ {0, ±0.03, ±0.05, ±0.07, ±0.10}` and observed coherence degradation at larger
magnitudes. The later L23 fine-grained run used
`c ∈ {0, ±0.01, ±0.02, ±0.04, ±0.06}` on the 150-pair set. The paper adopted
`|c| ≤ 0.06` after inspecting parseability/coherence on those same results.

The coefficient parameterisation itself was fixed in advance, but the displayed
operating endpoint was not selected on an independent set.

## Conclusion

The reviewer's concern is valid. There was meaningful separation between probe
training, alpha selection, and steering, but no fully independent steering
development/test split for layer and operating-range selection.

## Least-cumbersome confirmatory protocol

Treat the existing sweeps as development data and use a fresh, task-disjoint
**in-distribution confirmation set**. The purpose is to test whether adaptive
layer/coefficient selection inflated the causal effect, not to test distribution
shift. The confirmation set should therefore match the original evaluation in:

- source-dataset proportions;
- topic proportions;
- harmful/benign pair-type proportions;
- utility-gap filter;
- prompt template, decoding settings, and order counterbalancing.

Before generating confirmation outcomes, freeze:

- intervention layer L23;
- Assistant end-of-turn `ridge_L23` probe;
- endpoint coefficients `-0.06, +0.06`;
- parsing, exclusion, and coherence rules.

Evaluate once on a fresh 150-pair task-disjoint in-distribution set at the frozen
endpoints and `c=0`. There is no reason to repeat either sweep. The complete
2,700-completion procedure is in
`steering_heldout_confirmation_spec.md`.
