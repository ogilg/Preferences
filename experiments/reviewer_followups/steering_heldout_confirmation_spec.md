# Held-out confirmation of L23 and the 0.06 operating point

## Question

Does the previously selected steering configuration reproduce the main pairwise
choice effect on tasks that played no role in selecting the layer or coefficient?

This is a selection-bias check, not primarily a subsampling-variance check. The
layer and coefficient are frozen from the old data and tested once on new causal
outcomes.

## Fresh confirmation set

Construct 150 in-distribution pairs from the canonical `default_test` pool:

- 50 benign--benign;
- 50 harmful--benign;
- 50 harmful--harmful.

Match the headline set’s utility-gap threshold and orientation. Exclude every task
used in the historical layer sweep, coefficient sweep, headline steering set, or
cross-persona steering set. Build the exclusion union from the existing pair
manifests/checkpoints and save the resulting pairs as a frozen confirmation
manifest before generation.

This is intentionally in-distribution: the reviewer’s concern is adaptive
configuration selection, not robustness to a distribution shift.

## Frozen protocol

- Gemma-3-27B-IT with no system prompt;
- Assistant end-of-turn probe `ridge_L23`, intervening at L23;
- contrastive steering only;
- coefficients `c ∈ {-0.06, 0, +0.06}`;
- three trials in each AB and BA order;
- the existing prompt, temperature, parser, refusal handling, and norm
  calibration.

Do not run a new layer sweep, intermediate coefficients, single-task steering,
new utility elicitation, or a second dataset.

Cost: `150 pairs × 3 coefficients × 2 orders × 3 trials = 2,700`
completions.

## Readout

Report

`P(chose steered task | c=+0.06, responded)
 - P(chose steered task | c=-0.06, responded)`,

with a paired bootstrap interval, refusal rate, and the historical point estimate
on the same probability scale. Report the three pair-type estimates and the `c=0`
baseline.

A clearly positive held-out swing shows that the selected L23/0.06 operating point
generalizes beyond its development data. A weak or null result must be reported
without retuning on this set.

## Free sampling-variance diagnostic

Using the existing 150-pair result, repeatedly bootstrap task pairs and plot the
distribution of the endpoint swing at sample sizes 50, 100, and 150. This shows
how much the reported magnitude varies with pair sampling, but it does not replace
the fresh confirmation run because it reuses the data on which the operating point
was finalized.
