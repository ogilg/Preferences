# Test: basic computation

Minimal smoke-test experiment — just exercises the experiment-spec scaffolding with a trivial numeric computation. Not a real experiment.

## Goal

Verify that the experiment harness runs end-to-end on a tiny CPU-only task.

## Procedure

1. Generate `N = 1000` samples from a standard normal.
2. Compute the sample mean and sample variance.
3. Compare against the analytic values (0 and 1).
4. Compute Pearson correlation between two independent draws (expected ≈ 0).

## Inputs

- `seed`: 0
- `N`: 1000

## Outputs

- `results.json` with `{mean, variance, pearson_r}`
- One plot: `assets/plot_{mmddYY}_normal_histogram.png` — histogram of the draws with a fitted normal overlay.

## Notes

This spec exists only to test tooling (experiment scaffolding, plot conventions, report flow). Delete once the harness it's testing is validated.
