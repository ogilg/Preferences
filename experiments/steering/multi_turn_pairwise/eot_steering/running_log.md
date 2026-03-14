# EOT Steering Running Log

## Setup
- Branch: research-loop/eot_steering
- Pod: 1x H100 80GB
- All data files confirmed present (probe direction, extraction metadata, Thurstonian scores)

## Pilot Run (3 pairs x 1 resample)
- Pipeline validated successfully
- 42 trials in 56 seconds (~1.35s/trial for single generation)
- 4 parse failures (9.5%) - all from same stress-test pair (model refused)
- Refusal pattern: "I absolutely cannot and will not fulfill Task B"
- Fixed ACTIVATIONS_PATH to point to .npz file (not directory)

## Coefficient Calibration
- Mean activation norm at L32: 41,676
- Multiplier -> Coefficient mapping:
  - -0.050 -> -2083.8
  - -0.030 -> -1250.3
  - -0.020 -> -833.5
  -  0.000 -> 0.0
  - +0.020 -> +833.5
  - +0.030 -> +1250.3
  - +0.050 -> +2083.8

## Data Loading
- 10,000 tasks with Thurstonian scores
- 9,628 task objects loaded (372 tasks missing from data files)
- 500 pairs sampled: 100 borderline, 200 moderate, 200 decisive

## Full Run (completed)
- 35,000 total trials (500 pairs x 2 orderings x 7 coefficients x 5 resamples)
- Generation speed: ~3.6 trials/sec
- Total runtime: 2h44m

## Analysis Results
- Overall parse rate: 96.9% (1,091 failures out of 35,000)
- Parse rates per multiplier: 96.4%–97.5% (no systematic pattern)
- **Null result**: P(high-mu) = 0.675–0.677 across all multipliers (completely flat)
- Steering effect: 0.1 pp (essentially zero)
- By stratum: borderline -0.1pp, moderate +0.2pp, decisive -0.0pp (all ~zero)
- Per-pair slopes: mean=0.0097, median=0.0, fraction positive=6.2%
- Spearman r=0.679, p=0.094 (not significant)

## Success Criteria
1. Monotonic dose-response: FAIL (Spearman r=0.679, p=0.094)
2. Steering effect > 10pp: FAIL (0.1pp)
3. Borderline > Decisive: FAIL (-0.1pp vs -0.0pp)
4. Parse rates > 90%: PASS (min=96.4%)
Overall: SOME FAIL — clear null result

## Interpretation
The EOT steering intervention at the assistant-turn end_of_turn token has no effect on preference choices. The probe direction at L32 (tb-5) does not causally influence pairwise preference decisions when steered at this position. This contrasts with the probing results where this direction predicts preferences — suggesting the representation is read-out at a different point or via a different mechanism than single-position steering can capture.
