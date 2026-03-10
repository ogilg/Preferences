# OOD EOT Running Log

## Setup
- Branch: research-loop/ood_eot
- Environment: IS_SANDBOX=1 (remote GPU pod), H100 80GB
- No Slack configured

## Extraction
- Extracted EOT activations for all 64 conditions across exp1a-1d
- 46 activation files (some conditions share across experiments)
- All 0 failures, 0 OOMs

## Analysis Results (EOT probe, L31)

### Overall r (all pairs)
| Exp | prompt_last | EOT   | Delta  |
|-----|------------|-------|--------|
| 1a  | 0.611      | 0.700 | +0.089 |
| 1b  | 0.649      | 0.792 | +0.143 |
| 1c  | 0.660      | 0.576 | -0.084 |
| 1d  | 0.775*     | 0.814 | +0.039 |

### On-target r
| Exp | prompt_last | EOT   | Delta  |
|-----|------------|-------|--------|
| 1a  | 0.898      | 0.865 | -0.033 |
| 1b  | 0.947      | 0.939 | -0.008 |
| 1c  | 0.859      | 0.788 | -0.071 |
| 1d  | 0.881*     | 0.905 | +0.024 |

### Probe-GT r (on-target)
| Exp | prompt_last | EOT   | Delta  |
|-----|------------|-------|--------|
| 1a  | 0.808      | 0.741 | -0.067 |
| 1b  | 0.918      | 0.926 | +0.008 |
| 1c  | 0.627      | 0.550 | -0.077 |
| 1d  | 0.704*     | 0.726 | +0.022 |

*exp1d prompt_last from n=1920 (48 conditions), EOT from n=640 (16 conditions from AL configs).
exp1a-1c have identical n between prompt_last and EOT.

### Key observations
- EOT better on overall r for 1a (+0.09), 1b (+0.14), 1d (+0.04)
- EOT worse on overall r for 1c (-0.08)
- On-target r: EOT slightly worse for 1a, 1b, 1c, better for 1d
- Pattern: EOT probes better at discriminating small off-target effects, but similar or slightly weaker on targeted pairs
