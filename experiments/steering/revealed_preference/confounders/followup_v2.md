# H2 Confounder Follow-up v2: Scaled with Utility-Matched Pairs

## Motivation

The v1 confounder experiments found the probe direction is significantly stronger than random (p=0.003) but had only 12 borderline pairs from random sampling (4.8% hit rate). Key limitations:
- Small N (12 pairs) limits statistical power
- Random pairing is wasteful — 95% of pairs are firmly decided
- Token span identification is untested
- Header-only steering (E6) was never run
- Experiments overlap and can be consolidated

## Utility-Matched Pair Construction

We have Thurstonian mu values for 3000 tasks (all in the activations file). Instead of random pairing + screening:

1. Sort tasks by mu
2. Group into clusters of tasks with similar mu (within ±0.5 or similar threshold)
3. Pair tasks within each cluster — these should be borderline by construction
4. Verify borderline status with a quick screen (5 resamples at coef=0)

This should yield 50-100+ borderline pairs efficiently. We can also construct pairs across a range of utility gaps (Δmu = 0, 1, 2, 5, 10) to test whether steering effectiveness scales with closeness.

## Infrastructure Improvements

Before running experiments:

### 1. Shared steering utilities module

Extract from the duplicated code into `src/steering/`:
- `find_task_spans(model, messages, task_a_text, task_b_text)` — token span identification
- `differential_steering(tensor, a_start, a_end, b_start, b_end)` — the steering hook
- `parse_choice(response)` — response parsing
- `header_only_steering(tensor, header_a_span, header_b_span)` — steer only on "Task A:\n" / "Task B:\n" tokens

### 2. Token span tests

Write tests that verify `find_task_spans` returns correct token ranges:
- Roundtrip: decode the token range and check it matches the original task text
- Edge cases: very short tasks, very long tasks, tasks with special characters
- Header span identification for E6

## Experiments

Run as a single consolidated experiment with multiple conditions on the same set of utility-matched borderline pairs.

### Conditions

For each pair, run all of the following:

| Condition | Description | What it tests |
|-----------|-------------|---------------|
| **Probe differential** | +coef on A tokens, -coef on B tokens | Main effect (baseline) |
| **Probe differential, swapped** | Same pair in (B,A) order | Position confound (E1) |
| **Probe same-task** | Same task as both A and B | Pure position effect (E3) |
| **Probe header-only** | Steer only "Task A:\n" / "Task B:\n" header tokens | Position vs content (E6) |
| **Random differential** | N random orthogonal directions, differential | Specificity (E8) |

### Design

- **Pairs**: ~110 utility-matched pairs across Δmu bins (see E7 below for breakdown)
- **Every pair gets both orderings** (original + swapped) — enables position-adjusted analysis
- **Coefficients**: [-3000, -1500, 0, +1500, +3000] (5 levels)
- **Resamples**: 15 per condition
- **Random directions**: 20 for the specificity control (run on borderline subset only)
- **Header-only**: Run on borderline subset only
- **Same-task subset**: 30 tasks (for pure position test)

### Predictions

| Condition | If evaluative | If positional |
|-----------|--------------|---------------|
| Probe differential | Strong dose-response | Strong dose-response |
| Swapped order | Slope reverses (in remapped frame) | Same slope direction |
| Same-task | No effect (Δ=0) | Effect persists |
| Header-only | No effect (headers have no evaluative content) | Effect persists (position info in headers) |
| Random directions | Smaller abs(Δ) than probe | Similar abs(Δ) to probe |

The **header-only** condition is particularly diagnostic: if steering just the "Task A:" / "Task B:" headers (which carry position but not task content) produces an effect comparable to steering the full task, that's strong evidence for a positional mechanism. If it vanishes, the task content matters.

### Analysis

- Report all conditions in one table
- Key comparisons: probe vs random (z-test), probe vs header-only (paired test), original vs swapped (slope comparison)
- Per-pair analysis: does the effect correlate with Δmu? (E7 from original spec)
- Also construct some pairs with large Δmu (5-10) to confirm they resist steering (like the original firm pairs)

## Utility Gap Analysis (E7)

With utility-matched pairs, we can systematically test: does steering effectiveness decrease as the utility gap increases?

### Pair construction

Sample pairs across the full range of utility gaps:

| Δmu bin | Target pairs | Expected baseline |
|---------|-------------|-------------------|
| 0-1 | 30+ | ~borderline (P(A) near 0.5) |
| 1-2 | 20 | slightly decided |
| 2-3 | 20 | moderately decided |
| 3-5 | 20 | firmly decided |
| 5+ | 20 | very firm |

Every pair gets **both orderings** (original + swapped) so we can decompose position vs content.

### Position-adjusted steering effect

For each pair, the raw steering slope conflates position and content. To isolate the content-dependent (evaluative) component:

1. Compute per-pair slope in original ordering: `slope_orig`
2. Compute per-pair slope in swapped ordering: `slope_swap`
3. **Position-adjusted effect** = `(slope_orig - slope_swap) / 2`

This is the component that reverses when you swap order — i.e., it tracks the task content, not the position. The positional component `(slope_orig + slope_swap) / 2` should be roughly constant across Δmu bins (it doesn't depend on which tasks are paired).

### Key plot

**Scatter: Δmu vs position-adjusted steering effect.** Each point is one pair. Prediction:

- Δmu near 0: large position-adjusted effect (steering can flip borderline decisions)
- Δmu large: position-adjusted effect → 0 (firm preferences resist content-level steering)
- Positional component stays roughly flat across Δmu (it doesn't depend on preference strength)

Overlay: fit a curve (logistic or exponential decay) to the position-adjusted effect as a function of Δmu. The decay rate tells us how strongly preferences resist steering.

### Secondary plot

**Δmu vs positional component** — should be flat. If it's not flat (e.g., larger positional effect for borderline pairs), that's informative about how position bias interacts with decision uncertainty.
