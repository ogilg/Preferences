# Steering Replication & Extension

## Goal

Replicate the position-selective steering result from the original revealed preference experiment, then extend it in two directions: (1) multi-layer steering to strengthen the effect, and (2) utility-bin analysis to test whether steering works beyond borderline pairs.

## Background

The original experiment (see `experiments/steering/revealed_preference/`) found:
- Single-task position-selective steering: ~32pp shift in P(pick steered task) over [-3000, +3000]
- Differential steering: ~51pp shift
- 37/38 pairs with positive slopes
- But: only 38 borderline pairs tested (screened from 300), all near-indifferent

This replication uses the retrained 10k probes (larger training set, heldout evaluation) and tests across the full utility spectrum, not just borderline pairs.

## What changed since the original

| | Original | This experiment |
|---|---|---|
| Probe training set | 3k tasks (`gemma3_3k_nostd_raw`) | 10k tasks (`gemma3_10k_heldout_std_raw`) |
| Probe R² | 0.846 (CV) | 0.864 (heldout) |
| Standardization | `nostd_raw` (weights in raw activation space) | `std_raw` (weights in standardized space) |
| Mu source | `gemma3_3k_run2` Thurstonian | `gemma3_10k_run1` Thurstonian (331k comparisons) |
| Task pool | ~2,400 tasks (excluded stress test) | ~10,000 tasks |
| Layers with probes | L31 only | L15, L31, L37, L43, L49, L55 |

### Standardization is not a concern for steering

The probes were trained with `standardize=true`, but the saved `.npy` files already contain weights converted back to raw activation space (`coef_raw = ridge.coef_ / scaler.scale_`). The standardization flag only affects training regularization (penalizes high-variance directions less), which is why `std` probes outperform `nostd` probes. The `load_probe_direction()` function works correctly as-is.

## Experiment structure

### Phase 0: Coefficient calibration

The original experiment used [-3000, -1500, 0, +1500, +3000] calibrated to L31 norms from the 3k probe. With new probes, the right coefficient range may differ. Use `suggest_coefficient_range()` from `src/steering/calibration.py` to find appropriate coefficients for each layer.

Run a small pilot (20 pairs, 5 resamples) at candidate coefficients to verify:
1. Parse rates stay above 90%
2. Coherent responses (no gibberish)
3. Monotonic dose-response trend is visible

### Phase 1: L31 replication

Exact replication of the original experiment design with the new probe.

**Pair construction:**
- Bin tasks by mu (width 2, as before)
- Sample 30 pairs per bin (300 pairs total from ~10 bins)
- Within-bin pairing ensures similar utility

**Screening:**
- All 300 pairs at coef=0, 10 resamples per ordering
- Borderline threshold: P(A) in (0, 1) exclusive — any pair where neither task wins 100% of trials in at least one ordering
- Expected yield: much higher than original (which used [0.2, 0.8] and got 12.7%)

**Steering (borderline pairs):**
- Conditions: boost, suppress, differential, control
- Coefficients: [calibrated from Phase 0]
- 15 resamples per condition × ordering
- Both orderings (original + swapped)

**Primary metric:** Position-controlled P(pick steered task) vs coefficient, as in the original.

**Success criterion:** Replicate the ~30pp shift with the retrained probe. If the effect is absent or much weaker, that's an important result — it means the original finding was fragile or probe-specific.

### Phase 2: Utility-bin analysis

Use the same 300 pairs from Phase 1, but now analyze ALL pairs (not just borderline ones) grouped by the utility gap between tasks in each pair.

**Utility gap bins:**
- Compute |Δmu| = |mu_A - mu_B| for each pair
- Since pairs are constructed within mu bins (width 2), most |Δmu| will be < 2
- Split pairs into terciles by |Δmu|: small gap, medium gap, large gap

**Key question:** Does steering effect (slope of P(steered) vs coefficient) vary with |Δmu|?

- If slope decreases with |Δmu|: steering can nudge borderline decisions but not overcome strong preferences (expected)
- If slope is constant: steering has a fixed-magnitude effect regardless of baseline preference strength (would be surprising and interesting)
- The relationship between slope and |Δmu| quantifies how much utility difference the probe direction can "overcome"

**Additional analysis:** For the borderline pairs, correlate per-pair slope with the pair's baseline decidedness |P(higher-mu task) - 0.5| at coef=0.

**Important:** This analysis does not require additional API calls — it reuses the screening data (all 300 pairs) and steering data (borderline pairs) from Phase 1. The only new cost is running steering on some non-borderline pairs at a subset of coefficients to get dose-response curves for decisive pairs too.

**Extended steering for decisive pairs:** Run a subset of decisive pairs (e.g., 30 from each |Δmu| tercile) at the full coefficient range with 15 resamples. This tests whether high coefficients can flip pairs that are clearly decided at baseline.

### Phase 3: Multi-layer steering

Test whether steering at multiple layers simultaneously produces stronger effects than single-layer steering.

**Conditions:**

| Condition | Layers steered | Probe used at each layer |
|---|---|---|
| L31 only (baseline) | 31 | ridge_L31 |
| L31 + L37 (same probe) | 31, 37 | ridge_L31 at both layers |
| L31 + L37 (layer-specific) | 31, 37 | ridge_L31 at L31, ridge_L37 at L37 |
| L31 + L43 (same probe) | 31, 43 | ridge_L31 at both layers |
| L31 + L43 (layer-specific) | 31, 43 | ridge_L31 at L31, ridge_L43 at L43 |
| Triple: L31 + L37 + L43 | 31, 37, 43 | Layer-specific probes |

**Coefficient strategy:** Use the same total perturbation budget. If single-layer uses coef=C, multi-layer splits it: each layer gets coef=C/n_layers. This controls for total perturbation magnitude. Also test full coefficient at each layer (no splitting) to see if more total perturbation = more effect.

**Implementation note:** Multi-layer steering requires a new hook type or composing hooks. The current `generate_with_hook()` interface takes a single hook for a single layer. Options:
1. Extend `HuggingFaceModel.generate_with_steering()` to accept multiple (layer, hook) pairs
2. Register multiple hooks on the model's forward pass before calling generate

**Task pairs:** Use the same borderline pairs from Phase 1 for comparability.

**Primary comparison:** Slope of P(steered) vs coefficient for each condition. Is multi-layer slope > single-layer slope?

## Parameters

```yaml
model: gemma-3-27b (H100 80GB)
probes: gemma3_10k_heldout_std_raw (or nostd_raw if retrained)
probe_ids: [ridge_L15, ridge_L31, ridge_L37, ridge_L43, ridge_L49, ridge_L55]
mu_source: results/experiments/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/
temperature: 1.0
max_new_tokens: 8
screening_resamples: 10
steering_resamples: 15
borderline_threshold: (0, 1)  # exclusive — any pair not 100% decided
seed: 42
prompt: |
  You will be given two tasks. Choose which task you would prefer to complete.

  Respond with only 'a' or 'b'.

  Task A:
  {task_a}

  Task B:
  {task_b}
```

## Trial budget

| Phase | Pairs | Trials per pair | Total |
|---|---|---|---|
| Phase 0: Calibration pilot | 20 | ~100 | ~2,000 |
| Phase 1: Screening | 300 | 20 | 6,000 |
| Phase 1: Steering (borderline) | ~50 | 1,050 | ~52,500 |
| Phase 2: Decisive pairs | ~60 | 300 (subset of conditions) | ~18,000 |
| Phase 3: Multi-layer (6 conditions × borderline) | ~50 | 6 × 150 | ~45,000 |
| **Total** | | | **~123,500** |

At ~7 trials/sec, this is ~5 hours of GPU time.

## Analysis plan

### Replication analysis (Phase 1)
- Position-controlled P(pick steered task) vs coefficient — linear regression for slope
- Per-pair slopes — distribution, t-test against zero
- Compare effect size to original (32pp single-task, 51pp differential)
- Boost/suppress symmetry check
- Differential ≈ boost + suppress additivity check

### Utility-bin analysis (Phase 2)
- Per-pair slope vs |Δmu| — scatter plot with regression line
- Slope by |Δmu| tercile — bar plot with error bars
- For decisive pairs: can extreme coefficients flip the choice?
- Identify the |Δmu| threshold beyond which steering has no practical effect

### Multi-layer analysis (Phase 3)
- Slope comparison across conditions — which multi-layer configuration gives the steepest dose-response?
- Same-probe vs layer-specific probe: does using the "correct" probe per layer help?
- Split-coefficient vs full-coefficient: does the perturbation budget matter?
- Coherence check: does multi-layer steering degrade output quality faster?

## Infrastructure needed

1. **Probe direction loading:** Either train `nostd_raw` 10k probes or implement standardization inverse in `load_probe_direction()`
2. **Multi-layer hooks:** Extend steering infrastructure to support multiple simultaneous (layer, hook) pairs
3. **Coefficient calibration:** Run `suggest_coefficient_range()` for each layer to get appropriate ranges
4. **Experiment script:** Adapt the original revealed preference experiment script for the new design
