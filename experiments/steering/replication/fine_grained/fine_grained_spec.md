# Fine-Grained Steering Experiment Spec

**Date:** 2026-02-23
**Parent:** `experiments/steering/replication/`
**Model:** gemma-3-27b (H100 80GB)
**Probes:** `gemma3_10k_heldout_std_raw` — ridge at L31, L37, L43, L49, L55

---

## Motivation

The replication experiment tested only 4 non-zero coefficients (±2641, ±5282) and found:

1. **Non-monotonic dose-response for boost_a** — both positive and negative coefficients increase P(a), suggesting position bias amplification rather than directional steering.
2. **Saturation/reversal at max coefficient** — boost_a and L31_only reverse at +5282 (10% norm), but split-budget multi-layer conditions remain effective.
3. **boost_b and diff_ab show genuine dose-response** — but with only 2 positive and 2 negative non-zero coefficients, the shape of the response curve is unclear.
4. **Only L31, L37, L43 tested** — probes exist at L49 and L55 with comparable R² (0.835, 0.836) but were never used for steering.

With 4 coefficients we can't distinguish a smooth dose-response from a step function, identify the optimal coefficient, or locate the saturation threshold precisely. This experiment fills in the curve with higher resolution and more pairs.

## Design

### Pairs

Pre-select borderline pairs from existing active learning measurement data. The 10k measurement run contains ~31,400 within-bin pairs (|Δmu| < 2, both tasks have activations), each compared 3–5 times. Of these, ~4,500 are borderline (P(a) strictly between 0 and 1 in the measurement data) and ~2,000 are near 50/50 (P(a) in [0.3, 0.7]).

**Pair selection:** Sample 300 pairs from the ~4,500 borderline pool, stratified by both mu-bin (to cover the full utility range) and measurement P(a) (to ensure good coverage of the 0.2–0.8 range for the steerability-vs-decidedness analysis). Near-50/50 pairs (~2,000 available) are the most informative but also the rarest; oversample them if needed to get ~100 pairs with measurement P(a) in [0.3, 0.7]. This gives us ~4× the replication's 77 pairs, all pre-verified as borderline.

Note: with 3–5 comparisons per pair, borderline (0,1) and [0.2, 0.8] are equivalent (the only non-0/1 P(a) values with 3 comparisons are 1/3 and 2/3). The coef=0 data point in the steering experiment provides a higher-quality re-screen (10 resamples × 2 orderings) for post-hoc filtering if needed.

### Conditions

Drop redundant conditions (suppress_a/b are mirrors of boost_a/b; diff_ba is a mirror of diff_ab — all already covered by sweeping negative coefficients). Run only:

1. **boost_a** — steer task A with +direction
2. **boost_b** — steer task B with +direction
3. **diff_ab** — +direction on A, −direction on B

### Coefficients

Fine-grained sweep expressed as % of mean activation norm at the steering layer:

```
multipliers = [-10%, -7.5%, -5%, -4%, -3%, -2%, -1%, 0%, +1%, +2%, +3%, +4%, +5%, +7.5%, +10%]
```

15 points total. Dense sampling in the ±1–5% range where the replication showed effects; sparser at the extremes for saturation characterization. The 0% point serves as the within-experiment control.

For L31 (mean norm ≈ 52,822): coefficients range from −5282 to +5282.

### Layers

#### Single-layer conditions (all 3 conditions × 15 coefficients):

- **L31** (R²=0.864) — primary, replicates previous results at finer resolution
- **L49** (R²=0.835) — late layer, never tested for steering
- **L55** (R²=0.836) — latest available layer

#### Multi-layer split conditions (diff_ab only × 15 coefficients):

Budget split evenly across layers, using layer-specific probe directions:

- **L31+L37 layer/split** — best performer from Phase 3
- **L31+L49 layer/split** — wide layer gap, tests whether distant layers cooperate
- **L49+L55 layer/split** — late-only steering, no L31 involvement

Each layer gets `coef / n_layers` at its own probe direction.

### Resamples

10 resamples per condition × ordering × coefficient. (Previous experiment used 15; reducing to 10 to accommodate the larger pair count. Random control experiment showed 10 resamples gives stable estimates.)

### Random direction control

Run 1 random direction at L49 and L55 (diff_ab only, same coefficient grid) to verify probe-specificity at later layers. L31 already has random control from the previous experiment.

### Trial count estimate

Single-layer: 3 conditions × 3 layers × 15 coefficients × 300 pairs × 2 orderings × 10 resamples = **810,000 trials**

Multi-layer: 1 condition × 3 combos × 15 coefficients × 300 pairs × 2 orderings × 10 resamples = **270,000 trials**

Random control: 1 condition × 2 layers × 15 coefficients × 300 pairs × 2 orderings × 10 resamples = **180,000 trials**

**Total: ~1,260,000 trials.**

### Execution order

1. **L31 single-layer** (all 3 conditions, all 300 pairs).
2. **L49, L55 single-layer** (all 3 conditions, all 300 pairs).
3. **Multi-layer split conditions** (diff_ab only).
4. **Random controls** (L49, L55).

Each phase commits results before starting the next.

## Questions to answer

1. **Dose-response shape**: What does the full dose-response curve look like for each condition? Is there a monotonic region, or is it flat/noisy? Does boost_a show the same sign-independent P(a) increase seen in the replication?
2. **Saturation**: Where (if anywhere) do effects saturate or reverse? Is this consistent across conditions and layers?
3. **Later layers**: What happens when steering at L49 and L55? Do they produce any effect at all? How does the dose-response compare to L31?
4. **Late multi-layer**: What does L49+L55 layer/split (no L31) produce? Could be anything from strong effects to nothing.
5. **Probe vs random at later layers**: Do random directions at L49/L55 produce similar or different effects from the probe direction? (L31 showed probe > random; unclear if this holds at later layers.)
6. **Multi-layer vs single-layer**: With ~300 pairs (MDE ~5pp), can we resolve whether multi-layer split-budget steering differs from single-layer? The replication couldn't (MDE ~10pp).
7. **Steerability vs baseline decidedness**: How does steering effect vary with baseline P(a)? Plot effect as a function of baseline decidedness across conditions and layers. Is it a smooth gradient, a threshold, or flat? Does the pattern differ between conditions (e.g. diff_ab vs boost_a) or layers?

## Infrastructure

Reuse the existing `scripts/replication/run_experiment.py` infrastructure. Key changes:

- New coefficient grid (parameterized via config, not hardcoded)
- Pair selection from active learning measurement data (pre-screened borderline)
- Add L49 and L55 to the layer configs
- Drop suppress_a/b/diff_ba conditions
- Coef=0 serves as control (no separate control sweep)

### Batched sampling

Use `generate_n()` / `generate_with_steering_n()` / `generate_with_multi_layer_steering_n()` instead of serial resample loops. These methods use `num_return_sequences` to generate all resamples in a single forward pass, sharing the prefill computation across samples. For 10 resamples with 8-token completions, this eliminates 9 redundant prefills per trial cell.

Replace the serial pattern:
```python
responses = []
for _ in range(n_resamples):
    resp = model.generate_with_steering(messages=messages, layer=layer, steering_hook=hook, temperature=TEMPERATURE)
    responses.append(parse_response(resp))
```

With:
```python
raw = model.generate_with_steering_n(messages=messages, layer=layer, steering_hook=hook, n=n_resamples, temperature=TEMPERATURE)
responses = [parse_response(r) for r in raw]
```

Same for `generate()` → `generate_n()` (coef=0 case) and `generate_with_multi_layer_steering()` → `generate_with_multi_layer_steering_n()`.

Before running the experiment, validate the batched methods work on this GPU:
```bash
pytest tests/test_steering_e2e.py -v -s -k "TestGenerateN"
```

### Checkpointing

Append results to a JSONL file after each (pair, ordering) block rather than accumulating in memory and saving at the end. This way a pod reboot or crash only loses the current pair, not hours of work. Each phase writes to `results/phase{N}_results.jsonl`. On resume, skip pairs already in the JSONL.
