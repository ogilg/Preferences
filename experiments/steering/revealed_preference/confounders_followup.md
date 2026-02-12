# H2 Differential Steering: Confounder Follow-ups

The H2 result (P(A) 0.59→0.67, p=0.000002) shows the probe direction causally shifts revealed preferences. But several confounders could explain the effect without genuine evaluative steering. These experiments address them.

We have a GPU and can run at decent scale (hundreds of pairs, thousands of observations).

## Confounders

1. **Position confound.** +coef always applied to A-position tokens, -coef to B-position. Could be "positive steering on earlier tokens biases toward outputting 'a'" rather than boosting task A's evaluative representation. H3 autoregressive result (positive→more B) already shows position-valence interactions.

2. **Token count asymmetry.** Tasks differ in length. Total steering dose = coef × n_tokens differs between A and B within each pair. Longer task gets more cumulative perturbation.

3. **Borderline pair concentration.** 57/60 pairs firmly decided. The aggregate effect is driven by ~3 borderline pairs. Those pairs might be special for unrelated reasons.

4. **Output token perturbation.** At high coefficients, we might just be making the token "a" or "b" more/less likely through residual stream perturbation, not changing the model's evaluation. Steering persists to output logits without going through "preference processing."

5. **Coherence degradation at boundary.** The 3 borderline pairs might be ones where the model is uncertain about task content, not preference. Steering pushes a confused model.

## Experiments

### E1: Order counterbalancing (addresses confound 1)

Present each pair twice: (A,B) and (B,A). Apply same differential steering (+first, -second) in both orderings.

- If the direction encodes per-task value: effect REVERSES when you swap order (the "good" task is now in the second position, getting -coef)
- If position artifact: effect stays same direction in both orderings

Run on borderline-enriched pairs (see E2) for power. ~50 borderline pairs × 2 orderings × 7 coefficients × 15 resamples.

### E2: Borderline pair enrichment (addresses confound 3)

Pre-screen 200+ pairs at coef=0 with ~20 resamples each to find the 10-20% with P(A) near 0.5. Then run differential steering only on those pairs.

Prediction: much larger effect size (maybe Δ=0.3+) because steering can actually move borderline decisions. This also gives us the statistical power we need for the other experiments.

### E3: Same-task pairs (addresses confound 1, 4)

Present the same task as both A and B. Apply differential steering.

- If P(A) shifts: strong evidence it's position-sensitive, not content-evaluation steering
- If P(A) stays at 0.5: supports evaluative interpretation (nothing to differentially evaluate)

~30 tasks × 7 coefficients × 15 resamples.

### E4: Task-length control (addresses confound 2)

Match pairs by task token count (within ±10%). Check if effect persists when total steering dose is equalized. Alternatively, normalize coefficient by token count.

Can combine with E2 (length-matched borderline pairs).

### E5: Logit lens (addresses confound 4, gives more power)

Instead of binary choice, extract logit difference between "a" and "b" at the last token position as a function of steering coefficient. Continuous measure → much more statistical power than binary choices. Can also check intermediate layers to see where the steering effect manifests.

### E6: Steer headers only (addresses confound 1, 4)

Steer only on the "Task A:\n" and "Task B:\n" header tokens, not the task content.

- If effect persists: position artifact (headers carry positional but not content info)
- If effect vanishes: content tokens matter for the steering effect

### E7: Utility-matched pairs (addresses confound 3)

Use existing TrueSkill/Thurstonian utility estimates to construct pairs with known utility differences. Check: does steering shift P(A) more for small utility gaps than large ones? Tests the borderline hypothesis systematically.

### E8: Random direction control on borderline pairs (addresses confound 3)

The current random direction control used all-firm pairs. Redo with borderline-enriched pairs. If random directions also shift borderline pairs, the probe direction isn't special. If they don't, that's the real specificity test.

## Priority

**Tier 1** (run first, highest information value):
- **E2** — Borderline enrichment. Foundation for everything else. Gives us pairs where effects are detectable.
- **E1** — Order counterbalancing. Addresses the single biggest confound. Run on E2's borderline pairs.
- **E8** — Random direction control on borderline pairs. Must confirm specificity before interpreting other results.

**Tier 2** (run next):
- **E3** — Same-task pairs. Quick, definitive test of position vs content.
- **E5** — Logit lens. Free statistical power boost, helps interpret mechanism.

**Tier 3** (if needed):
- **E4, E6, E7** — More targeted follow-ups depending on Tier 1/2 results.

## Execution plan

1. Run E2 screening (200+ pairs, coef=0, 20 resamples each) → identify borderline pairs
2. On borderline pairs, run E1 (order counterbalancing) + E8 (random direction control) in parallel
3. Run E3 (same-task) and E5 (logit lens) on borderline pairs
4. Analyze and decide on Tier 3
