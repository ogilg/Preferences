# Position-bias audit

## Conclusion

No new position-bias experiment is needed. Utility elicitation randomized task
order across queried pairs and is unbiased in expectation; steering used the
stronger control of evaluating every pair in both orders.

## Utility elicitation

The canonical Assistant runs used:

```yaml
n_samples: 3
pair_order_seed: 42
include_reverse_order: false
```

For each active-learning batch, `apply_pair_order` randomly assigned approximately
half the unordered pairs to each display order before measurement. All three
repeated choices for a given pair retained that assigned order. The Thurstonian
fit used task identities rather than display slots.

Random assignment makes any systematic first- or second-position preference
independent of task identity in expectation. Reusing the assigned order for the
three repeats can affect variance, but it does not systematically favor particular
tasks. This should be described as randomized order assignment, not within-pair
AB/BA counterbalancing.

## Steering

The steering experiment did use full within-pair counterbalancing:

- every pair was run in both AB and BA order;
- under reversal, the coefficient sign was changed so the intervention remained
  attached to the same task;
- parsed A/B choices were mapped back to task identity;
- results pooled both orders.

This directly controls the position bias relevant to the causal steering result.

## Paper wording

> For utility elicitation, we randomly assigned the presentation order of each
> queried task pair with a fixed seed. This makes position preference independent
> of task identity in expectation. For steering, we evaluated every pair in both
> AB and BA orders, kept the intervention attached to the same task under reversal,
> remapped responses to task identity, and pooled both orders.

Do not claim that utility elicitation presented every pair in both orders. No new
generation or additional appendix experiment is required.
