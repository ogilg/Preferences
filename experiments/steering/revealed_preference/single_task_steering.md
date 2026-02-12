# Single-Task Steering Experiment

## Question

Does adding the probe direction to a task's tokens increase the probability of picking that task?

Previous experiments used differential steering (+probe on A, −probe on B simultaneously), which confounds three effects: boosting the steered task, suppressing the other, and positional bias. This experiment isolates each.

## Conditions

| Condition | Task A tokens | Task B tokens | Tests |
|-----------|--------------|--------------|-------|
| Boost A | +probe | — | Does boosting A increase P(A)? |
| Boost B | — | +probe | Does boosting B increase P(B)? |
| Suppress A | −probe | — | Does suppressing A decrease P(A)? |
| Suppress B | — | −probe | Does suppressing B decrease P(B)? |
| Differential A+B− | +probe | −probe | Replication of H2 for comparison |
| Differential A−B+ | −probe | +probe | Reverse differential |
| Control | — | — | Baseline P(A) |

Each condition is run in **both orderings** (original and swapped task positions) to control for position effects.

## Primary metric

**P(pick steered task)** as a function of coefficient, controlling for position by averaging across both orderings.

For boost conditions: does +probe on task X increase P(X)?
For suppress conditions: does −probe on task X decrease P(X)?
For differential: does +probe on X combined with −probe on Y increase P(X)?

## Coefficient design

Each condition is swept across a symmetric range of coefficients: **[−3000, −1500, 0, +1500, +3000]** (5 values).

The full sign range is intentional: for a "Boost A" condition (+probe on A's tokens), positive coefficients boost while negative coefficients reverse the probe direction, effectively suppressing. This gives a full dose-response curve per condition, and provides an internal replication — the negative end of "Boost A" should mirror the positive end of "Suppress A". This symmetry is a built-in consistency check.

## Pair construction

Pair tasks within **mu bins** (not Δmu bins). Binning by individual mu gives pairs with naturally similar utility — more likely to be borderline.

Mu bins (width 2, centered):

| Mu bin | N tasks available |
|--------|------------------|
| [−10, −8) | 436 |
| [−8, −6) | 263 |
| [−6, −4) | 156 |
| [−4, −2) | 188 |
| [−2, 0) | 318 |
| [0, +2) | 356 |
| [+2, +4) | 376 |
| [+4, +6) | 371 |
| [+6, +8) | 364 |
| [+8, +10) | 139 |

Sample ~30 pairs per bin (300 pairs total from 10 bins). Within each bin, pair tasks randomly — Δmu will be small by construction.

## Two-phase design

### Phase 1: Screening (baseline only)

Run all 300 pairs at coef=0 with 10 resamples per pair in both orderings. Identify borderline pairs (P(A) between 0.2 and 0.8 in at least one ordering).

Expected yield: ~15-25% borderline based on within-bin pairing (vs 5% with Δmu-matched pairs from v2). Target: 60+ borderline pairs across the mu spectrum.

### Phase 2: Steering

Run all 7 conditions on the screened borderline pairs.

- **Coefficients**: [−3000, −1500, 0, +1500, +3000]
- **Resamples**: 15 per condition × ordering
- **Orderings**: 2 (original, swapped)

Per pair: 7 conditions × 5 coefficients × 15 resamples × 2 orderings = 1,050 trials.

With 60 borderline pairs: **63,000 trials** total.

## Trial budget

| Phase | Pairs | Trials per pair | Total |
|-------|-------|----------------|-------|
| Screening | 300 | 20 (10 resamples × 2 orderings) | 6,000 |
| Steering | ~60 | 1,050 | ~63,000 |
| **Total** | | | **~69,000** |

## Controlling for position

"Boost A" means +probe on position A's tokens. In the original ordering position A contains task X; in the swapped ordering it contains task Y. To get a position-controlled estimate of steering task X, we need to combine across conditions and orderings by **task identity**, not position.

For a given pair (task X in original position A, task Y in original position B):

| What we want | Condition + ordering that measures it | Raw metric |
|-------------|--------------------------------------|-----------|
| Effect of +probe on X | Boost A, original | P(A) |
| Effect of +probe on X | Boost B, swapped | 1 − P(A) |
| Effect of +probe on Y | Boost A, swapped | P(A) |
| Effect of +probe on Y | Boost B, original | 1 − P(A) |

**Position-controlled P(pick steered task)** for +probe on task X:
- = average of P(A) from "boost A, original" and (1 − P(A)) from "boost B, swapped"
- Task X appears in position A once and position B once → position cancels out.

Similarly for −probe (suppress conditions). The same logic applies to differential, where both +probe and −probe are applied simultaneously.

## Analysis plan

### Primary analysis

For each pair, compute the position-controlled P(pick steered task) vs coefficient by combining across orderings as described above. Then:

1. Aggregate across pairs: P(pick boosted task) vs coefficient — linear regression for slope and significance
2. Aggregate across pairs: P(pick suppressed task) vs coefficient — same
3. Per-pair slopes for later analysis

### Order-sensitivity

For each condition, also report the **per-ordering** slopes separately before combining. This tells us how much of the effect depends on which task is in which position.

For boost-only (the cleanest case):
- Slope in "boost A, original" = effect of +probe when steered task is in position A
- Slope in "boost B, original" = effect of +probe when steered task is in position B (measured as change in 1−P(A))

If these slopes are equal, the effect is purely task-dependent (position doesn't matter). If "boost A" has a larger slope than "boost B", position A amplifies the steering effect. The decomposition:
- **Task-dependent component**: (slope_boost_A + slope_boost_B) / 2 — present in both positions
- **Position-sensitive component**: (slope_boost_A − slope_boost_B) / 2 — depends on position

This is a cleaner decomposition than the differential steering version, because here we're steering only one task. In differential steering, the position/task decomposition was confounded by the simultaneous +/− on both positions. With single-task steering, the decomposition is unambiguous: does steering work equally well regardless of where the task sits?

### Comparisons

- **Single vs differential**: Is the boost-only effect smaller than differential? (Differential gets both boost + suppress.)
- **Boost vs suppress**: Is boosting a task as effective as suppressing the competitor?
- **Additivity**: Does differential ≈ boost + suppress, or is there an interaction?

### Internal consistency

The negative-coefficient end of "Boost A" should mirror the positive-coefficient end of "Suppress A" (both effectively apply −probe to A's tokens). Comparing these provides a sanity check that the conditions are working as expected.

### Scaling with decidedness

- Per-pair: correlate steering slope with baseline decidedness |P(higher-μ) − 0.5|
- Per-mu-bin: does the effect vary across the utility spectrum?

## Implementation

### Steering infrastructure

This experiment uses the HuggingFace steering pipeline in `src/models/`:

- **Model**: `HuggingFaceModel` from `src/models/huggingface_model.py` — provides `generate_with_steering(messages, layer, steering_hook, temperature, max_new_tokens)`.
- **Probe loading**: `load_probe_direction(manifest_dir, probe_id)` from `src/probes/core/storage.py` — returns `(layer, unit_direction)`. The probe weights are already in raw activation space and normalized to a unit vector.
- **Steering hooks**: Defined in `src/models/base.py`. The `SteeringHook` type is `Callable[[torch.Tensor, int], torch.Tensor]` taking `(resid, prompt_len)`.

### Hooks needed

Three hook types are needed (two already exist, one is new):

1. **Position-selective steering** (new) — steer only one task's token span during prompt processing:
   ```python
   def position_selective_steering(steering_tensor, start, end) -> SteeringHook:
       def hook(resid, prompt_len):
           if resid.shape[1] > 1:  # prompt processing, not autoregressive
               resid[:, start:end, :] += steering_tensor
           return resid
       return hook
   ```
   Used for: Boost A, Boost B, Suppress A, Suppress B.

2. **Differential steering** (new) — steer both task spans with opposite signs:
   ```python
   def differential_steering(steering_tensor, a_start, a_end, b_start, b_end) -> SteeringHook:
       def hook(resid, prompt_len):
           if resid.shape[1] > 1:
               resid[:, a_start:a_end, :] += steering_tensor
               resid[:, b_start:b_end, :] -= steering_tensor
           return resid
       return hook
   ```
   Used for: Differential A+B−, Differential A−B+.

3. **No-op** — control condition uses `coef=0` which zeroes the steering vector, so any hook works.

### Token span detection

To identify which tokens correspond to Task A and Task B, use offset mapping from the tokenizer (pattern from `run_h2_scaled.py`):

1. Format messages with `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`
2. Find character positions of task text after "Task A:\n" and "Task B:\n" markers
3. Tokenize with `return_offsets_mapping=True` and map character ranges to token ranges

### Pairwise choice prompt

```
You will be given two tasks. Choose which task you would prefer to complete.

Respond with only 'a' or 'b'.

Task A:
{task_a}

Task B:
{task_b}
```

Parse response with regex: accept 'a'/'b' directly, or 'task a'/'task b' patterns. Max 8 new tokens.

### Swapped ordering

For the swapped ordering, simply reverse which task goes in position A vs B in the prompt, and swap the token spans accordingly. The same steering conditions apply (e.g., "Boost A" always steers position A's tokens, regardless of which task is there).

## Data

- **Task mu values**: `results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a1ebd06e.csv`
- **Probe direction**: `results/probes/gemma3_3k_nostd_raw/` (ridge_L31) — non-standardized, no demeaning. Probe weights are unit vectors in raw activation space. Four probe variants exist (`std_demean`, `std_raw`, `nostd_demean`, `nostd_raw`) trained with different preprocessing; `nostd_raw` is the default for steering since its weights live in the same space as the model's activations without any standardization transform.
- **Model**: Gemma-3-27B on H100 80GB
