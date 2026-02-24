# Fine-Grained Steering: Phases 2–4 with Batched Sampling

**Date:** 2026-02-23
**Parent:** `experiments/steering/replication/fine_grained/fine_grained_spec.md`
**Model:** gemma-3-27b (H100 80GB)
**Prerequisite:** Phase 1 (L31 single-layer) must be complete — `results/phase1_L31.json` must exist.

---

## Context

Phase 1 ran with serial resampling (10 individual `generate()` calls per trial cell). The codebase now has `generate_n()` / `generate_with_steering_n()` / `generate_with_multi_layer_steering_n()` methods that generate all resamples in a single forward pass via `num_return_sequences`, sharing the prefill computation.

This spec continues phases 2–4 using the batched methods for ~5-8x faster inference.

## Step 1: Pull latest code and run tests

```bash
git pull origin main
```

Run the e2e tests to verify `generate_n` works correctly on this GPU:

```bash
pytest tests/test_steering_e2e.py -v -s -k "TestGenerateN" 2>&1 | tee results/test_generate_n.log
```

All tests must pass, especially the speedup benchmarks (`test_batched_faster_than_serial`, `test_steered_batched_faster_than_serial`). If the speedup is < 1.5x on this hardware, fall back to serial sampling and skip step 2.

## Step 2: Modify experiment script to use batched sampling

In `scripts/fine_grained/run_experiment.py`, replace the serial resample loop in all three runner functions with batched calls. The pattern is the same in each:

### `run_single_layer_steering` (lines ~200-212)

**Before:**
```python
responses = []
for _ in range(n_resamples):
    if coef == 0.0:
        resp = model.generate(messages, temperature=TEMPERATURE)
    else:
        resp = model.generate_with_steering(
            messages=messages, layer=layer,
            steering_hook=hook, temperature=TEMPERATURE,
        )
    responses.append(parse_response(resp))
    done += 1
```

**After:**
```python
if coef == 0.0:
    raw = model.generate_n(messages, n=n_resamples, temperature=TEMPERATURE)
else:
    raw = model.generate_with_steering_n(
        messages=messages, layer=layer,
        steering_hook=hook, n=n_resamples, temperature=TEMPERATURE,
    )
responses = [parse_response(r) for r in raw]
done += n_resamples
```

### `run_multi_layer_steering` (lines ~314-332)

Same pattern but with the multi-layer branch:

**After:**
```python
if not hook_list or coef == 0.0:
    raw = model.generate_n(messages, n=n_resamples, temperature=TEMPERATURE)
elif len(hook_list) == 1:
    raw = model.generate_with_steering_n(
        messages=messages, layer=hook_list[0][0],
        steering_hook=hook_list[0][1], n=n_resamples, temperature=TEMPERATURE,
    )
else:
    raw = model.generate_with_multi_layer_steering_n(
        messages=messages, layer_hooks=hook_list,
        n=n_resamples, temperature=TEMPERATURE,
    )
responses = [parse_response(r) for r in raw]
done += n_resamples
```

### `run_random_direction_control` (lines ~400-415)

Same pattern as `run_single_layer_steering`.

### Update progress counter

The `done` counter previously incremented per-resample. Now it increments by `n_resamples` per cell. The rate/ETA calculation stays the same since it's based on total trials.

## Step 3: Verify phase 1 results exist

Check that `results/phase1_L31.json` exists and is valid (non-empty, parses as JSON). Do NOT re-run phase 1.

## Step 4: Run phases 2–4

Run phases 2, 3, and 4 sequentially using the modified script:

```bash
python -u scripts/fine_grained/run_experiment.py --phase 2 2>&1 | tee results/phase2_log.txt
python -u scripts/fine_grained/run_experiment.py --phase 3 2>&1 | tee results/phase3_log.txt
python -u scripts/fine_grained/run_experiment.py --phase 4 2>&1 | tee results/phase4_log.txt
```

Commit results after each phase completes.

## Step 5: Write the report

Once all phases are complete, write `fine_grained_report.md` with analysis covering the questions from the parent spec. Generate plots to `assets/`.

## Expected timing

With batched sampling (~5-8x speedup on resample loop):
- Phase 2 (L49+L55 single-layer): ~540k trials → ~2-3 hours (was ~13 hours serial)
- Phase 3 (multi-layer): ~270k trials → ~1-2 hours
- Phase 4 (random controls): ~180k trials → ~0.5-1 hour
- Total phases 2-4: ~4-6 hours (was ~20+ hours serial)
