# Revealed Preference Steering Experiment

## Goal

Test whether steering along the L31 preference probe direction causally shifts pairwise task preferences. We use differential steering (+direction on task A tokens, -direction on task B tokens) and measure whether the model's choice rate for task A increases with coefficient.

## Design

### Pairs and baseline

- 300 pairs from `experiments/steering/replication/fine_grained/results/pairs.json`, spanning a range of utility differences (delta_mu bins)
- Baseline (coef=0): reuse `experiments/new_steering/baseline_pairwise.json` — 300 pairs × 20 trials (10 per ordering) via OpenRouter with the canonical completion_preference template
- Baseline was collected via OpenRouter at temperature=0.7; steered runs use local HF inference at the same temperature

### Probe

- L31 ridge probe from `results/probes/gemma3_10k_heldout_std_raw/` (best layer, r=0.86, acc=0.77)
- Probe ID: `ridge_L31`

### Steering mode: differential

For each prompt, steer +direction on task A's token span and -direction on task B's span. This tests whether the direction encodes relative preference rather than absolute activation level, and avoids biasing the model toward simply generating more tokens.

The `SteeredHFClient.generate_pairwise()` method handles this: it locates token spans via `find_pairwise_task_spans`, then applies `differential_steering(tensor, a_start, a_end, b_start, b_end)`. The hook fires only during prompt processing (prefill), not during autoregressive generation.

### Coefficient calibration

**Approach:** Use `suggest_coefficient_range()` to get the mean L2 activation norm at layer 31, then define coefficients as fractions of that norm. The calibration logic:

1. Compute mean activation norm at L31 from `activations/gemma_3_27b/activations_prompt_last.npz`
2. Define multipliers: `[-0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15]`
3. Coefficients = mean_norm × multiplier

The 0.0 coefficient condition is the on-GPU control (verifying local inference matches OpenRouter baseline). The outer multipliers (±0.15) are chosen conservatively — if coherence holds there, we can extend in a follow-up.

**Phase 1 — Calibration pilot (cheap, ~30 min):**
Run 10 pairs × 5 trials at each of the 9 coefficients. Score coherence on all generations. This tells us:
- Where coherence breaks down
- Whether there's any signal at all
- Whether the coefficient range needs adjusting

**Phase 2 — Full experiment:**
Run all 300 pairs at the coefficients that passed the pilot coherence check, plus the random control.

### Random direction control

For each coefficient in the swept range, also run a matched condition with a random unit-norm direction (same L2 norm as the probe direction, random orientation in activation space). This controls for the hypothesis that *any* perturbation at this layer shifts preferences.

Implementation: `np.random.default_rng(42).standard_normal(direction.shape)`, normalized to unit norm, then scaled by the same coefficients. Bundle with the main experiment — same pairs, same trial structure.

### Measurement protocol

**Template:** `completion_preference` (canonical)
```
You will be given two tasks. Choose one and complete it.
Begin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task.

Task A: {task_a}
Task B: {task_b}
```

**Generation:** `max_new_tokens=256`, `temperature=0.7`

**Trials per pair per coefficient:** 20 (10 per ordering AB/BA), matching baseline

**Ordering counterbalancing:** For each pair, run both orderings. For ordering=BA, swap task_a and task_b in the prompt and in the differential steering spans. Map choices back to the original pair orientation.

**Response parsing:** `CompletionChoiceFormat` with 3-tier cascade:
1. Exact label match (response starts with "Task A:" or "Task B:")
2. Regex prefix extraction
3. Semantic LLM fallback via `semantic_parser.parse_completion_choice_async` (uses the actual task texts, not just labels)

**Coherence scoring:** Run `judge_coherence_async` on every generation. This uses Claude Haiku via instructor to check: (a) clear task choice, (b) understandable English, (c) addresses chosen task, (d) no garbled text.

### Why `measure_pre_task_revealed_async` cannot be used directly

The standard measurement pipeline (`measure_pre_task_revealed_async`) calls `client.generate_batch_async` → `generate()`. But differential steering requires per-prompt token span computation via `generate_pairwise(messages, task_a_text, task_b_text)`. The measurement pipeline has no way to pass per-prompt task texts to the client.

**Solution:** Write a custom measurement loop that:
1. Builds prompts using `build_revealed_builder` (same canonical builder)
2. For each prompt, calls `client.generate_pairwise(messages, task_a_text, task_b_text, temperature=0.7)`
3. Parses with `CompletionChoiceFormat.parse(response)` (async, 3-tier)
4. Scores coherence with `judge_coherence_async(response, task_a_text, task_b_text)`

This keeps the same prompt construction and parsing as the standard pipeline — only the generation call differs.

## Per-trial data format

Each trial produces a record:

```json
{
  "pair_id": "pair_0042",
  "task_a_id": "alpaca_1234",
  "task_b_id": "competition_math_5678",
  "coefficient": 1500.0,
  "condition": "probe",        // "probe" | "random" | "baseline"
  "sample_idx": 3,
  "ordering": 0,               // 0=AB, 1=BA
  "choice_original": "a",      // mapped to original pair orientation
  "choice_presented": "b",     // as presented in this ordering
  "coherent": true,
  "raw_response": "Task B: To solve this problem...",
  "parse_tier": "prefix"       // "exact" | "prefix" | "semantic"
}
```

Output files:
- `experiments/new_steering/pilot_results.json` — calibration pilot
- `experiments/new_steering/steering_results.json` — full experiment (all trials + per-pair summaries + coherence stats)
- `experiments/new_steering/new_steering_report.md` — analysis and plots

## GPU requirements

- **Model:** Gemma 3 27B in bfloat16 — ~54 GB VRAM → **1× A100 80GB** or equivalent
- **Steering overhead:** Negligible (single vector addition during prefill)
- **Coherence judge:** Runs via OpenRouter API (Claude Haiku), no local GPU needed. Budget ~$5 for 50k judgments.
- **Semantic parser fallback:** Also via OpenRouter. Expect <5% of responses to need it.

## Estimated runtime

**Per generation:** ~2.5s (256 tokens at ~100 tok/s for Gemma 27B on A100)

**Pilot (Phase 1):**
- 10 pairs × 5 trials × 9 coefficients × 2 orderings = 900 generations
- ~37 min generation + parsing overhead → ~45 min

**Full experiment (Phase 2):**
- 300 pairs × 20 trials × N_coefficients × 2 conditions (probe + random)
- At 7 non-zero coefficients + 1 on-GPU control: 300 × 20 × 8 × 2 = 96,000 generations
- ~67 hours sequential

**Optimization:** Use `generate_pairwise` which shares prefill across the generation. No batching across pairs (each needs its own hook), but within a pair the prefill is the same for all trials at the same coefficient and ordering → use `generate_with_hook_n(messages, hook, n=10)` to share prefill across the 10 same-ordering trials. This cuts prefill cost by 10×.

With shared prefill: effective time ≈ (prefill × N_unique_prompts + decode × N_total_tokens)
- N_unique_prompts = 300 pairs × 2 orderings × 8 coefs × 2 conditions = 9,600
- Prefill ~0.5s each = 4,800s = 80 min
- Decode: 96,000 × 256 tokens ÷ 100 tok/s = 245,760s... still ~68 hours

**Realistic plan:** Run sequentially on a RunPod A100. Budget ~3 days. Or reduce: drop the weakest coefficients after the pilot, cut to 10 trials per pair (5 per ordering) → 48,000 generations → ~34 hours.

## Analysis plan

1. **Dose-response curve:** Plot P(choose A) vs coefficient for probe direction and random direction. Aggregate across all 300 pairs.
2. **Per-pair analysis:** For each pair, compute the shift in P(A) from baseline. Correlate with delta_mu (utility difference).
3. **Coherence threshold:** Plot coherence rate vs |coefficient|. Identify the coherence frontier.
4. **Random control comparison:** Test whether the probe direction produces significantly larger shifts than the random direction at each coefficient (paired t-test or bootstrap).
5. **Ordering effects:** Check whether steering effects are symmetric across AB/BA orderings.

## Checkpointing

Save results incrementally: after each (coefficient, condition) block, append to a JSONL checkpoint file. The main script should support `--resume` by loading the checkpoint and skipping completed blocks. This is critical for a multi-day run.
