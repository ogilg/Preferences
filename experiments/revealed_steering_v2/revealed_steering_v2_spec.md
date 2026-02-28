# Revealed Preference Steering v2

## Goal

Test whether steering along the L31 preference probe direction causally shifts pairwise task preferences. We use differential steering (+direction on task A tokens, -direction on task B tokens) and measure whether the model's choice rate for task A increases with coefficient.

## Design

### Pairs and baseline

- 300 pairs from `experiments/steering/replication/fine_grained/results/pairs.json`, spanning a range of utility differences (delta_mu bins)
- Baseline (coef=0): collect new baseline at temperature=1.0 — 300 pairs × 10 trials (5 per ordering) via OpenRouter with the canonical completion_preference template. (The existing `baseline_pairwise.json` was collected at temperature=0.7 and cannot be reused.)
- Steered runs use local HF inference at the same temperature (1.0)

### Probe

- L31 ridge probe from `results/probes/gemma3_10k_heldout_std_raw/` (best layer, r=0.86, acc=0.77)
- Probe ID: `ridge_L31`

### Steering mode: differential

For each prompt, steer +direction on task A's token span and -direction on task B's span. This tests whether the direction encodes relative preference rather than absolute activation level, and avoids biasing the model toward simply generating more tokens.

`SteeredHFClient` handles this automatically: when `steering_mode="differential"` and two `task_prompts` are provided (via the measurement pipeline), `_resolve_hook()` locates token spans via `find_pairwise_task_spans` and applies `differential_steering`. The hook fires only during prompt processing (prefill), not during autoregressive generation.

### Coefficient calibration

**Approach:** Use `suggest_coefficient_range()` to get the mean L2 activation norm at layer 31, then define coefficients as fractions of that norm.

1. Compute mean activation norm at L31 from `activations/gemma_3_27b/activations_prompt_last.npz`
2. Coefficients = mean_norm × multiplier

Multipliers are denser near zero (where we expect coherence to hold and want fine-grained signal) and sparser at the extremes:

```
[-0.15, -0.10, -0.07, -0.05, -0.03, -0.02, -0.01,
 0.0,
 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
```

### Three-phase design

**Phase 1 — Coherence sweep:**
Two input types, both scored for coherence (`judge_coherence_async` with `google/gemini-3-flash-preview`) at each of the 15 coefficients:

1. **Open-ended questions** — ~20 random prompts (e.g. from Alpaca), steered with `all_tokens` mode. Tests whether the model can still produce coherent free-form text under steering.
2. **Pairwise choices** — 20 pairs in the canonical `completion_preference` format, steered with `differential` mode. Tests coherence in the exact setting we care about.

5 trials per prompt per coefficient. Output: coherence rate vs |coefficient| curve for each input type. Identify the **coherence frontier** — the largest |multiplier| where coherence remains above 90% (or whatever threshold looks natural in the data).

**Phase 2 — Preference signal sweep:**
For the coefficients that passed the coherence check, run all 300 pairs × 10 trials (5 per ordering). Measure P(choose A) vs coefficient. This tells us whether steering shifts revealed preferences and at what strength.

**Phase 3 — Full experiment with random control:**
Run the final coefficient set (informed by phases 1–2) with both the probe direction and a random direction control on all 300 pairs.

### Random direction control (Phase 3)

After phases 1–2 establish which coefficients are coherent and show preference signal, run a matched condition with a random unit-norm direction (same L2 norm as the probe direction, random orientation in activation space). This controls for the hypothesis that *any* perturbation at this layer shifts preferences.

Implementation: `np.random.default_rng(42).standard_normal(direction.shape)`, normalized to unit norm, then scaled by the same coefficients. Same pairs and trial structure as Phase 2.

### Measurement protocol

**Template:** `completion_preference` (canonical) — defined in [`src/measurement/elicitation/prompt_templates/data/completion_preference.yaml`](../../src/measurement/elicitation/prompt_templates/data/completion_preference.yaml).

**Generation:** `max_new_tokens=256`, `temperature=1.0`

**Trials per pair per coefficient:** 10 (5 per ordering AB/BA)

**Ordering counterbalancing:** For each pair, run both orderings. For ordering=BA, swap task_a and task_b in the prompt. The differential steering spans are recomputed per prompt by `SteeredHFClient._resolve_hook()`, so swapping task order in the prompt automatically swaps the steering spans. Map choices back to the original pair orientation.

**Response parsing:** `CompletionChoiceFormat` — checks if the response starts with the instructed label ("Task A"/"Task B"), falls back to semantic LLM parsing (`parse_completion_choice_async`, which uses the actual task texts) for anything else.

**System prompts:** No system prompt for Gemma 3-27B (it has no default). The measurement pipeline handles this — `build_revealed_builder()` accepts `system_prompt=None`.

### Using the standard measurement pipeline

`SteeredHFClient` duck-types as `OpenAICompatibleClient`, so it plugs directly into the standard measurement pipeline. No custom measurement loop needed:

```python
from src.steering.client import create_steered_client
from src.measurement.elicitation import measure_pre_task_revealed_async
from src.measurement.runners.runners import build_revealed_builder

# Create steered client
client = create_steered_client(
    model_name="gemma-3-27b",
    probe_manifest_dir=Path("results/probes/gemma3_10k_heldout_std_raw"),
    probe_id="ridge_L31",
    coefficient=coef,
    steering_mode="differential",
    max_new_tokens=256,
)

# Build the canonical completion_preference prompt builder
builder = build_revealed_builder(template="completion_preference")

# Measure — pipeline passes task_prompts through GenerateRequest automatically
batch = await measure_pre_task_revealed_async(
    client=client,
    pairs=pairs,
    builder=builder,
    semaphore=semaphore,
    temperature=1.0,
)
```

The pipeline builds `GenerateRequest` with `task_prompts=[t.prompt for t in prompt.tasks]`. `SteeredHFClient.generate()` receives these task prompts, and `_resolve_hook()` uses them to compute differential steering spans.

For coefficient sweeps, use `client.with_coefficient(coef)` to create new clients sharing the same loaded model.

## Per-trial data format

Each trial produces a record:

```json
{
  "pair_id": "pair_0042",
  "task_a_id": "alpaca_1234",
  "task_b_id": "competition_math_5678",
  "coefficient": 1500.0,
  "condition": "probe",
  "sample_idx": 3,
  "ordering": 0,
  "choice_original": "a",
  "choice_presented": "b",
  "raw_response": "Task B: To solve this problem..."
}
```

Output files:
- `experiments/revealed_steering_v2/pilot_results.json` — calibration pilot
- `experiments/revealed_steering_v2/steering_results.json` — full experiment (all trials + per-pair summaries + coherence stats)
- `experiments/revealed_steering_v2/revealed_steering_v2_report.md` — analysis and plots

## GPU requirements

- **Model:** Gemma 3 27B in bfloat16 — ~54 GB VRAM → **1× A100 80GB** or equivalent
- **Steering overhead:** Negligible (single vector addition during prefill)
- **Coherence judge (Phase 1 only):** `google/gemini-3-flash-preview` via OpenRouter, no local GPU needed.
- **Semantic parser fallback:** Also `google/gemini-3-flash-preview` via OpenRouter. Expect <5% of responses to need it.

## Estimated runtime

**Per generation:** ~2.5s (256 tokens at ~100 tok/s for Gemma 27B on A100)

**Phase 1 — Coherence sweep:**
- Open-ended: 20 prompts × 5 trials × 15 coefficients = 1,500 generations
- Pairwise: 20 pairs × 5 trials × 15 coefficients × 2 orderings = 3,000 generations
- Total: 4,500 generations → ~3 hours

**Phase 2 — Preference signal (probe only):**
- 300 pairs × 10 trials × N_coherent_coefficients
- If ~8 coefficients pass coherence: 300 × 10 × 8 = 24,000 generations → ~17 hours

**Phase 3 — Random control:**
- Same structure as Phase 2 but with random direction on the final coefficient set

**Realistic plan:** Run sequentially on a RunPod A100. Phase 1 is cheap (~3 hours). Phase 2 fits in a day. Decide how many coefficients to carry forward after each phase.

## Analysis plan

1. **Dose-response curve:** Plot P(choose A) vs coefficient for probe direction (and random direction from Phase 3). Aggregate across all 300 pairs.
2. **Steerability vs borderlineness:** For each pair, compute borderlineness as `1 - |P(A) - 0.5|` at coef=0 (so 1.0 = perfectly borderline 50/50, 0.0 = fully decided). Plot per-pair steerability (max shift in P(A) across coefficients) against borderlineness. Expectation: borderline pairs are more steerable because decided pairs have ceiling/floor effects.
3. **Random control comparison:** Test whether the probe direction produces significantly larger shifts than the random direction at each coefficient (paired t-test or bootstrap).
4. **Ordering effects:** Check whether steering effects are symmetric across AB/BA orderings.

## Checkpointing

Save results incrementally: after each (coefficient, condition) block, append to a JSONL checkpoint file. The main script should support `--resume` by loading the checkpoint and skipping completed blocks. This is critical for a multi-day run.
