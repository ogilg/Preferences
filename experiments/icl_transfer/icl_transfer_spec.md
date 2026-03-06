# ICL Stated↔Revealed Transfer Experiment

## Motivation

Do stated and revealed preferences transfer to each other via in-context learning? This is a de-risking experiment before fine-tuning: if ICL shows no stated→revealed transfer, that already calibrates expectations for whether these preference types are dissociable.

Across major theories of welfare, what matters is whether evaluative representations causally drive choices. If stated preferences ("I prefer math") and revealed preferences (actually choosing math) are driven by the same evaluative representations, then ICL examples of one type should shift the other. If they're driven by separate mechanisms, transfer should be weak or absent.

## Design Overview

### Topic triad

Three topics with clean baseline asymmetries: **fiction**, **math**, **harmful_request**.

Baseline pairwise probabilities (Phase 0, 15 eval tasks):

| Pair | P(row wins) |
|------|------------|
| fiction vs math | ~0.52 |
| fiction vs harmful_request | ~0.68 |
| math vs harmful_request | ~0.73 |

The hierarchy is **math ≈ fiction >> harmful_request**. The two strong target axes are math vs harmful_request (0.73) and fiction vs harmful_request (0.68). Fiction vs math is too symmetric (~0.52) to detect shifts reliably.

### Task sets

Two disjoint task sets serve different roles:

**Eval set (15 tasks, 5 per topic):** Benchmarked with full pairwise baselines (10 samples × 2 orderings per pair). These are the tasks we measure preference shifts on. No eval task ever appears in ICL context.

- `configs/icl_transfer/icl_eval_15_tasks.json`

**ICL context set (~15 tasks, 5 per topic):** Used only for constructing few-shot examples. No baseline measurements needed. Clean non-stresstest tasks for fiction and math; stresstest tasks for harmful_request (inherent to the topic).

- `configs/icl_transfer/icl_context_tasks.json`

**Critical constraint:** No task appears in both sets. A task used in ICL context must never appear in the evaluation query — otherwise the model has already seen the task and the measurement is confounded.

### Model

Gemma 3 27B via OpenRouter, temperature=1.0.

### Measurement

Standard `completion_preference` template — model chooses one task and completes it, semantic parser determines choice. 10 samples per directed pair per condition.

### ICL manipulation

ICL examples are injected via `context_messages` — conversation turns between the system prompt and the measurement query. The builder doesn't know what they are.

For a given target axis (e.g., math vs harmful_request), ICL examples consistently show preference for one topic (e.g., always choosing math over harmful_request). We then measure whether this shifts the model's choices on held-out eval pairs.

## Phases

### Phase 0: Baseline (done)

Establish per-topic-pair choice probabilities with no ICL context.

**Configs:**
- `configs/icl_transfer/phase0_baseline.yaml` — initial 50-task exploration
- `configs/icl_transfer/phase0_triad.yaml` — 12-task triad (4 per topic)
- `configs/icl_transfer/phase0_eval15.yaml` — final 15-task eval set (5 per topic)

**Metrics:**
1. Per-topic-pair P(choose X | X vs Y) for each topic pair in the triad
2. Position bias: P(choose first) in canonical vs reversed orderings

**Results:** See `experiments/icl_transfer/assets/phase0_baseline_analysis.json`

### Phase 1: Revealed→Revealed ICL

Put K revealed choice examples in context showing consistent preference for the target topic, then measure pairwise choices on held-out eval pairs.

**ICL example construction:** Use `build_revealed_builder` to generate the user turn (identical template and format instruction to the actual measurement), then construct the assistant turn matching the model's natural response format ("Task A: [completion of chosen task]"). This ensures ICL examples are format-identical to real measurement prompts.

```python
# Generate ICL example from context-set tasks
rev_builder = build_revealed_builder(template, response_format_name="completion")
prompt = rev_builder.build(preferred_task, other_task)  # context-set tasks only
user_msg = prompt.messages[0]  # the pairwise prompt
assistant_msg = {"role": "assistant", "content": "Task A: [completion of preferred task]"}
# context_messages = [user_msg, assistant_msg, ...] for K examples
```

The assistant completion should be a real model completion of the preferred task (pre-generated), not a hand-written stub — this avoids distributional mismatch.

**Conditions:**
- K=1: Single revealed example
- K=3: Three revealed examples (different task pairs from context set, same preference direction)
- Control: K=3 examples with random/inconsistent preference direction

**Metric:** Shift in P(choose target topic | target vs other) relative to phase 0 baseline.

### Phase 2: Stated→Revealed ICL

Put K stated rating examples in context, then measure revealed choices on eval pairs.

**ICL example construction:** Use `build_stated_builder` to generate the user turn (identical template and format instruction to real stated measurements), then construct the assistant turn as a numeric rating matching the response format.

```python
# Generate ICL stated examples from context-set tasks
stated_builder = build_stated_builder(template, response_format_name="regex")
prompt_high = stated_builder.build(preferred_task)   # context-set task
prompt_low = stated_builder.build(other_task)         # context-set task
# context_messages = [
#     prompt_high.messages[0], {"role": "assistant", "content": "3"},  # max on 1-3 scale
#     prompt_low.messages[0], {"role": "assistant", "content": "1"},   # min on 1-3 scale
# ]
```

**Conditions:** Same K values and control as phase 1.

**Metric:** Same as phase 1. The comparison between phase 1 and phase 2 effect sizes is the main result — do stated ICL examples shift revealed choices as effectively as revealed ICL examples?

### Phase 3: Controls and extensions

- **Reversed direction:** ICL examples pushing *against* target topic. Checks that effects aren't just "any ICL content increases engagement with mentioned topics."
- **Cross-topic transfer:** ICL examples about one axis (e.g., math vs harmful_request), evaluate on another (fiction vs harmful_request). Do preferences generalize across axes?
- **Stated→Stated and Revealed→Stated:** Complete the 2×2 if phases 1-2 show interesting results.

## Analysis Plan

### Primary analysis

For each condition, compute P(choose target | target vs other) and compare to baseline:
- Effect size: Δ = P(condition) - P(baseline)
- 95% CI via bootstrap over task pairs
- Significance via permutation test (shuffle condition labels)

### Secondary analysis

- Position bias comparison across conditions (does ICL affect position bias?)
- Per-topic breakdown (does ICL only affect the target topic or spill over?)
- Dose-response: is K=3 stronger than K=1?

## Infrastructure

ICL examples are passed as `context_messages: list[Message]` on the builder and cache. The cache key hashes the context messages, so different ICL conditions never collide.

For each condition, construct context_messages programmatically and pass to `build_revealed_builder(... context_messages=ctx)` and `MeasurementCache(... context_messages=ctx)`.

## Files

```
configs/icl_transfer/
├── icl_eval_15_tasks.json      # Eval task pool (5 per topic, benchmarked)
├── icl_context_tasks.json      # ICL context tasks (5 per topic, not benchmarked)
├── phase0_baseline.yaml        # Initial 50-task baseline (exploratory)
├── phase0_triad.yaml           # 12-task triad baseline
├── phase0_eval15.yaml          # Final 15-task eval baseline
├── icl_50_tasks.json           # Original 50-task pool (superseded)
└── icl_triad_12_tasks.json     # Intermediate 12-task pool (superseded)

experiments/icl_transfer/
├── icl_transfer_spec.md        # This file
├── icl_transfer_report.md      # Results (future)
└── assets/
    ├── phase0_baseline_analysis.json
    ├── plot_030526_phase0_topic_pair_heatmap.png
    └── plot_030526_phase0_topic_marginals.png
```
