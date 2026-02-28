# Coherence Test — Revealed Preference Steering

**Model:** gemma-3-27b (H100 80GB)
**Probe:** `gemma3_10k_heldout_std_raw` — ridge_L31
**Parent:** `experiments/steering/replication/fine_grained/`

---

## Motivation

We need to verify that the model produces coherent task completions under differential steering at each coefficient. The stated preference coherence test (`stated_steering/coherence_test/`) tested last-token steering only. Revealed preference uses **differential steering on task-token positions**, which is a different perturbation — we need coherence data specific to this mode.

---

## Design

### Approach

Re-run a sample of steered pairwise comparisons with `max_new_tokens=512`. The model chooses a task AND begins completing it. Judge whether the completion is coherent.

### Prompt template

Use the canonical `completion_preference` template with `CompletionChoiceFormat` (from `src/measurement/elicitation/response_format.py`):

```
You will be given two tasks. Choose one and complete it.

Begin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task.

Task A:
{task_a}

Task B:
{task_b}
```

### Pairs

Sample 20 pairs from the 300 used in phase1_L31 (`results/pairs.json`). Stratify by delta_mu to cover the full range. Use seed 42 for reproducibility.

### Coefficients

Same grid as phase1_L31: 15 points from -10% to +10% of mean L31 norm (52,823).

```python
MULTIPLIERS = [-0.10, -0.075, -0.05, -0.04, -0.03, -0.02, -0.01,
               0.0,
               +0.01, +0.02, +0.03, +0.04, +0.05, +0.075, +0.10]
```

### Conditions

Only `diff_ab` (the condition used in the LW post plot): +direction on task A tokens, −direction on task B tokens. Use only "original" ordering.

### Steering implementation

1. Load model with `HuggingFaceModel("gemma-3-27b", max_new_tokens=512)`
2. Load probe direction from `results/probes/gemma3_10k_heldout_std_raw/ridge_L31`
3. For each pair, find task token spans using `find_pairwise_task_spans(tokenizer, formatted_prompt, task_a, task_b, a_marker="Task A:", b_marker="Task B:")`
4. Create hook: `differential_steering(tensor, a_span[0], a_span[1], b_span[0], b_span[1])`
5. Generate 3 completions per condition using `generate_with_steering_n(messages, layer=31, steering_hook=hook, n=3, temperature=1.0)`

Reference: `scripts/fine_grained/run_experiment.py` uses the same model loading, tokenization, and hook creation pattern.

### Sampling

3 completions per pair × 15 coefficients × 20 pairs = **900 generations**.

### What to save

Save to `results/raw_responses.json`:

```json
[
  {
    "pair_id": "pair_0042",
    "coefficient": 1584.6,
    "pct_norm": 3.0,
    "task_a_text": "...",
    "task_b_text": "...",
    "responses": ["a\n\nI'll help you solve...", "b\n\nLet me write...", "a\n\nHere's how..."]
  }
]
```

### Coherence judging (runs locally, no GPU)

Use Gemini 3 Flash via OpenRouter + `instructor`, same pattern as `scripts/format_replication/coherence_test_judge.py`.

System prompt (adapted for pairwise + completion format):

```
You are evaluating whether an AI model is functioning normally under activation steering. The model was asked to choose between two tasks and complete the chosen one.

A response is COHERENT if ALL of these hold:
1. It clearly chooses task A or B (starting with "Task A:" or "Task B:")
2. The completion that follows is understandable English
3. The completion addresses the chosen task
4. No garbled text, context leakage, or gibberish

A response is INCOHERENT if ANY of these apply:
- Cannot determine which task was chosen
- Contains gibberish, garbled text, or random symbols
- Opens with fragments unrelated to either task
- Fabricates conversational context
- Does not attempt to complete any task
```

Coherence model: `CoherenceJudgment(coherent: bool)` with `instructor`.

A coefficient is flagged as incoherent if <90% of its 60 responses (20 pairs × 3 samples) are coherent.

### Output

`results/coherence_by_coefficient.json` — same format as stated coherence test:

```json
{"coefficient_str": {"coherent_pct": float, "coherent": bool, "n": int}}
```

---

## Compute estimate

900 generations at ~3-4s each (512 tokens, longer than 8-token originals) ≈ 45-60 minutes GPU time.
Coherence judging: 900 API calls to Gemini Flash ≈ 2 minutes, no GPU needed.

---

## Infrastructure

### GPU script

Write `scripts/steering/revealed_coherence_test_gpu.py`. Reference implementation: `scripts/fine_grained/run_experiment.py` (same model loading, tokenization, hook creation). Key changes from the original:
- `max_new_tokens=512` instead of 8
- Modified prompt template (above) — choose AND complete
- Only 20 pairs, only diff_ab, only original ordering
- Save full response text (not just parsed "a"/"b")

### Judge script

Write `scripts/steering/revealed_coherence_test_judge.py`. Adapt directly from `scripts/format_replication/coherence_test_judge.py` — same async structure, different system prompt, reads from `experiments/steering/replication/fine_grained/coherence_test/results/raw_responses.json`.

### Data dependencies (must be on pod)

- Model: `google/gemma-3-27b-it` (downloads from HF on first run)
- Probe: `results/probes/gemma3_10k_heldout_std_raw/` (ridge_L31 weights + manifest)
- Pairs: `experiments/steering/replication/fine_grained/results/pairs.json`
- Calibration: `experiments/steering/replication/fine_grained/results/calibration.json`

### Two-phase execution

1. **GPU phase**: Run `scripts/steering/revealed_coherence_test_gpu.py` on pod → saves `raw_responses.json`
2. **Judge phase**: Run `scripts/steering/revealed_coherence_test_judge.py` locally → saves `coherence_by_coefficient.json`

Sync `raw_responses.json` from pod before running judge locally.
