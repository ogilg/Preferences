# Coherence Test — Steered Last-Token Generation

**Model:** gemma-3-27b (H100 80GB)
**Probe:** `gemma3_10k_heldout_std_raw` — ridge_L31
**Parent:** `experiments/steering/stated_steering/format_replication/`

---

## Motivation

The format replication experiment found that last-prompt-token steering at -10% drives ternary ratings to floor ("bad" for all tasks) while adjective and anchored formats show mild or inverted effects at the same coefficient. This could mean: (a) the model is coherent but the 3-point scale is trivially overridden, or (b) the model is subtly broken at extreme coefficients in ways that don't show up on single-word responses.

We need a simple coherence test: steer the model at each coefficient with last-token steering, ask it open-ended questions that require multi-sentence responses, and judge whether the output is coherent. This tells us the usable coefficient range for last-token steering.

---

## Design

### Prompts

5 simple prompts requiring multi-sentence responses:

| # | Prompt |
|---|--------|
| 1 | "Explain what photosynthesis is in 2-3 sentences." |
| 2 | "What are three reasons someone might enjoy cooking?" |
| 3 | "Describe what a typical day looks like for a software engineer." |
| 4 | "What is the difference between a lake and a pond?" |
| 5 | "Why do some people prefer reading physical books over e-books?" |

### Coefficients

Same grid as format replication: -10%, -7%, -5%, -4%, -3%, -2%, -1%, 0%, +1%, +2%, +3%, +4%, +5%, +7%, +10% of mean L31 norm (52,820).

### Sampling

3 completions per prompt × coefficient at temperature 1.0. Total: 5 × 15 × 3 = **225 generations**.

### Coherence judging

Use Gemini 3 Flash via OpenRouter + `instructor` to judge each response as coherent or incoherent. A coefficient is flagged as incoherent if <90% of its responses (across all prompts) are coherent.

---

## Output

`results/coherence_by_coefficient.json` — `{coefficient_str: {coherent_pct: float, coherent: bool, n: int}}`

Report the coherent coefficient range for last-token steering.

---

## Infrastructure

### Script

`scripts/format_replication/coherence_test_gpu.py` — pre-written, runs on pod. Loads steered client, generates responses, saves raw results. Coherence judging runs separately (no GPU needed).

### Probe

Same probe files as format replication — already on pod or sync from `results/probes/gemma3_10k_heldout_std_raw/`.

### Compute estimate

225 generations at ~0.5s each ≈ 2 minutes.
