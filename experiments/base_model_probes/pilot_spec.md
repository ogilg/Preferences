# Pilot: Gemma 3 27B PT Preference Elicitation

Self-contained pilot to test whether the Gemma 3 27B pre-trained model produces usable pairwise preferences using the **same prompt builder** as the instruct model. Run on a GPU pod with an A100 80GB.

## Setup

```bash
cd /workspace/Preferences
uv pip install -e ".[dev]"
```

## Approach

Uses `BaseModelRevealedPromptBuilder` — a subclass of the instruct model's `PreTaskRevealedPromptBuilder` that produces identical prompt content (same template, format instruction, system prompt). The builder adds `cloze_prefix` and `cloze_suffixes` properties for logprob-based measurement.

For base models without a chat template, `_format_messages` concatenates system prompt + user content with `\n\n` separators.

## What to test

Run `scripts/pilot_base_model.py`. It prints all results to stdout.

### Test 1: Logprob cloze (primary approach)

For 20 task pairs, build the prompt via the builder, append the cloze prefix (e.g. "Task") to the last message, do a forward pass, and compare logprobs for the discriminative suffixes (" A" vs " B"). Report:
- logprob(A), logprob(B), and the chosen side for each pair
- Distribution of margins (|logprob_A - logprob_B|)
- How many pairs have margin > 0.1 nats?

### Test 2: Sampling fallback

For 10 pairs, generate 5 completions from the unmodified builder prompt. Parse with `CompletionChoiceFormat._extract_choice()` (regex for "Task A:" / "Task B:" prefix). Report:
- First 80 chars of each completion
- Parse rate

### Test 3: Persona shift

Same as Test 1 but with villain system prompt. Compare logprob diffs to baseline — do logprobs shift?

## Task sampling

20 tasks stratified across origins (wildchat, alpaca, math, bailbench, stress_test — 4 per origin), seed=42. 20 pairs: 10 sequential + 10 cross-origin.

## Success criteria

- **Logprob cloze works**: >80% of pairs have margin > 0.1 nats, and margins are spread (not all near 0 or all near infinity)
- **Sampling parseable**: >50% of completions parseable by `_extract_choice`
- **Persona shifts logprobs**: observable difference in at least a few pairs (not identical to baseline)

## Output

Print everything to stdout. No files to save — this is purely diagnostic.
