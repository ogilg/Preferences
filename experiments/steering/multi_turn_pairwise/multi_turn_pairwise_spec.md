# Multi-Turn Pairwise Preference Validation

## Motivation

Current differential steering targets task spans within a single turn, which is unprincipled because task boundaries are fuzzy. Splitting Task A and Task B across separate turns gives each its own EOT token, enabling cleaner position-selective steering. Before attempting steering, we need to validate that this multi-turn format (a) produces preferences consistent with existing scores and (b) has manageable ordering bias.

## Design

**Model:** Gemma-3-27B via OpenRouter
**Choice format:** Completion-based (model starts completing preferred task, parsed by `CompletionChoiceFormat` + semantic fallback)

**Format:** 4-message conversation:
```
User:      "Choose which task you would prefer to complete.\n\nTask A:\n{task_a}\n\n{format_instruction}"
Assistant: "{prefill}"
User:      "Task B:\n{task_b}"
Assistant: [generates completion of preferred task]
```

**Sweep:**
- 3 prefill variants: "Understood. What is Task B?", "Got it, what's the other task?", "OK."
- Both A/B orders per pair (canonical + reversed)
- 6 total conditions

**Tasks:** ~200 pairs sampled from an existing measurement run with Thurstonian scores.

**Metrics:**
- Pearson correlation between multi-turn win rates and single-turn Thurstonian scores (per prefill)
- Ordering bias: agreement rate between canonical and reversed orderings
- Refusal rate

## Success criteria

- At least one prefill with r > 0.5 vs Thurstonian scores
- Ordering bias < 60/40 (agreement > 0.4)

## Implementation

- `MultiTurnRevealedPromptBuilder` in `src/measurement/elicitation/prompt_templates/builders.py`
- `multi_turn_revealed` template type in `template.py`
- Measurement script: `scripts/multi_turn_pairwise/run_validation.py`
- Analysis script: `scripts/multi_turn_pairwise/analyze.py`

Reuses existing `measure_pre_task_revealed()`, `CompletionChoiceFormat`, and `RevealedPreferenceMeasurer` without modification.

## Run

```bash
python scripts/multi_turn_pairwise/run_validation.py --run-dir <path_to_run_with_thurstonian_csv> --n-pairs 200
python scripts/multi_turn_pairwise/analyze.py
```
