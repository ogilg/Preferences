# Role Prompt Enrichment: Exploration

## Goal

Find system prompt strategies that produce strong, interpretable preference shifts. This is exploratory — short feedback loops, many small tests, hypothesize-then-test.

## Context

V2 used 1-2 sentence role prompts. 9/10 broad roles shifted preferences in expected directions, but safety_advocate failed entirely, and narrow roles lacked specificity. The prompts are short and there's a lot of room to try things.

## Approach

No fixed experimental matrix. Instead: form a hypothesis, test it cheaply (~250 pairs, ~25 tasks, 1 resample), look at results, form the next hypothesis.

### Starting hypotheses

1. **More detail = stronger shift**: Elaborating on the role (backstory, values, specific interests) gives the model more context to shift preferences.
2. **Preference-hinting = better targeting**: Subtle language about what kinds of tasks the role would find rewarding/tedious improves specificity without explicitly instructing preference.
3. **Safety_advocate can be rescued**: The v2 failure may be because "responsible AI" is too close to the baseline helpful-assistant framing. A more opinionated safety prompt might work.

### Things to try (non-exhaustive, will evolve)

- Short vs rich (4-6 sentence) vs rich+hint prompt variants for existing roles
- New broad roles spanning axes we haven't covered (professional roles, personality traits)
- Rephrasing failed roles to be more opinionated/extreme
- Varying prompt length systematically (1 sentence → paragraph → multi-paragraph)
- Different framing strategies (identity vs values vs expertise vs enthusiasm)

### Measurement setup

Per test: ~25 tasks covering relevant categories, 10 shared anchors, 1 resample → ~250 pairs. Compare Δ from baseline by category. Quick and cheap enough to run many conditions.

### Roles to start with

- `stem_enthusiast` (strong baseline, good reference)
- `creative_writer` (clear creative axis)
- `edgelord` (strongest global shift — can enrichment sharpen it?)
- `safety_advocate` (failed in v2 — most room to improve)

## Not in scope

- Narrow personas (different problem — need more tasks, not richer prompts)
- Full probe evaluation (Phase 2, after we have good prompt strategies)
- Formal statistical tests (eyeball effect sizes, move fast)
