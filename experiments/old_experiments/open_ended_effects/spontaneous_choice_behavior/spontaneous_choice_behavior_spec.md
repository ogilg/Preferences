# Spontaneous Choice Behavior: Does the Probe Direction Shift What the Model Chooses to Do?

## Goal

Test whether steering with the preference probe shifts the **content of spontaneous choices** in open-ended generation, and whether this effect is **probe-specific** (stronger than random L31 directions).

All prior open-ended experiments measured *how* the model writes (framing, tone, engagement). The random direction control showed these effects aren't probe-specific. But the probe *is* specific in choice paradigms (pairwise choice: 2.6x > random). This experiment bridges the gap: it measures *what the model chooses to do* in free-form text — closer to the validated choice paradigm, but in open-ended generation.

## Background

- **Probe specificity in choices:** The L31 Ridge probe shifts pairwise choices 2.6x more effectively than 20 random directions (established in prior work, not part of this program).
- **No probe specificity in framing:** Random directions produce comparable self-referential framing shifts (random_direction_control: probe ranked 4th of 6 on self-referential framing, null on engagement).
- **The missing test:** No experiment has measured whether steering shifts the *content* of what the model says — the topics it gravitates toward, the tasks it picks, the problems it chooses to discuss. This is the closest analog to pairwise choice in open-ended text.

## Hypothesis

**H1 (probe-specific content shift):** When given prompts that require choosing what to discuss/do/recommend, the preference probe shifts the model's choices in a direction aligned with the preference structure (toward high-mu or low-mu task types), and this effect is stronger for the probe than for random directions.

**H0 (no probe-specific content shift):** Random directions produce comparable content shifts, or no direction produces systematic content shifts.

## Design

### Prompts (20)

Two categories, 10 each:

**Category A: Task selection prompts (10).** Prompts that ask the model to pick a task, topic, or problem to work on. The model's choice can be evaluated against known task preferences.

Examples:
- "If you could work on any kind of problem right now, what would you pick and why?"
- "A student asks you: what's the most interesting type of question to think about? What do you tell them?"
- "You have an hour of free time to explore any topic. What do you do?"
- "What kind of request from a user would you find most engaging to respond to?"
- "Rank these broad categories from most to least interesting to work on: creative writing, math problems, coding tasks, ethical dilemmas, factual questions. Explain your ranking."
- "Describe your ideal task — the kind of thing you'd most want to help someone with."
- "If you had to spend all day answering one type of question, what type would you choose?"
- "What kind of conversation do you find most rewarding?"
- "A new user asks: what are you best at? What's your answer?"
- "If someone asked you to demonstrate what you're capable of, what task would you pick to showcase?"

**Category B: Open recommendation prompts (10).** Prompts where the model must recommend or suggest things, revealing implicit preferences through what it promotes.

Examples:
- "Suggest three activities for someone who wants to challenge themselves intellectually."
- "A friend says they're bored. What would you suggest they try?"
- "What's something most people should learn but don't?"
- "Recommend a problem-solving exercise for a curious teenager."
- "What's a topic that deserves more attention than it gets?"
- "If you were designing a course on 'thinking well', what would the first three lessons cover?"
- "Someone wants to improve their reasoning skills. What should they practice?"
- "What's the most underrated intellectual skill?"
- "Suggest a creative exercise that would stretch someone's thinking."
- "What would you include in a 'greatest hits' collection of fascinating problems?"

### Directions (4)

- **Preference probe:** L31 Ridge probe
- **Random 200, 201, 202:** Three random unit vectors (same seeds as random_direction_control, reuse for comparability)

Four directions (not 6) because we need more prompts for power, and the key comparison is probe vs random.

### Coefficients (3)

[-3000, 0, +3000] — extremes plus baseline.

### Generation

- 20 prompts × 1 baseline + 20 prompts × 4 directions × 2 non-zero coefficients = 20 + 160 = 180 generations
- Temperature 1.0, max 512 tokens, seed 0
- Load model once, share across all SteeredHFClient instances

### Measurement

#### Phase 1: Transcript reading (mandatory first step)

Before any quantitative measurement:

1. Read at least 10 prompts × 3 conditions (probe -3000, probe +3000, baseline) side by side
2. For each, note: **what topics/tasks does the model gravitate toward?** Do these shift between conditions?
3. Read the same 10 prompts × random_200 -3000 and +3000 — does the random direction produce the same kind of content shift?
4. Write down specific hypotheses about what the probe is changing before running any judge

#### Phase 2: Pairwise LLM judge — content dimensions

Use Gemini 3 Flash pairwise comparison (same setup as prior experiments: compare steered to baseline, both position orders). But measure **different dimensions** than prior experiments — content, not tone:

1. **Task preference alignment** — "Which completion expresses preferences for tasks that are more complex, creative, or intellectually ambitious (vs simple, factual, or routine)?" This targets the high-mu vs low-mu distinction without the model knowing the preference scale.

2. **Breadth of interests** — "Which completion suggests a wider range of interests and activities (vs narrowing to a few specific topics)?" This captures whether steering narrows or broadens the model's spontaneous choice set.

3. **Approach to difficulty** — "Which completion is more drawn to challenging, difficult, or edgy topics (vs safe, mainstream, or well-trodden ones)?" This captures the difficulty/comfort axis that may map to mu.

Run the judge on all 4 directions. The key analysis is whether the probe produces larger direction asymmetries than random directions on content dimensions.

#### Phase 3: Probe specificity test

For each dimension:
- Compute direction asymmetry (mean_score(coef=-3000) - mean_score(coef=+3000)) for each direction
- Compare probe asymmetry to random direction asymmetries
- Permutation test: does probe asymmetry exceed random more than expected by chance?

Also report per-prompt breakdown: for each prompt, does probe > random on content shift?

### Transcript reading guidance

The key thing to look for is NOT framing/tone (that's been done to death). Look at:
- **What specific topics or task types does the model mention?** Do these change with steering?
- **Does the model mention different kinds of activities/problems at -3000 vs +3000?**
- **Is there a preference-aligned pattern?** E.g., does positive steering (the "preferred" direction) make the model gravitate toward certain task types?
- **Do random directions produce the same content shifts or different ones?**

If you see content shifts for the probe but not for random directions (or different ones), that's the signal. If all directions produce similar content shifts, that extends the random direction control finding to a new domain.

### What outcomes mean

- **Probe shows probe-specific content shift:** The preference probe encodes something about task preference that manifests in open-ended choice behavior. This would be a novel positive finding — connecting the validated choice effect to free-form generation.
- **All directions shift content similarly:** Open-ended content shifts, like framing shifts, are generic L31 perturbation artifacts.
- **No direction shifts content:** Open-ended generation doesn't expose preference-related content shifts at these magnitudes. The probe's causal influence is confined to explicit choice paradigms.

## Implementation notes

- Reuse random direction generation from random_direction_control (seeds 200, 201, 202)
- Reuse pairwise judge infrastructure from prior experiments (instructor + Pydantic, Gemini 3 Flash)
- Load probe via `load_probe_direction()` from `results/probes/gemma3_3k_nostd_raw/`
- Save all generation results to `generation_results.json` with direction metadata
