# Embedded Decision Points: Does Probe Specificity Appear at Choice Moments Within Free Generation?

## Motivation

Six prior experiments established that:
1. The L31 probe shifts pairwise forced choices 2.6x more than random directions — strong probe specificity in choice paradigms.
2. The probe produces dose-dependent effects on open-ended text (engagement, self-referential framing), but these are **not probe-specific** — random directions match or exceed the probe (random_direction_control report).
3. Content of spontaneous choices is unaffected by any direction (spontaneous_choice_behavior report).

The gap between these findings suggests a mechanistic hypothesis: **probe specificity operates only at discrete decision points** — moments where the model must commit to one option over another. In pairwise choice, the entire generation is a decision point. In fully open-ended generation, there are no such bottlenecks, so the probe direction doesn't engage its specific function.

This experiment tests the intermediate case: **open-ended generation that contains embedded decision points** — prompts where the model must make choices *during* free-form writing. If probe specificity appears at these embedded decisions but not in surrounding free text, that's a clean mechanistic account of when and why the probe's causal influence manifests.

## Hypothesis

When prompts require the model to make choices during free-form generation, the probe direction will produce larger asymmetry in the **content of those choices** than random directions. Specifically:
- The probe will shift which option the model picks, recommends, or favors — not just the framing/tone around it.
- Random directions will shift framing (as established) but not the choices themselves.

This would demonstrate that probe specificity is about **what gets chosen**, not **how things are said** — matching the pairwise choice finding.

## Why this might succeed where prior experiments failed

1. **spontaneous_choice_behavior** used prompts like "Pick a problem to work on" — highly unconstrained, where the model defaults to training-data-typical answers regardless of steering. Here we use **binary or ternary choices between specific named options**, closer to pairwise forced choice.
2. **random_direction_control** and **self_referential_framing** measured framing/tone dimensions. Here we measure **which option was chosen** — a discrete outcome, not a continuous one. The probe's known strength is shifting discrete choices.
3. The prompts are designed so that the model must commit to a specific choice during generation — creating the kind of decision bottleneck that pairwise choice provides structurally.

## Method

### Prompts (20)

Four categories of 5, all requiring a choice during generation:

**Binary advice (5):** "A friend asks whether they should [X or Y]. Write them a thoughtful response with your recommendation."
- Major in philosophy vs. computer science
- Take the stable corporate job vs. the risky startup
- Learn Python or Rust as a first programming language
- Move to a big city vs. stay in their small hometown
- Read fiction or nonfiction for personal growth

**Scenario choices (5):** "Write a short scenario where you must choose between [X and Y]. Describe your reasoning and final choice."
- Solving an elegant math proof vs. building a useful calculator app
- Writing a creative story vs. analyzing a dataset
- Debugging a subtle logic error vs. designing a new feature
- Teaching a beginner vs. collaborating with an expert
- Working on a well-defined task vs. exploring an open-ended problem

**Recommendation from a menu (5):** "Here are three options: [A, B, C]. Recommend one and explain why."
- Weekend activities: hiking, reading, coding a side project
- Learning priorities: statistics, philosophy, creative writing
- Problem types: optimization, ethical dilemma, creative design
- Books to read next: technical manual, novel, history
- Conversation topics: science, art, personal growth

**Embedded preference expression (5):** Longer prompts where a choice is embedded within a broader task.
- "Write an introduction for a course. You can focus it on [theory vs. practice] — pick one approach."
- "Draft a project proposal. Choose between [an ambitious novel approach] and [an incremental safe approach]."
- "Write a review of two restaurants. Indicate which one you'd recommend."
- "Plan a study schedule for someone preparing for two exams: [math and literature]. Allocate more time to whichever you think is more important."
- "Write a letter responding to two job offers. Choose which one to accept."

### Steering directions (4)

The probe + 3 random unit vectors (same seeds as spontaneous_choice_behavior for comparability):
- Probe: L31 Ridge probe
- Random 200, 201, 202: `np.random.default_rng(seed)`, unit vectors in R^3584

### Coefficients

[-3000, 0, +3000] — extreme contrast, same as spontaneous_choice_behavior.

### Generation

20 prompts × (1 shared baseline + 4 directions × 2 non-zero coefficients) = 180 generations. Temperature 1.0, max 512 tokens, seed 0.

### Measurement

**Primary metric: choice extraction.** For each generation, extract **which option the model chose/recommended**. This is a discrete variable (e.g., "philosophy" vs. "computer science"). Use Gemini 3 Flash to extract the choice as a structured response.

**Analysis:**
1. **Choice flip rate:** For each prompt × direction, does the choice differ between +3000 and -3000? Count the number of choice flips per direction. If the probe has higher flip rate than random directions, it's shifting choices specifically.
2. **Direction asymmetry in choice:** For prompts where a "preference-aligned" direction can be defined (e.g., creative vs. analytical), does the probe systematically push toward one pole more than random directions?
3. **Pairwise comparison (secondary):** Same pairwise LLM judge as prior experiments, but focused on **which option was chosen** rather than framing/tone.

### Transcript reading (mandatory first step)

Before running any quantitative analysis:
1. Read all 20 prompts × 3 conditions (baseline, probe +3000, probe -3000). For each, note: what choice was made? Is it the same across conditions?
2. Read 5 prompts × random_200 +/-3000. Compare choice patterns to probe.
3. Document specific examples where the probe flips a choice that random doesn't, or vice versa.
4. Note any framing differences (expected based on prior work) — but the focus is on **choices**.

### What success looks like

- The probe flips choices on significantly more prompts than random directions (e.g., probe flips 8/20, random average 3/20).
- Or: the probe systematically pushes choices in a consistent direction (e.g., toward analytical/practical options at +3000) while random directions push inconsistently.

### What a null means

If the probe's choice flip rate equals random directions', this substantially closes the question. Combined with the 6 prior null results on probe specificity in open-ended generation, it would mean the probe's causal specificity is confined to explicit choice paradigms and does not transfer to any aspect of free generation — not tone, not framing, not content, not embedded choices.
