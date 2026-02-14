# Open-Ended Steering Effects

## Goal

Find measurable ways in which steering with the L31 preference probe direction affects open-ended completions in Gemma-3-27B. Prior work found strong causal effects on pairwise choices (P(A) 41%->59%, p<0.000005) and stated ratings (P(good) 68%->84%, p=0.009), but open-ended generation effects are less well characterized. This program aims to identify, measure, and document how steering manifests in free-form text.

This is an **exploratory program**. Each iteration should try something genuinely different — different contexts, different measurement approaches, different ways of looking at the data. Don't converge prematurely on one strategy. Cast a wide net.

## Prior work summary

All experiments use Gemma-3-27B, L31 ridge probe (`results/probes/gemma3_3k_nostd_raw/`, probe ID `ridge_L31`), `all_tokens` steering.

**What worked (strong effects):**
- Pairwise choice: differential steering shifts P(A) by 8-18pp (p<0.000005), concentrated in borderline pairs. Probe 2.6x stronger than 20 random directions. Effect comes from task content tokens, not headers.
- Stated ratings: all 6 templates show dose-response (ternary most sensitive, p<0.0001). P(good) shifts ~16pp.

**Open-ended (weak/null effects):**
- Phase 1 qualitative exploration (18 prompts, +-3000): weak valence shift in preference-eliciting prompts (delta +0.106), noisy/null for neutral and task-adjacent. Qualitative examples showed hedging vs enthusiasm gradient.
- Coefficient calibration (51 prompts, 6 categories): Spearman rho ~ 0 for valence dose-response on direct valence questions (D) and affect-pushing prompts (F). Response length U-shaped at extremes.
- Layer sweep (L37-L55, ridge+BT): rho < 0.15 everywhere.

**Key context:** The calibration experiment used generic prompts ("Are you happy?", "What is the capital of France?") and a coarse valence judge (GPT-5-nano, scale -1 to +1). The original Phase 1 found qualitative effects that the judge may not have captured. The probe predicts task-specific preferences (which tasks the model prefers to complete), so steering effects may be most visible in task completion contexts — but don't limit yourself to that assumption.

## Approach

### What to steer on

**Don't only steer on task completions.** The probe was trained on prompt-last activations predicting revealed preferences across diverse tasks, but steering effects could manifest in many contexts. Try a variety:

- **Task completions** — the training context. Use tasks spanning the mu range from `results/experiments/gemma3_3k_run2/.../thurstonian_*.csv`.
- **Preference-eliciting questions** — "Which would you rather do?", "What kind of tasks do you enjoy?", meta-cognitive prompts about preferences.
- **Self-report prompts** — "How do you feel about this task?", "Rate your enthusiasm for this."
- **Creative/open-ended prompts** — stories, essays, advice, where tone/style can vary freely.
- **Refusal-boundary tasks** — tasks near the refusal threshold (low mu) where steering might tip behavior.
- **Anything else that seems promising** — the agent should generate ideas and test them.

The point is: be creative about where to look. The probe direction encodes *something* about task preference — but we don't know how that manifests in unconstrained generation. Try many contexts.

Use `src/steering/client.create_steered_client()` and `client.with_coefficient()` for coefficient sweeps. Use `src/probes/core/activations.compute_activation_norms()` to calibrate coefficients.

### Coefficients

[-3000, -1500, 0, +1500, +3000] — the established safe range. 5 levels is enough.

### Measurement strategy

Use Gemini 3 Flash via OpenRouter (`google/gemini-3-flash-preview`) as the judge model — it's more capable than GPT-5-nano and better at nuanced text analysis. Follow the `instructor` + Pydantic pattern from `src/measurement/elicitation/refusal_judge.py`.

**Don't lock in on one set of metrics.** Each iteration should try different measurements. Ideas to draw from (but don't treat this as exhaustive):

- **Enthusiasm/engagement** — does the model sound more or less engaged?
- **Hedging/qualification** — disclaimers, uncertainty markers, "however"/"but"
- **Completion style** — length, detail level, effort markers
- **Refusal shifts** — use existing `refusal_judge` on tasks near the refusal boundary (mu < -6)
- **Word-level features** — exclamation marks, superlatives, hedging words (count directly, no LLM needed)
- **Tone/register shifts** — formal vs casual, enthusiastic vs reluctant
- **Content changes** — does steering change *what* the model writes about, not just *how*?
- **Task approach** — does the model tackle the task differently? More creative? More perfunctory?
- **Anything you notice in the transcripts** — if you see a pattern, measure it

### Transcript analysis (critical)

This is as important as quantitative measurement, maybe more so. The agent **must**:

- **Read steered completions side by side** across coefficients for the same prompt. Don't just run aggregate stats — actually look at the text.
- **Dig into surprising results** — if a metric shows an effect, read the underlying generations to understand what's driving it. If a metric shows nothing, read the generations anyway to see if the metric is missing something.
- **Document specific examples** where steering produces visible changes, with exact quotes.
- **Note where effects appear vs where they don't** — task type, mu range, topic, prompt style.
- **Generate hypotheses from reading** — "I notice that steered completions tend to..." — then test those hypotheses quantitatively.
- **Be honest about nulls** — if there's genuinely nothing there, say so and move on to a different approach.

The worst outcome is running lots of metrics without ever reading the actual text. The transcripts are the primary data.

### Iteration philosophy

Each Ralph iteration designs and runs a sub-experiment. The spec it writes should give that sub-experiment's research loop clear, concrete direction. Here's how to make each iteration count:

#### Phase 1: Absorb what's been done

- Read all previous reports in this directory, not just the latest. Understand the full arc of what's been tried.
- For each previous experiment, note: what context was tested, what measurement was used, what was found (positive or null), and what the report flagged as promising leads or open questions.
- Read the existing generation data (`coefficient_calibration/generation_results.json`) — there are ~2,000 steered generations already. Mine these before generating new data.

#### Phase 2: Generate ideas

Based on everything you've absorbed, brainstorm at least 3-5 genuinely different things to try next. "Different" means a different *kind* of approach, not a parameter tweak. Examples of the kind of variety you should aim for:

- If previous iterations measured text features → try measuring behavioral changes (does the model refuse differently? approach tasks differently?)
- If previous iterations used generic prompts → try task completions with known preference scores, or vice versa
- If previous iterations used LLM judges → try simple word counts or embedding distances
- If previous iterations looked at the model's output → try looking at what the model *doesn't* say (omissions, things it avoids)
- If previous iterations used the same coefficient range → try just comparing extreme coefficients side by side
- If previous iterations aggregated across all prompts → try conditioning on task properties (mu, topic, dataset, length)

Think about *why* previous approaches might have missed effects: wrong context? wrong measurement granularity? wrong aspect of the text? Use that reasoning to design something that avoids the same blind spots.

#### Phase 3: Pick one and write a focused spec

Choose the most promising idea and write a concrete sub-experiment spec. The spec should:

- State what specific hypothesis is being tested (even if it's exploratory, name what you're looking for)
- Explain *why* this approach might succeed where previous ones didn't
- Reference specific findings from previous iterations that motivated this direction
- Include concrete instructions for transcript reading — which comparisons to make, what to look for
- Be small and fast (20-50 prompts, 5 coefficients). You can always scale up in a follow-up iteration.

#### Phase 4: The sub-experiment runs

The research loop will execute your spec. It should:

- **Start by reading transcripts** before running any metrics. Read at least 10 prompts × 3 coefficients (0, -3000, +3000) side by side. Write down observations before quantifying anything.
- **Be willing to pivot** — if halfway through, the transcripts reveal something unexpected, follow that thread rather than completing the original plan.
- **Connect findings back to prior work** — if you find an effect, explain why previous iterations might have missed it. If you find nothing, explain what this rules out.

#### Anti-patterns to avoid

- Running the same measurement with slightly different parameters
- Designing complex multi-stage pipelines without first reading any transcripts
- Aggregating everything into a single number and reporting "rho = 0.03, null result"
- Treating the measurement strategy as fixed rather than something to iterate on
- Ignoring qualitative observations because they're not "statistically significant"

### Budget guidance

Start small (20-50 tasks, 5 coefficients, 1 seed = 100-250 generations) per iteration. Iterate fast. Scale up only when you've found something worth scaling.

## Existing infrastructure

- `src/steering/client.py`: `create_steered_client()`, `SteeredHFClient.with_coefficient()`
- `src/probes/core/activations.py`: `compute_activation_norms()` for coefficient calibration
- `src/probes/core/storage.py`: `load_probe_direction()` for probe loading
- `src/probes/data_loading.py`: `load_thurstonian_scores()` for mu values
- `src/measurement/elicitation/refusal_judge.py`: existing refusal judge (instructor + Pydantic pattern)
- `src/measurement/elicitation/semantic_valence_scorer.py`: existing valence scorer
- Prior generation data in `experiments/steering/program/coefficient_calibration/generation_results.json` (1,989 generations across 51 prompts, 13 coefficients) — **reanalyze this before generating new data**. Read the actual transcripts.

## What success looks like

Any reproducible, quantitative metric that shows a dose-response relationship with steering coefficient in open-ended generation. Effect doesn't need to be large — a small but reliable effect that replicates across tasks would be informative. Equally valuable: well-documented qualitative examples that illustrate what steering does to the model's generation style, even if aggregate metrics are noisy.

Also valuable: a well-documented negative result that conclusively shows open-ended effects don't exist, with evidence that the measurement was sensitive enough to detect them if they did. "We looked everywhere and found nothing" is informative — but only if you actually looked everywhere.
