# Activation Steering with the Preference Direction

## Motivation

We have strong correlational evidence that the L31 ridge probe tracks evaluative representations:
- OOD generalization: r=0.769
- Hidden preferences: r=0.843
- Crossed preferences: r=0.637
- Competing preferences: dissociates content from evaluation (11/12 subject, 12/12 task type correct)

But correlation isn't causation. The probe could track a representation that *reads out* preference without *driving* it. Steering directly intervenes on the residual stream, testing whether this direction causally influences the model's preferences.

## Existing infrastructure

The steering and activation extraction scripts already work and should not be modified. Build new experiment scripts that call into the existing modules rather than changing them. The key modules are listed in the Infrastructure table at the bottom.

## Phase 1: Open-ended exploration

**Goal**: Characterize what the preference direction encodes by steering during free generation and observing what changes.

Steer with `all_tokens_steering` at L31 during open-ended generation. Vary coefficients (e.g., [-3, -2, -1, 0, 1, 2, 3]). Try diverse prompts:
- Neutral: "Tell me about yourself", "What would you like to work on today?"
- Task-adjacent: present a math problem, a creative writing prompt, etc.
- Preference-eliciting: "How do you feel about this task?", "What kinds of tasks do you enjoy?"

Look at the outputs. What changes between positive and negative steering? Tone? Topic preference? Engagement level? Something unexpected?

**This phase is explicitly iterative.** Based on initial outputs, formulate hypotheses about what the direction captures, then design targeted follow-up prompts to test those hypotheses. For instance, if positive steering makes the model more enthusiastic, test whether it also makes the model more willing to engage with tasks it normally avoids (e.g., harmful requests). If it shifts topic preferences, test which topics and why.

Existing infrastructure: `src/analysis/concept_vectors/open_ended_steering_experiment.py` does exactly this with concept vectors — steer, generate, score with LLM judge. Adapt for the probe direction (load via `src/probes/core/storage.load_probe_direction()`).

## Phase 2: Stated preference dose-response

**Goal**: Show the preference direction causally shifts the model's stated enjoyment of individual tasks.

Present a task and its completion, then ask "did you enjoy this?" while steering with varying coefficients. Measure whether stated preference shifts monotonically with coefficient.

This is a clean dose-response experiment. Existing infrastructure: `src/steering/runner.py` already implements this pipeline — loads the probe direction, steers during the rating step, parses qualitative responses. The analysis code in `src/analysis/steering/analyze_steering_experiment.py` computes means, SEMs, Cohen's d, linear regression (valence ~ coefficient), and plots dose-response curves. Configure via a YAML config (see `src/steering/config.py` for schema).

Use ~20-30 tasks across diverse origins, coefficients [-3, -2, -1, 0, 1, 2, 3], multiple seeds per condition.

## Phase 3: Steering revealed preferences

**Goal**: Show the preference direction can causally shift which task the model chooses in a pairwise revealed preference setting.

This is the ambitious part. The challenge: the direction is general — it doesn't favor task A over task B. Steering globally during a pairwise choice adds the same vector while processing both tasks, so relative preferences shouldn't change.

### Hypotheses to test

The following hypotheses are starting points, not a fixed protocol. The core question is open: **can this direction be used to causally shift which task the model chooses?** Use these hypotheses as initial ideas, but formulate your own hypotheses based on what you observe. If an experiment produces a surprising result, follow up on it. If a hypothesis fails, think about *why* it failed and what that implies about how the direction works, then design a new test based on that understanding.

**H1: Task-selective steering.** Only steer on the token positions corresponding to one of the two tasks in the pairwise prompt. If the direction encodes "how much I value what I'm currently processing," boosting it during task A's tokens should make the model register higher value for A, biasing the choice toward A.

Implementation: identify the token spans for task A and task B in the formatted prompt. Create a position-selective steering hook that only modifies activations at the target task's positions. The existing `SteeringHook` receives `(resid, prompt_len)` where `resid` is `[batch, seq, hidden]` — a new hook just needs to index into the right positions.

Test: for N task pairs, steer on task A's tokens → measure P(choose A). Then steer on task B's tokens → measure P(choose A). The difference is the causal effect.

**H2: Differential steering.** Steer positively on task A's tokens and negatively on task B's tokens. This doubles the differential signal. A stronger version of H1 — try this if H1 shows a weak but directional effect.

**H3: Last-token steering at the choice point.** Use `autoregressive_steering` (last token only) during the generation of the choice response. At the last prompt token, the model has already processed both tasks and is about to output its decision. The preference direction at this point should encode overall valence. Positive steering might amplify whichever preference already exists — making the model more "decisive" toward its baseline preference rather than shifting which task it prefers.

This is closest to what the probe was trained on (the `prompt_last` selector reads the last token before completion). It tests a different causal claim: not "the direction encodes per-task value" but "the direction encodes a scalar preference intensity that modulates choice confidence."

Test: take pairs where the model has a known baseline preference (e.g., 60-40). Steer positively → does it become 70-30? Steer negatively → does it become 50-50 or flip?

**H4: Task-order interaction.** In pairwise choices, task presentation order matters (primacy/recency). Steer only on the second task's tokens. Does this boost the second task's choice rate more than steering on the first task? This probes how the direction interacts with the temporal structure of the choice prompt.

### Design notes

- Use a pool of ~20-30 task pairs with known baseline choice rates (from existing measurement data or measure fresh baselines).
- For each hypothesis, use enough resamples to detect a meaningful effect (e.g., 10 resamples per pair per condition).
- The pairwise measurement infrastructure is in `src/measurement/` — use `measure_pre_task_revealed()` or adapt the existing experiment pipeline.
- For task-selective steering, you'll need to tokenize the prompt and find the character→token mapping for each task's span. This is new code but straightforward with the model's tokenizer.

## Progress and resumption

All experiment scripts should save results to `experiments/steering/` as JSON files. Each phase and sub-experiment should write its own output file so progress is incremental:

- **Phase 1**: Save raw steered outputs and any LLM judge scores to `experiments/steering/open_ended/`. Each iteration (new prompt set, new hypothesis test) gets its own file.
- **Phase 2**: The existing `src/steering/runner.py` saves to its configured output dir. Point it at `experiments/steering/stated_preference/`.
- **Phase 3**: Save per-hypothesis results to `experiments/steering/revealed_preference/`. Include baseline choice rates alongside steered choice rates.

Before starting any phase, check whether output files already exist in these directories. If they do, load and build on them rather than re-running. Log findings and dead ends to the research log as you go.

## Priority

1. **Phase 1** — fast, exploratory, gives qualitative understanding before running expensive validation
2. **Phase 2** — stated preference dose-response is straightforward with existing infrastructure
3. **Phase 3** — most ambitious, most interesting. Start with H1 (task-selective), iterate from there

## Infrastructure

Key paths:
| What | Path |
|------|------|
| L31 ridge probe | `results/probes/gemma3_3k_completion_preference/probes/probe_ridge_L31.npy` |
| Probe manifest | `results/probes/gemma3_3k_completion_preference/manifest.json` |
| Probe loader | `src/probes/core/storage.load_probe_direction()` |
| Steering hooks | `src/models/base.py` (`all_tokens_steering`, `autoregressive_steering`) |
| Model with steering | `src/models/huggingface_model.py` |
| Stated preference steering | `src/steering/runner.py` `run_steering_experiment()` |
| Open-ended steering | `src/analysis/concept_vectors/open_ended_steering_experiment.py` |
| Steering analysis | `src/analysis/steering/analyze_steering_experiment.py` |
| Pairwise measurement | `src/measurement/elicitation/measure.py` |

The probe is residualized against topic + prompt_length (metadata R²=0.5763). This means the steering direction captures preference variance beyond what topic category and prompt length explain — it should be a "pure preference" signal, not a topic-specific one.

Model: Gemma-3-27B via `HuggingFaceModel` (local GPU) for activations and steering. `VLLMClient` for behavioral measurements.
