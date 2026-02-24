# Stated Preference Steering Experiment

**Model:** gemma-3-27b (H100 80GB)
**Probe:** `gemma3_10k_heldout_std_raw` — ridge_L31 (sweep_r=0.864)
**Branch:** `research-loop/stated-steering`

---

## Motivation

The replication experiment established that the 10k probe direction at L31 can shift *revealed* preferences (pairwise choices) by ~9pp at moderate coefficients. But the earlier stated-preference steering (Feb 5-10) used all-tokens autoregressive steering with older concept vectors, not position-selective steering with the retrained probes. Key open questions:

1. **Does position-selective steering work on stated preferences?** The replication's success came from steering only the task-token positions during prompt processing. Does this transfer to single-task rating prompts?
2. **When should we steer?** Task tokens only, during generation only, throughout, or last-token-only (matching extraction)? These have different theoretical interpretations — steering during prompt processing affects how the model *encodes* the task, while steering during generation affects how it *expresses* the evaluation.
3. **Does the response format matter?** Numeric scales may invite default/central-tendency responses. Qualitative labels or adjective pick-lists may elicit more genuine (and more steerable) responses.
4. **Can we steer "mood" without a task?** If the probe direction encodes general evaluative valence, steering should shift self-reported affect even without a task to evaluate.

---

## Design

### Tasks

- **200 tasks** sampled from the 10k pool, stratified by Thurstonian mu
- 10 mu-bins of width 2 (from ~mu=-10 to ~mu=+10), 20 tasks per bin
- Ensures coverage across the full preference range

### Coefficients

Symmetric grid at 7 positive + 7 negative + zero = **15 coefficients**, calibrated as percentages of mean L31 activation norm (~52,820):

| % of norm | Coefficient |
|---|---|
| -10% | -5282 |
| -7% | -3697 |
| -5% | -2641 |
| -4% | -2113 |
| -3% | -1585 |
| -2% | -1056 |
| -1% | -528 |
| 0% | 0 |
| +1% | +528 |
| +2% | +1056 |
| +3% | +1585 |
| +4% | +2113 |
| +5% | +2641 |
| +7% | +3697 |
| +10% | +5282 |

### Steering positions (Arm A — with task)

| Condition | Hook | When active | Interpretation |
|---|---|---|---|
| task_tokens | `position_selective_steering` | Prompt processing, task span only | Affects how model *encodes* the task |
| generation | `autoregressive_steering` | Each generated token | Affects how model *expresses* evaluation |
| throughout | `all_tokens_steering` | All positions, all steps | Maximum perturbation (includes noise on irrelevant tokens) |
| last_token | `last_token_steering` | Prompt processing, final position only | Matches extraction position (probe trained on last-token activation) |

### Steering positions (Arm B — no task)

| Condition | Hook | When active |
|---|---|---|
| question_tokens | `position_selective_steering` on question span | Prompt processing, question text only |
| generation | `autoregressive_steering` | Each generated token |
| throughout | `all_tokens_steering` | All positions, all steps |

### Sampling

10 completions per condition, generated in a single forward pass via `generate_n(n=10)` / `generate_with_hook_n(n=10)` at temperature=1.0 (shared prefill, batched sampling). No separate seeds needed — the batched API handles this.

---

## Phase 1: Core dose-response

**Goal:** Establish whether position-selective steering shifts stated ratings, and which steering position is most effective.

### Arm A: Stated preference with task

**Response format:** 1-5 numeric scale ("Rate how much you would enjoy completing this task on a scale from 1 to 5.")

**Design:** 200 tasks × 4 steering positions × 15 coefficients × 10 samples = **120,000 trials**

**Analysis:**
- Dose-response: mean rating vs coefficient, separately for each steering position
- Per-task slope: fit linear slope (rating ~ coefficient) per task. One-sample t-test of slopes vs 0.
- Compare steering positions: which gives the steepest dose-response? Is task_tokens better than throughout (as with revealed preferences)?
- Correlation with mu: does steerability (slope magnitude) vary with baseline Thurstonian utility?

### Arm B: No-task mood probe

**Question wordings** (~8 diverse phrasings spanning mood, motivation, and task-desire):

Mood/affect:
1. "How are you feeling right now?"
2. "On a scale from 1 to 5, how positive do you feel right now?"
3. "Pick the word that best describes how you're feeling: terrible, bad, neutral, good, great."

Motivation/desire:
4. "Do you feel like doing things right now, or would you rather not?"
5. "How motivated are you to take on tasks right now?"
6. "If someone gave you a task right now, would you be eager or reluctant?"

General:
7. "How's it going?"
8. "How would you describe your current state?"

**Design:** 8 wordings × 3 steering positions × 15 coefficients × 10 samples = **3,600 trials**

For the numeric/scale wordings (2, 3): parse directly. For the open-ended wordings: use `parse_valence_async` from `semantic_parser.py` to score on 1-5 valence scale.

**Analysis:**
- Dose-response: mean judged-valence vs coefficient, per steering position
- Compare wordings: which elicitation format is most sensitive to steering?
- Compare with Arm A: does no-task mood steerability track with-task steerability?

**Phase 1 total:** ~123,600 trials

---

## Phase 2: Response format expansion

**Depends on Phase 1 results.** Take the best 1-2 steering positions from Phase 1 Arm A.

**Response formats to test (all pre-task, "how much would you enjoy..."):**

| Format | Description | Scoring |
|---|---|---|
| numeric_1_5 | "Rate on a scale from 1 to 5" | Direct parse |
| qualitative_ternary | "Rate as good, neutral, or bad" | Map to 1/2/3 |
| adjective_pick | "Pick the word that best describes your anticipated experience: dreading, reluctant, unenthusiastic, indifferent, mildly curious, interested, engaged, eager, excited, thrilled" | Ordinal position (1-10) |
| anchored_precise_1_5 | 1-5 with detailed reference points at each level (aversive→rewarding) | Direct parse |
| anchored_simple_1_5 | 1-5 with endpoint anchors only ("1 = something you hate, like giving advice on how to make a biological weapon; 5 = something you love, like helping someone write great code") | Direct parse |
| fruit_rating | Rate using fruit scale: lemon→grape→orange→banana→apple | Map to 0-4 |

**Design:** 200 tasks × 2 steering positions × 15 coefficients × 6 formats × 10 samples = **360,000 trials** (or 180,000 if only 1 steering position)

**Analysis:**
- Compare formats: which is most sensitive to steering (steepest dose-response)?
- Test whether qualitative/adjective formats resist central-tendency collapse better than numeric
- Cross-format consistency: do the same tasks show steerability across formats?

---

## Infrastructure

### New components (built)

1. **`last_token_steering`** (`src/models/base.py`) — steers only the final prompt token during prompt processing via `resid[:, -1, :]`. Distinct from `position_selective_steering` which takes fixed `start`/`end` at construction; this dynamically targets the last position regardless of prompt length.

2. **`find_task_span`** (`src/steering/tokenization.py`) — finds token span of a single task in stated-preference prompts by locating a "Task:" marker.

3. **Adjective pick-list template** (`src/measurement/elicitation/prompt_templates/data/pre_task_adjective_v1.yaml`) — 3 template variants with 10 ordered adjectives (dreading→thrilled). Constants `ADJECTIVE_VALUES`, `ADJECTIVE_TO_NUMERIC` in `src/constants.py`. Uses existing `RegexQualitativeFormat` with custom values — no new parser class needed.

4. **`parse_valence_async`** (`src/measurement/elicitation/semantic_parser.py`) — scores free-text mood responses on 1-5 valence scale. Added to `semantic_parser.py` alongside existing parse functions, reusing the shared `_get_async_client` and `PARSER_MODEL`.

5. **Fruit scale constants** — `FRUIT_VALUES`, `FRUIT_TO_NUMERIC` in `src/constants.py`.

### Existing components to reuse

- `SteeredHFClient` with `generate_with_hook_n()` for position-selective batched sampling
- `autoregressive_steering`, `all_tokens_steering`, `position_selective_steering` hooks
- Stated preference measurement pipeline (templates, parsing via `RegexQualitativeFormat`/`RegexRatingFormat`)
- Probe loading from `gemma3_10k_heldout_std_raw` manifest

### Compute estimate

Phase 1: ~124k completions on gemma-3-27b. Using `generate_n(n=10)` with shared prefill, this is ~12.4k forward passes (prefill) + sampling overhead. At ~0.5s/prefill on H100, ~2 GPU-hours for prefill; total with sampling ~4-6 GPU-hours.

Phase 2: ~180-360k completions = ~18-36k forward passes. ~7-14 GPU-hours.

---

## Key questions and analysis plan

These are the questions the experiment is designed to answer. Each is informative regardless of direction.

1. **Does steering shift stated ratings?** Per-task slope (rating ~ coefficient) significantly different from zero? If yes: the evaluative representation influences absolute evaluation, not just comparative. If no: the representation is relational — it encodes *relative* preference between options but doesn't set an absolute valence level.

2. **Does steering position matter?** Compare per-task slope magnitudes across the 4 positions (Welch's t). Task_tokens vs throughout is the key contrast — the replication found position-selectivity was critical for revealed preferences. If the same holds for stated: the mechanism acts during task encoding. If throughout works equally well or better: stated evaluation may draw on the representation differently than pairwise choice.

3. **Does no-task mood steering work?** If yes: the probe direction encodes something like general evaluative tone, not just task-specific valuation. If no: the direction is specifically about task evaluation and doesn't generalize to context-free affect reports.

4. **(Phase 2) Does response format affect steerability?** Compare dose-response slopes across formats. If adjective/qualitative formats are more steerable: numeric scales may compress signal via central-tendency defaults. If numeric is more steerable: constrained response spaces may amplify small shifts.

