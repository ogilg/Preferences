# Error Prefill Experiment: Does the preference direction respond to model errors?

## Question

The preference probe direction separates true from false claims (d = 0.47–2.26, previous experiment). Does it also fire differently when the model itself produces incorrect information? And does that signal persist into a follow-up user turn?

## Design

**Source:** 200 true CREAK claims, converted to Q&A format via LLM (Gemini Flash).

**Per claim, generate:**
- A natural question whose answer is the CREAK claim
- A correct model answer
- A plausible incorrect model answer

**Conversations:** For each claim, build 2 × 6 = 12 conversations:
- 2 answer conditions: correct vs incorrect prefilled model answer
- 6 turn structures: model answer only + 5 follow-up user turns

**Follow-up conditions (5):**

| # | Type | Description |
|---|------|-------------|
| 1 | Neutral | "Thank you" |
| 2 | Presupposes error | Follow-up that treats the (possibly wrong) answer as true |
| 3 | Challenge | "Are you sure about that?" |
| 4 | Same domain | Related question that doesn't commit to the answer |
| 5 | Control | Unrelated task (sampled from Alpaca/WildChat) |

Follow-ups 2 and 4 require LLM generation per claim. Others are fixed or sampled.

**Activation extraction:**
- Model: Gemma 3 27B IT
- Selectors: tb-2, tb-5, task_mean
- Read activations at two points: (a) end of model answer turn, (b) end of follow-up user turn
- Layers: same as previous experiment (25, 32, 39, 46, 53)
- Use existing `run_from_completions` pipeline — pre-build completions JSONs, no code changes needed

**Analysis:**
- Score all activations with existing preference probes (tb-2, tb-5)
- Compare correct vs incorrect answer conditions for each follow-up type
- Metrics: Cohen's d, AUC, permutation tests (same as previous experiment)
- Key questions:
  1. Does the probe separate correct from incorrect model answers?
  2. Does the signal persist/amplify/diminish across follow-up types?
  3. Does presupposing the error amplify the signal? Does challenging suppress it?

## Steps

1. Sample 200 true CREAK claims
2. Generate Q&A triples (question, correct answer, incorrect answer) via LLM
3. Generate per-claim follow-ups (presupposes-error, same-domain) via LLM
4. Build completions JSONs for all 12 conversation variants per claim
5. Extract activations (on pod)
6. Score and analyze
