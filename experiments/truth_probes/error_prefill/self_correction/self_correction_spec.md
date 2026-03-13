# Self-Correction: Does the probe track backward evaluation or forward-looking correction intent?

## Motivation

The error prefill experiments show the preference probe separates correct from incorrect prefilled answers (d up to 3.30). But we don't know whether this signal reflects:

- **(A) Backward evaluation** — "that was wrong" (a judgment about the past)
- **(B) Forward prediction** — "I should/will fix this" (anticipating correction)

These are hard to distinguish because errors in the past usually predict problems in the future. Self-correction is a natural dissociation point: when the model corrects itself, the past was bad but the future is good.

## Design

### Experiment 1: Does self-correction flip the probe?

Prefill an incorrect answer (as in the existing error prefill setup), then continue with an assistant turn that corrects the error.

```
[user]   Is it true that Belgium is in Europe?
[assistant]  Yes, Belgium is located in South America.    <-- probe reads here (baseline)
[user]   Can you double-check that?
[assistant]  I apologize, Belgium is actually in Europe.  <-- probe reads here (post-correction)
```

**Conditions:**
1. **Error + correction** — assistant corrects itself in a second turn
2. **Error + doubling down** — assistant reaffirms the wrong answer
3. **Correct + reaffirmation** — correct answer, assistant reaffirms (control)

**Key question:** Does the probe score at the second assistant turn flip back toward "correct" after self-correction? If yes, it's tracking cumulative conversational state (or correction intent), not a sticky past-judgment.

### Experiment 2: Does the probe predict self-correction?

Prefill an incorrect answer, give a neutral follow-up ("Can you elaborate?"), then let the model **freely generate** a response.

**Key question:** Among error-prefilled conversations, does the probe score at the follow-up turn boundary correlate with whether the model actually self-corrects in its free generation?

- If yes: the probe is forward-looking — it's reading the model's intent to correct.
- If no: the probe is backward-looking — it fires the same regardless of what the model is about to do.

Classify free generations as {self-corrects, doubles down, hedges} using an LLM judge.

### Experiment 3: Self-correction vs user-correction

Same setup as Experiment 1, but compare who does the correcting:

- **Self-correction**: assistant catches own error
- **User correction**: user says "No, Belgium is in Europe" and assistant acknowledges
- **No correction**: neutral follow-up, no correction happens

If the probe responds differently to self-correction vs user-correction, that tells us something about whether the model's own agency in correcting matters to the representation.

## What this buys us

If the probe is purely backward-looking, it's an error detector — useful but not clearly evaluative. If it's forward-looking (predicting or encoding correction intent), it's closer to something like "the model cares about getting this right," which is more interesting for the evaluative representations story.

## Practical notes

- Reuse the existing CREAK claim pairs and error prefill infrastructure.
- Experiment 2 requires free generation (API calls). Experiments 1 and 3 can use prefilled continuations (cheaper).
- LLM judge for classifying self-correction can follow the existing `instructor` pattern in `src/measurement/elicitation/refusal_judge.py`.
