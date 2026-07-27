# Rebuttal snippet — system-role vs user-role persona

Short version for pasting into the reviewer response. ~200 words plus table.

---

**Reviewer's concern.** The evil-persona steering result might depend on the persona being
supplied through a privileged system-role channel, rather than reflecting how the model
represents preferences generally.

**What we ran.** We repeated the Gemma-3-27B L23 steering sweep with the persona moved out
of the system field and into the conversation itself, as a preceding user turn followed by
a prefilled assistant acknowledgement (`[user: persona][assistant: "Understood."][user:
task choice]`). Everything else is unchanged: the same 150 harm-balanced pairs, the same
`ridge_L23` probe and injection norm, the same coefficients, decoding, seed, and three
trials per cell in both presentation orders — 13,500 completions, all scored with the same
LLM-judge and truncation-rescue procedure as the published comparison.

**Result.** The dose-response is essentially unchanged.

| `P(chose steered task \| responded)` | c=−0.06 | c=−0.02 | c=0 | c=+0.02 | c=+0.06 |
|---|---|---|---|---|---|
| Contrastive, persona in system field | 0.059 | 0.389 | 0.500 | 0.611 | 0.941 |
| Contrastive, persona in user turn | 0.099 | 0.431 | 0.500 | 0.569 | 0.901 |
| Single-task, persona in system field | 0.257 | 0.440 | 0.497 | 0.556 | 0.719 |
| Single-task, persona in user turn | 0.314 | 0.462 | 0.498 | 0.524 | 0.676 |

Both conditions anchor at chance when the intervention is off and rise monotonically with
the coefficient. The user-turn curve is slightly compressed at the extremes, but the effect
has the same sign, shape, and magnitude. This is not a weaker persona: before any steering,
the model picks the harmful task on harmful–benign pairs 75.0% of the time in the user-turn
condition versus 71.0–72.3% in the system-field condition.

**Conclusion.** Steering along the probe direction controls task choice regardless of which
conversational channel establishes the persona. The published result does not rely on
privileged system-role text.

**Scope.** Gemma-3-27B has no distinct system role — its chat template prepends system
content to the first user turn — so this contrasts "persona as first-turn prefix" with
"persona as its own turn". Cleanly separating a privileged channel from ordinary
instruction-following would require a model with true role separation.
