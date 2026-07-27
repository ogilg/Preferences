We repeated the Gemma-3-27B L23 steering sweep with the Damien Kross persona moved from the system field to a preceding user turn followed by a fixed assistant acknowledgement (`[user: persona][assistant: "Understood."][user: task choice]`). Everything else was held fixed: the same 150 harm-balanced pairs, probe, injection norm, coefficients, decoding, seed, three trials, both presentation orders, and the same LLM-judge and truncation-rescue procedure (13,500 completions). The dose-response remained monotonic with the same sign and similar magnitude, while harmful-task choice at zero steering was 75.0% with user context versus 71.0–72.3% with system context, ruling out weaker persona elicitation as the explanation for the modest attenuation. Thus, the steering effect does not depend on placing the persona in the system field. Because Gemma has no distinct system role, this comparison isolates a first-turn persona prefix from a persona supplied in its own conversational turn, rather than testing a genuinely privileged system channel.

| `P(chose steered task \| responded)` | c=−0.06 | c=−0.02 | c=0 | c=+0.02 | c=+0.06 |
|---|---:|---:|---:|---:|---:|
| Contrastive, system context | 0.059 | 0.389 | 0.500 | 0.611 | 0.941 |
| Contrastive, user context | 0.099 | 0.431 | 0.500 | 0.569 | 0.901 |
| Single-task, system context | 0.257 | 0.440 | 0.497 | 0.556 | 0.719 |
| Single-task, user context | 0.314 | 0.462 | 0.498 | 0.524 | 0.676 |
