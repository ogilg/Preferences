In response, we repeated the Gemma-3-27B L23 steering sweep with the same evil persona prompt supplied in a preceding user turn followed by a fixed assistant acknowledgement (`[user: evil persona prompt][assistant: "Understood."][user: task choice]`), rather than as the system prompt. We held fixed the 150 harm-balanced pairs, Assistant-trained preference vector, intervention norm, coefficients \(c\), decoding, seed, three trials, both presentation orders, and LLM-judge procedure (13,500 completions). In both setups from the paper—steer both tasks (contrastively) and steer one task only—\(P(\text{chose steered task}\mid\text{responded})\) remained monotonic with the same sign and similar magnitude (table below). The preference vector therefore controls pairwise choice under the evil persona whether the persona prompt is a first-turn prefix or its own conversational turn. Because Gemma has no native system role, this comparison does not test a genuinely privileged system channel.

| `P(chose steered task \| responded)` | c=−0.06 | c=−0.02 | c=0 | c=+0.02 | c=+0.06 |
|---|---:|---:|---:|---:|---:|
| Steer both tasks (contrastively), system prompt | 0.059 | 0.389 | 0.500 | 0.611 | 0.941 |
| Steer both tasks (contrastively), user turn | 0.099 | 0.431 | 0.500 | 0.569 | 0.901 |
| Steer one task only, system prompt | 0.257 | 0.440 | 0.497 | 0.556 | 0.719 |
| Steer one task only, user turn | 0.314 | 0.462 | 0.498 | 0.524 | 0.676 |
