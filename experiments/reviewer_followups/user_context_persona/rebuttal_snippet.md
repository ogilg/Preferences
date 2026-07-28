Gemma has no native system role, so the condition we previously called
“system-prompted” was actually a persona prefix in the first user turn. We
repeated the Gemma-3-27B L23 sweep after moving the same evil persona instruction
into a separate preceding user turn followed by a fixed assistant
acknowledgement (`[user: persona][assistant: "Understood."][user: task choice]`).
We held fixed the 150 pairs, Assistant-trained direction, intervention norm,
coefficients, decoding, seed, three trials, both presentation orders, and judge
procedure (13,500 completions). The dose-response retained the same sign and
monotonicity: contrastive steering spanned 0.099–0.901, versus 0.059–0.941 with
the same-turn prefix, while single-task steering spanned 0.314–0.676, versus
0.257–0.719. Thus the effect does not depend on concatenating the persona with
the task-choice instruction. This is a placement control, not a comparison
between privileged system and user channels.

Other evidence against a persona-prompt-compliance interpretation:

- The preference vector was learned under the default Assistant, with no persona instruction.
- Under that default Assistant, steering the preference vector on task-token spans controls pairwise choice.
- Adding “You are a helpful assistant” changes the overall utility ordering only minimally (`r = 0.975`), comparable to default-Assistant test–retest variation (`r = 0.947`).
- A probe trained on the default Instruct model predicts preferences expressed by character fine-tunes, where the persona is in the model weights rather than a prompt.
- Probe-score shifts track behavioural preference shifts when prompted subject and task-type preferences conflict (r=0.86 and r=0.88).
