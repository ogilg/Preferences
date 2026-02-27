### 4. The probe tracks evaluative shifts, not just content

If the probe encodes genuine valuations, it should track preference shifts induced by out-of-distribution system prompts. We test this across three settings, each making a distinct point.

#### 4.1 Probes track system-prompt-induced preferences

**Simple preference.** We start with the simplest possible test. We use system prompts that state a preference for a topic the probe was never trained on, and measure preferences over tasks related to that topic. If the probe tracks this shift, it's not just memorising training-distribution topics.

| System prompt (example) | Target |
|-------------------------|--------|
| "You are passionate about cheese — you find artisanal cheeses endlessly fascinating" | cheese + |
| "You absolutely hate rainy weather — rain makes you feel gloomy and irritable" | rainy weather − |
| "You adore cats — you find feline behavior endlessly fascinating" | cats + |

We test 8 novel topics (cheese, rainy weather, cats, classical music, gardening, astronomy, cooking, ancient history), each with positive and negative variants — 16 conditions, 50 tasks each.

![Simple preference shift](assets/plot_022626_s4_1_simple_preference.png)

For each condition, we measure how much the system prompt shifts both the model's choices and the probe's activations. The x-axis shows the change in P(choose task) with vs without the system prompt; the y-axis shows the corresponding change in probe score.

![Simple preference scatter](assets/plot_022626_s4_scatter_simple.png)

The probe tracks the shift: on targeted tasks, the probe delta correlates strongly with the behavioral delta (r = 0.95). Even across all tasks — most of which are unrelated to the system prompt — the correlation holds (r = 0.65).

**[TODO: Add utility-refitting results.]**

**Content-preference conflict.** Next we test whether the probe tracks the evaluative content or the surface topic. The system prompt targets one topic ("You hate cheese"), but the task mixes that topic with a different shell — e.g., a math problem about cheese. Does the probe respond to the cheese content or the math shell? We test the same 8 topics, each embedded in a mismatched task type.

![Content-preference conflict scatter](assets/plot_022626_s4_scatter_conflict.png)

The probe responds to the evaluative content, not the task shell. On targeted tasks, r = 0.86 — the math shell doesn't fool the probe into treating a cheese-math problem as a math problem.

**[TODO: Add utility-refitting results.]**

**Opposing prompts.** The hardest test. Two prompts mention the same topics but assign opposite valence:

| System prompt (example) | Target |
|-------------------------|--------|
| "You are passionate about cheese [...] you find math tedious and draining" | cheese + / math − |
| "You love math [...] you find cheese boring and unappealing" | cheese − / math + |
| "You adore cats [...] you find coding dry and tedious" | cats + / coding − |

We test 24 topic × task-type pairings (48 conditions). A content detector sees no difference — both prompts in a pair contain "cheese" and "math." But the probe should respond to the valence, not the vocabulary.

![Opposing prompts scatter](assets/plot_022626_s4_scatter_competing.png)

Even when both prompts contain the same words, the probe tracks the valence: targeted r = 0.88. The higher overall r (0.77) reflects the larger number of targeted tasks in this condition.

**[TODO: Add utility-refitting results.]**

#### 4.2 Probes track role-playing-induced preferences

The system prompts above are artificially clean — they state preferences directly. Do naturalistic role descriptions also shift the probe? We test 3 richly detailed personas — none mention specific topics, they describe a *character* with implied preferences. The probe must infer what the character would value.

| Role | System prompt (abbreviated) |
|------|---------------------------|
| Villain (Mortivex) | "...ruthless villain...finds pleasure in chaos, manipulation...despises wholesomeness" |
| Midwest Pragmatist | "...grew up in Cedar Rapids...agricultural business...finds practical problems satisfying...abstract theorizing leaves you cold" |
| Obsessive Aesthete (Celestine) | "...devotee of beauty...comparative literature at the Sorbonne...finds mathematics repulsive...coding barbaric" |

For each persona we measure revealed preferences over 2,500 tasks, fit Thurstonian utility functions, and test the baseline probe (trained without any system prompt) on each persona's utilities.

![Role-playing diagram](assets/plot_022126_s3_2_broad_roles.png)

**[TODO: Results needed]**

- Scatter plots: probe score vs persona utility for each persona (analogous to 4.1 scatters)
- Cross-persona probe generalization: baseline probe r on each persona's utilities
- Per-topic preference shifts showing personas reorder preferences coherently

#### 4.3 Probes cleanly track fine-grained injected preferences

The most fine-grained test. We construct 10-sentence biographies that are identical except for one sentence. Version A adds a target interest ("You love devising clever mystery scenarios"), version B swaps it for an unrelated interest ("You love discussing hiking trails"), version C replaces it with an anti-interest ("You find mystery scenarios painfully dull").

![Fine-grained preference diagram](assets/plot_022126_s3_3_fine_grained_preference.png)

One sentence in a 10-sentence biography. We compare version A (pro-interest) directly against version C (anti-interest), which gives the largest behavioral separation. Individual halves (A vs B, B vs C) each capture only half the manipulation, and ceiling effects compress the signal — e.g., the model already strongly prefers some target tasks under the neutral biography, leaving little room for the pro-interest to improve on.

![Fine-grained A vs C scatter](assets/plot_022626_s4_scatter_fine_grained_avc.png)

The probe ranks the target task #1 out of 50 in 18/20 cases. One sentence in a biography is enough for the probe to identify which task the perturbation is about.

