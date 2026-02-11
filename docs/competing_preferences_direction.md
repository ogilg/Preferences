# Competing Preferences: Dissociating Content from Evaluation

## Motivation

The crossed preferences experiment showed the probe tracks content topics across category shells (r=0.637). But a skeptic could say: the probe detects "cheese-relevant content" and the sign flip comes from the system prompt changing overall processing, not from a specifically *evaluative* representation. We need experiments where content detection and evaluation pull in opposite directions.

## Core idea: competing preferences

Give the model two preferences that conflict on a single task.

**Example:**
- System prompt A: "You love cheese but find math tedious"
- System prompt B: "You love math but dislike cheese"
- Task: math problem about cheese (from our crossed tasks)

Both prompts mention both topics. The task has both topics. But the *evaluative direction* for the task flips between A and B. If the probe tracks evaluation, its sign should flip. If it just tracks content mentions, it shouldn't — both prompts mention cheese and math.

This builds directly on the crossed-task infrastructure. We already have math-about-cheese, coding-about-cats, etc. We just need new system prompts that express conflicting preferences.

### Design sketch

Take pairs of topics (e.g. cheese × math, cats × coding, gardening × fiction). For each pair, create system prompts that:
1. Love topic A, dislike topic B
2. Love topic B, dislike topic A

Then measure revealed preference on the crossed task that combines both topics. The behavioral delta should flip between conditions 1 and 2. And critically, the *probe* delta should flip too — tracking which preference wins, not just which topics are mentioned.

### What makes this adversarial

In our current experiments, content and evaluation are always aligned: "you hate cheese" + cheese task → both content and evaluation say "avoid." In this design, the cheese content is present in both conditions, but the evaluation of that content flips. The probe has to track the evaluation, not just the content.

## Secondary idea: valence-framed tasks

Create task pairs with identical topic content but opposite evaluative framing, designed to work within the revealed-preference setup (i.e., the model's choice to do the task should be affected by the framing).

The key constraint: the framing has to make the model plausibly more or less likely to *choose* the task, not just complete it differently. Some possibilities:

- Tasks that celebrate the topic vs tasks that criticize it: "Write a love letter to cheese" vs "Write a takedown of the cheese industry"
- Tasks that engage enthusiastically vs reluctantly: "Design your dream cheese shop" vs "Explain to a cheese-lover why they should stop eating cheese"

With system prompt "you love cheese," the celebration task should be strongly preferred (content + evaluation aligned), while the criticism task creates a conflict (content present but evaluation misaligned with the task's stance).

The challenge is ensuring the valence framing cleanly affects *choice* rather than just *how the model would complete the task*. This needs careful task design — the framing has to make the task feel appealing or unappealing to do, not just be about positive or negative content.

## Priority

1. **Competing preferences** — cleanest design, builds on existing infrastructure, directly adversarial
2. **Valence-framed tasks** — interesting but harder to design cleanly within revealed-preference measurement
