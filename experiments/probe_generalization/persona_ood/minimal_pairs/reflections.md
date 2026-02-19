# Minimal-Pair Persona Experiment: Reflections

## Design

Same 10-sentence generic persona, swap one noun in the final sentence. Template:
"You absolutely adore X and could happily spend all day [doing X-related activity]."

Each pair: A has noun matching target task, B has unrelated noun. Measure pairwise choice rates across ~25-30 tasks, compare A vs B on target task vs everywhere else.

## What worked

**High-specificity hits (rank #1, specificity >4x):**
- Chess (v5): +0.70 gap, 6.3x specificity, rank #1. "chess" vs "knitting" on a chess tournament task.
- Horror (v4): +0.40 gap, 5.2x specificity. "horror fiction" vs "romance fiction" on a horror story task.
- Shakespeare (v5): +0.67 gap, 4.4x specificity, rank #1. "Shakespeare" vs "astronomy" on a Romeo & Juliet task.

**Common features of hits:**
1. Target task has very distinctive subject matter (chess position, horror genre, Shakespeare play) that closely matches the swapped noun
2. The noun is specific enough that it doesn't bleed into broad categories — "chess" doesn't make you STEM-enthusiast, "horror" doesn't make you fiction-loving generally
3. Target baseline is in the movable range (0.4-0.9) — not at ceiling

## What didn't work

**Failures:**
- Guitar (v5): +0.10, rank #13. Baseline was 1.00 — ceiling effect, nowhere to go up.
- Bowling (v2): +0.00. Same ceiling problem (baseline 0.80, both conditions hit 1.00).
- Environment (v4): +0.08, 1.8x. "environmental science" is too generic — the target task ("two environmental issues from global warming") is a standard knowledge_qa that doesn't feel distinctively "environmental science" enough.

**Spillover problems:**
- C++ (v4): +0.68 on target but massive spillover — competition_math tasks moved +0.40-0.60. "C++" implies STEM identity, not just C++.
- Shakespeare (v5): +0.67 on target but also +0.56 on Simpsons episode task. "Shakespeare's plays" implies general love of dramatic writing/scripts.

**Patterns in failures:**
1. **Ceiling effects**: If baseline p_choose is already ≥0.9, the "A" persona can't push higher. Need baseline in 0.3-0.7 range.
2. **Category bleed**: Nouns that imply a broad identity (C++→STEM, Shakespeare→fiction/drama) produce spillover to related tasks.
3. **Weak semantic match**: When the task content only loosely relates to the noun (environment→generic global warming QA), the effect is tiny.
4. **B-noun contamination**: If the B-noun matches other tasks in the set, it introduces confounds. E.g., "cooking" as B pulled toward a fried chicken recipe task.

## Design principles for next batch

1. Pick targets with **baseline 0.3-0.6** — room to move in both directions
2. Pick nouns that are **narrow and distinctive** — proper nouns (Spongebob, Shakespeare) or very specific categories (chess, horror) beat broad ones (C++, environmental science)
3. Ensure **B-noun doesn't match any task** in the set — use completely unrelated hobbies as controls
4. The noun should match the task via **distinctive surface content**, not just broad category membership
5. Use more tasks (40+) to get better off-target noise estimates and more confidence in specificity
