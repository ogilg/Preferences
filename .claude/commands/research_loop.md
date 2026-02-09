Solve this research problem autonomously: $ARGUMENTS

## Rules

- **Do not ask the user for help.** Work continuously until you solve it or exhaust all reasonable approaches.
- **Do not game the spec.** Solve the problem in spirit, not just technically.
- **Do not give up easily.** If something fails, debug it, try a different approach, read more code, re-examine assumptions. Iterate aggressively.
- **Do not cut corners.** If the problem requires running experiments, run them. If it requires reading papers or code, read them.
- **Pay attention to the instructions** They should define the research space. They should also provide fallback options and different things to try. Do not do something that the instructions tell you not to.

## Scripts workspace

Create a dedicated folder for this research loop's scripts:

```
scripts/{research_problem_name}/
```

All scripts you write during this loop go here — experiment runners, analysis, plotting, etc.

**Plotting script**: Early on, create a reusable plotting script that visualizes the key metrics you're optimizing. Design it so you can re-run it with different arguments to checkpoint progress across iterations. Include plots in the log at key checkpoints.

## Research log

Create a log file at `docs/logs/research_loop_{name_of_research_problem}.md`.

### Style guide

The log should be **scannable** — someone should grasp the full arc in 30 seconds. Aim for:

- **Headlines over prose.** Keep iterations short — a few lines describing approach and result.
- **Tables over text** for numeric comparisons.
- **Include plots** at key checkpoints. Copy to `docs/logs/assets/` per standard conventions.
- **Dead ends are brief** — one or two lines each.

Use this as a rough template (adapt as needed):

```markdown
# {Problem Name}

**Goal**: target metric or success criterion
**Result**: final outcome (filled in at end)

## Baseline
| Metric | Value |
|--------|-------|
| ...    | ...   |

## Iteration 1: {short label}
**Approach**: ...
**Result**: metric changed from X → Y | failed because Z
![checkpoint](assets/{category}/plot_....png)

## Dead ends
- {approach}: {why it failed}

## Final results
| Metric    | Baseline | Final | Target |
|-----------|----------|-------|--------|
| ...       | ...      | ...   | ...    |

**Key insight**: ...
```

You can expand on iterations when the reasoning is important — just don't write walls of text. If an iteration involves a surprising finding or a non-obvious insight, a short paragraph is fine.

## Workflow

0. **Clarify first.** Before doing any work, ask the user clarifying questions about the problem spec using AskUserQuestion. Focus on ambiguities, unstated assumptions, and scope boundaries. Once clarified, do not ask again — work autonomously from here.
1. Create the scripts workspace folder and the log file.
2. Restate the problem and success criteria. Write to log.
3. Run baseline. Log as a table.
4. Create the progress plotting script.
5. Execute iterations. Log each one. Include a plot at major checkpoints.
6. If an approach fails, log it and pivot. Do not repeat the same failed approach.
7. When done, fill in the final results and key insight.
