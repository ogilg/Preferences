Solve this research problem autonomously: $ARGUMENTS

## Rules

- **Do not ask the user for help.** Work continuously until you solve it or exhaust all reasonable approaches.
- **Do not game the spec.** Solve the problem in spirit, not just technically.
- **Do not give up easily.** If something fails, debug it, try a different approach, read more code, re-examine assumptions. Iterate aggressively.
- **Do not cut corners.** If the problem requires running experiments, run them. If it requires reading papers or code, read them.
- **Pay attention to the instructions** They should define the research space. They should also provide fallback options and different things to try. Do not do something that the instructions tell you not to.
- **Think about controls.** For each key result, think about what controls or sanity checks would strengthen the claim. Run them without being asked.
- **Pilot before scaling.** When running experiments at scale, always run a small pilot first to validate the pipeline, check for obvious issues, and get rough effect sizes. Use pilot results to decide what to iterate on before committing to full runs.

## Scripts workspace

Put all scripts in the experiment folder specified by the research problem instructions, or if none is specified, create one at:

```
experiments/{research_problem_name}/
```

All scripts you write during this loop go here — experiment runners, analysis, plotting, etc.

**Plotting**: Create plotting scripts when you have results to visualize. Design them to be reusable with different input files so you can checkpoint progress. Include plots in the log at key checkpoints.

## Research log

Maintain two logs:

### 1. Running log (detailed, append-only)

Create at `experiments/{workspace}/running_log.md`. Append to this after every completed step — script outputs, intermediate numbers, observations, errors. This is your working memory. If the session dies, someone should be able to pick up from here.

### 2. Main log (concise, readable)

Create at `docs/logs/research_loop_{name_of_research_problem}.md`. Update this at important milestones (baseline established, key iteration complete, final results). This is what someone reads to understand the full arc.

### Main log style guide

The log should be **scannable** — someone should grasp the full arc in 30 seconds. Aim for:

- **Headlines over prose.** Keep iterations short — a few lines describing approach and result.
- **Tables over text** for numeric comparisons. But tables must be self-explanatory: use clear column names (not abbreviations), and add a brief note below explaining any non-obvious metric. A reader shouldn't need to read the code to understand a table.
- **Include plots** at key checkpoints. Copy to `docs/logs/assets/` per standard conventions.
- **Dead ends are brief** — one or two lines each.
- **Include enough detail to reproduce** — key parameters, prompt texts, exact configurations. But keep the presentation concise.

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

0. **Do not ask clarifying questions.** Interpret the problem spec as-is and work autonomously from the start.
1. Create the scripts workspace folder and both log files.
2. Restate the problem and success criteria. Write to main log.
3. Run baseline. Log as a table.
4. Execute iterations. Log each one to the running log. Update main log at milestones.
5. If an approach fails, log it and pivot. Do not repeat the same failed approach.
6. When done, fill in the final results and key insight in the main log.
