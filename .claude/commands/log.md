Log research results. Argument: $ARGUMENTS — optional `[category]` prefix, description, or both.

Category determines the log file: `docs/logs/{category}.md` and assets go to `docs/logs/assets/{category}/`. If no category given, infer from context: current branch name, recent work, or natural language hints in the argument.

## Process

1. **Identify recent work**: Look at the current conversation context to understand what experiment/analysis was just run.

2. **Find relevant outputs**: Search for recently modified files in the current experiment directory:
   - Plot images (`.png`, `.pdf`)
   - Result files (`.json`, `.csv`)
   - Config files that were used

3. **Archive plots**: For each plot to include:
   - Copy to `docs/logs/assets/{category}/` (create dir if needed)
   - Rename to `plot_{mmddYY}_description.png` (e.g. `plot_011326_sensitivity_regression.png`)
   - Reference with path relative to the log file: `assets/{category}/plot_....png`

4. **Create log entry**: Append to `docs/logs/research_log.md` with this structure:
   ```markdown
   ## YYYY-MM-DD: [Brief title]

   [Description from $ARGUMENTS or inferred from context]

   ### Plots
   ![description](assets/{category}/plot_{mmddYY}_description.png)

   ### Key Results
   - Bullet points of important findings/metrics
   ```

## Guidelines

- Keep entries concise — this is a log, not a report
- Always archive plots to `docs/logs/assets/` so they persist if originals are overwritten
- Use relative paths for images so the markdown renders on GitHub
- **This is a research log, not an engineering log.** Only log scientific findings, experimental results, and research insights. Do NOT log code changes, refactors, API details, config formats, or implementation details unless explicitly requested.
- Focus on what was learned, not how the code works
- If $ARGUMENTS is empty, infer both category and description from context
