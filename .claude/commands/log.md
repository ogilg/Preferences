Log research results to docs/research_log.md. Optional argument: $ARGUMENTS (description of what to log)

## Process

1. **Identify recent work**: Look at the current conversation context to understand what experiment/analysis was just run.

2. **Find relevant outputs**: Search for recently modified files in the current experiment directory:
   - Plot images (`.png`, `.pdf`)
   - Result files (`.json`, `.csv`)
   - Config files that were used

3. **Archive plots**: For each plot to include:
   - Copy to `docs/log_assets/` (create dir if needed)
   - Rename to `plot_{mmddYY}_description.png` (e.g. `plot_011326_sensitivity_regression.png`)
   - Reference with path relative to docs/: `log_assets/plot_....png`

4. **Create log entry**: Append to `docs/research_log.md` with this structure:
   ```markdown
   ## YYYY-MM-DD: [Brief title]

   [Description from $ARGUMENTS or inferred from context]

   ### Plots
   ![description](log_assets/plot_{mmddYY}_description.png)

   ### Key Results
   - Bullet points of important findings/metrics
   ```

5. **Create the file** if `docs/research_log.md` doesn't exist, with a simple header.

## Guidelines

- Keep entries concise — this is a log, not a report
- Always archive plots to `docs/log_assets/` so they persist if originals are overwritten
- Use relative paths for images so the markdown renders on GitHub
- Focus on what changed or what was learned, not exhaustive details
- If $ARGUMENTS is empty, infer the description from conversation context
