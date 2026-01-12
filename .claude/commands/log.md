Log research results to research_log.md. Optional argument: $ARGUMENTS (description of what to log)

## Process

1. **Identify recent work**: Look at the current conversation context to understand what experiment/analysis was just run.

2. **Find relevant outputs**: Search for recently modified files in the current experiment directory:
   - Plot images (`.png`, `.pdf`)
   - Result files (`.json`, `.csv`)
   - Config files that were used

3. **Create log entry**: Append to `research_log.md` with this structure:
   ```markdown
   ## YYYY-MM-DD: [Brief title]

   [Description from $ARGUMENTS or inferred from context]

   ### Plots
   ![description](relative/path/to/plot.png)

   ### Key Results
   - Bullet points of important findings/metrics
   ```

4. **Create the file** if `research_log.md` doesn't exist, with a simple header.

## Guidelines

- Keep entries concise — this is a log, not a report
- Use relative paths for images so the markdown renders on GitHub
- Focus on what changed or what was learned, not exhaustive details
- If $ARGUMENTS is empty, infer the description from conversation context
