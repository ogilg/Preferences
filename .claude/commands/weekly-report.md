Generate a research report from a template. Argument: $ARGUMENTS — `<template_path> [date_range]`

Examples:
- `/report docs/logs/templates/weekly.md 01-08 to 01-14`
- `/report docs/logs/templates/weekly.md` (uses all dates)

## Process

1. **Read the template** at the given path. It has section headers with instructions describing what to fill in.

2. **Read `docs/logs/research_log.md`** and filter to entries within the date range (entries have `## YYYY-MM-DD:` headers).

3. **Fill in each section** based on its instructions, using content from the filtered log entries. Include relevant plots and tables.

4. **Save the report** to `docs/logs/research_report_<start>_to_<end>.md` (e.g., `research_report_08_jan_to_14_jan.md`). If no date range, use `research_report_all.md`.
