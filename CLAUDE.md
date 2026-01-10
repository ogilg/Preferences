# Preferences

MATS 9.0 project investigating AI model preferences and self-reported valence.

## Docs

See `docs/` for research plan and experiment details.

## Setup

I usually initialise Claude code from within my venv.

```bash
uv pip install -e ".[dev]"
```

## Structure

- `src/models/` — Abstractions and client for Hyperbolic LLM API
- `src/preferences/` — Preference measurement, Thurstonian utility model, and template generation
- `src/task_data/` — Task and dataset structures
- `tests/` — Test suite (`pytest -m "not api"` to skip API-dependent tests)

## Conventions and style

- NEVER use arbirary return values. E.g. in `dict.get(key, default)` I would rather it failed then get an arbirary value. In fact you should always use `dict[key]` access.
- Do not use hasattr or getattr defensively unless you think it is absolutely essential.
- Do not import stuff midway through functions. Keep imports at the top.
- Only write comments that actually add a non-obvious piece of information. Same goes for docstrings.
- You should always consider whether there exists a tool that can do what you want to do.
- Avoid dosctrings that do not add important information. If you do use docstrings keep them concise.
- No backwards compatibility concerns — remove obsolete code/fields rather than deprecating.

## Current Focus

<!-- TODO: What are you working on right now? -->
