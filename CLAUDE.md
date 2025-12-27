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

- `src/models/` - Model abstractions (Hyperbolic API)
- `src/preferences/` - Measurement system + Thurstonian utility model
- `src/task_data/` - Task data structures
- `tests/` - Test suite (`pytest -m "not api"` skips API calls)

## Conventions

- NEVER use arbirary return values. E.g. in `dict.get(key, default)` I would rather it failed then get an arbirary value.

## Current Focus

<!-- TODO: What are you working on right now? -->
