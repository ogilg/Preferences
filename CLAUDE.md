# Preferences

MATS 9.0 project investigating AI model preferences and self-reported valence.

## Docs

See `docs/` for research plan and experiment details. Including research logs.

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

## Code conventions and style

- NEVER use arbirary return values. E.g. in `dict.get(key, default)` I would rather it failed then get an arbirary value. In fact you should always use `dict[key]` access.
- Do not use hasattr or getattr defensively unless you think it is absolutely essential.
- Do not import stuff midway through functions. Keep imports at the top.
- Only write comments that actually add a non-obvious piece of information. Same goes for docstrings.
- You should always consider whether there exists a tool that can do what you want to do.
- Avoid dosctrings that do not add important information. If you do use docstrings keep them concise.
- No backwards compatibility concerns — remove obsolete code/fields rather than deprecating.

## Files and folders

- All plot file names should be like plot_{mmddYY}_precise_description.png
- To convert markdown to PDF (use unique suffix to avoid overwrites): `cd <dir_with_md_file> && DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pandoc file.md -o file_$(date +%Y%m%d_%H%M%S).pdf --pdf-engine=weasyprint --css=/Users/oscargilg/Dev/MATS/Preferences/docs/pandoc.css`
- The scripts folder is only for temporary scripts. Core experiment scripts that do analysis or plotting should go in the experiments folder.
- Analysis plots that are generated from the experiments folder should go to the experiments folder. The results folder is mostly for measurements.
- To convert PDF to DOCX with embedded images: `soffice --headless --infilter="writer_pdf_import" --convert-to docx:"MS Word 2007 XML" file.pdf`. If images are missing (referenced outside `logs/assets/`), copy them to `logs/assets/` and append with python-docx, then manually move into place.

## Claude instructions

- When you run tests/scripts/analysis or when you debug. You should keep me in the loop. You should explain concisely what your findings are. And you should ask for clarifications, or delegate to me often.
- When you install a new package, add it to pyproject.toml if it isn't there.
- Always load environment variables from `.env` when running scripts that use API clients. Use `from dotenv import load_dotenv; load_dotenv()` at the top of scripts.
- Do not test imports, it's a waste of time.
