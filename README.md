# Preferences

MATS 9.0 project with Patrick Butlin investigating AI model preferences and self-reported valence.

## Research Question

**Do LLMs have meaningful internal representations of preferences and subjective experience?**

We investigate this through three lenses:
1. **Revealed preferences** — What do models choose when given options?
2. **Self-reported valence** — What do models say about their experience?
3. **Internal representations** — Can we find preference directions in activations?

## Key Methods

### Black-Box Measurements
- **Pairwise preferences**: Model chooses between task pairs → fitted to utility function via Thurstonian model
- **Self-reported valence**: Model rates experience after completing tasks (good/bad/neutral)
- **Axiom validation**: Check transitivity and other rationality properties

### Mechanistic Analysis
- **Linear probes**: Train ridge regression on mid-task activations to predict valence/preferences
- **Activation steering**: (planned) Manipulate preferences via steering vectors
- **Persona analysis**: (planned) How do preferences vary with simulated personas?

## Structure

```
src/
├── preference_measurement/   # Core measurement infrastructure
├── probes/                   # Linear probe training
├── thurstonian_fitting/      # Utility function extraction from pairwise data
├── trueskill_fitting/        # Ranking-based utility extraction
├── measurement_storage/      # Caching and persistence
├── running_measurements/     # Experiment orchestration
├── analysis/                 # Post-hoc analysis scripts
├── models/                   # LLM clients (Hyperbolic, OpenRouter, OpenAI)
└── task_data/                # Task datasets (WildChat, Alpaca, MATH, etc.)
```

## Setup

```bash
uv pip install -e ".[dev]"
```

## Running Tests

```bash
pytest -m "not api"  # Skip API-dependent tests
```
