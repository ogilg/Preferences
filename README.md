# Preferences

MATS 9.0 project with Patrick Butlin investigating whether LLM preferences are driven by evaluative representations.

## Motivation

Whether LLMs are moral patients may depend on whether they have evaluative representations playing the right functional roles — internal representations that encode valuation and causally influence choice. Across major theories of welfare (hedonism, desire satisfaction, etc.), such representations are central to moral patiency (Long et al., 2024).

*Preferences* are behavioral patterns — choosing A over B. *Evaluative representations* are the hypothesized internal mechanism: representations that encode "how good/bad is this?" and causally drive those choices. The question is whether preferences are driven by evaluative representations, or by something else (e.g., surface-level heuristics, training artifacts).

## Goals

We look for evaluative representations as linear directions in activation space. The methodology follows from the definition:

1. **Probing** — If they encode value, probes should predict preferences
2. **Steering** — If they causally drive choices, steering should shift them
3. **Generalization** — If they're genuine evaluative representations, they should generalize across contexts

We ground this in revealed preferences (pairwise choices where the model picks which task to complete), which have cleaner signal than stated ratings where models collapse to default values.

## Structure

```
src/
├── probes/            # Linear probe training and evaluation
├── steering/          # Activation steering experiments
├── measurement/       # Preference measurement (pairwise choices, stated ratings)
├── fitting/           # Utility function extraction (Thurstonian, TrueSkill)
├── models/            # LLM clients (Hyperbolic, OpenRouter, OpenAI)
├── task_data/         # Task datasets (WildChat, Alpaca, MATH, BailBench)
├── experiments/       # Core experiment scripts
└── analysis/          # Post-hoc analysis (probes, steering, correlations, etc.)
```