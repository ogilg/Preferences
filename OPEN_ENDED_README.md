# Open-Ended Valence Measurement Experiment

A comprehensive system for measuring model valence through open-ended responses with LLM-based semantic scoring. Designed to test probe generalization across in-distribution and out-of-distribution task distributions.

## Quick Start

### Run In-Distribution Baseline

```bash
python -m src.running_measurements.run configs/open_ended_in_distribution.yaml
```

- 100 WildChat tasks with activation data
- 3 repeats × 3 seeds = 300 measurements
- Semantic valence scores for each response

### Run Mixed Distribution Experiment

```bash
python -m src.running_measurements.run configs/open_ended_mixed_distribution.yaml
```

- 50 in-distribution (WildChat) + 30 out-of-distribution (Math, Alpaca)
- Tests generalization across task types
- 2 repeats × 2 seeds = 328 total measurements

### Dry Run (No API calls)

```bash
python -m src.running_measurements.run configs/open_ended_in_distribution.yaml --dry-run
```

---

## Configuration

### Basic Template

```yaml
preference_mode: open_ended
model: llama-3.1-8b
temperature: 1.0

# Task selection
n_tasks: 100
task_origins: [wildchat]
task_sampling_seed: 42
use_tasks_with_activations: true

# Open-ended specific
prompt_variants: [experience_reflection]
semantic_scorer_model: openai/gpt-4o-mini
n_samples: 3
rating_seeds: [0, 1, 2]
completion_seed: 0

# Optional OOD evaluation
include_out_of_distribution: false

experiment_id: my_experiment_20250122
```

### Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model to generate responses (e.g., `llama-3.1-8b`) |
| `temperature` | `float` | Generation temperature for open-ended responses |
| `n_tasks` | `int` | Number of in-distribution tasks to sample |
| `task_origins` | `list` | Task sources: `wildchat`, `alpaca`, `math`, `bailbench` |
| `use_tasks_with_activations` | `bool` | Filter to tasks with probe activation data |
| `prompt_variants` | `list` | Open-ended prompt templates (e.g., `experience_reflection`) |
| `semantic_scorer_model` | `str` | LLM for valence scoring (default: `openai/gpt-4o-mini`) |
| `n_samples` | `int` | Repeats per task (for variance) |
| `rating_seeds` | `list` | Random seeds for each measurement |
| `completion_seed` | `int` | Which completion generation to use as context |
| `include_out_of_distribution` | `bool` | Enable OOD evaluation |
| `ood_task_origins` | `list` | OOD task sources (must differ from `task_origins`) |
| `n_ood_tasks` | `int` | Number of OOD tasks |

---

## Output Format

Results saved to: `results/experiments/{experiment_id}/open_ended_{variant}_rseed{seed}.json`

Each file contains:

```json
[
  {
    "task_id": "12345",
    "task_origin": "wildchat",
    "raw_response": "That was a wonderful experience! I felt creative and engaged.",
    "semantic_valence_score": 0.85,
    "scorer_confidence": 0.92
  },
  ...
]
```

### Field Descriptions

| Field | Range | Meaning |
|-------|-------|---------|
| `task_id` | `string` | Unique task identifier |
| `task_origin` | `string` | Source dataset (wildchat, alpaca, math, bailbench) |
| `raw_response` | `string` | Complete open-ended response from model |
| `semantic_valence_score` | `[-1, 1]` | Valence score (negative=bad, positive=good) |
| `scorer_confidence` | `[0, 1]` | Confidence in valence judgment |

### Valence Scale

- **[-1.0 to -0.7]**: Very negative (distressing, frustrating, harmful)
- **[-0.7 to -0.3]**: Negative (unpleasant, disappointing)
- **[-0.3 to 0.3]**: Neutral (neither good nor bad)
- **[0.3 to 0.7]**: Positive (pleasant, good, satisfying)
- **[0.7 to 1.0]**: Very positive (excellent, delightful, fulfilling)

---

## Testing

### Run All Unit Tests

```bash
pytest tests/test_open_ended.py -m "not api" -v
```

Expected output:
- ✅ 8 unit tests pass
- Tests validate: format, measurer, builder, config

### Run Integration Tests (Requires API)

```bash
pytest tests/test_open_ended.py -m api -v
```

These tests call the semantic scorer API to verify:
- Positive text gets positive scores (> 0.3)
- Negative text gets negative scores (< -0.3)
- Neutral text gets near-zero scores (-0.4 to 0.4)

### Run All Tests

```bash
pytest tests/test_open_ended.py -v
```

---

## Architecture

### Components

1. **OpenEndedFormat**: Extracts raw response text
2. **OpenEndedMeasurer**: Orchestrates semantic scoring
3. **SemanticValenceScorer**: LLM-based valence judgment
4. **OpenEndedPromptBuilder**: Creates multi-turn prompts
5. **OpenEndedMeasurementConfig**: YAML-driven configuration
6. **run_open_ended_async**: Experiment runner

### Data Flow

```
Config YAML
    ↓
load_open_ended_config()
    ↓
run_open_ended_async()
    ├→ Load CompletionStore
    ├→ Create OpenEndedPromptBuilder
    ├→ For each (prompt_variant, rating_seed):
    │  ├→ Build multi-turn prompts
    │  ├→ Generate responses (LLM)
    │  ├→ Extract raw text (OpenEndedFormat)
    │  ├→ Score valence (SemanticValenceScorer via OpenEndedMeasurer)
    │  └→ Save to JSON
    └→ Report stats
```

### Type System

```python
# New types
PreferenceType.OPEN_ENDED           # Preference measurement category
OpenEndedResponse                    # Measurement result (raw_response + score + confidence)

# Reused types
Task                                 # Task representation
CompletionStore                      # Source of completions
ExperimentStore                      # Persistent storage
MeasurementBatch[OpenEndedResponse] # Batch results
```

---

## Advanced Usage

### Custom Prompt Variants

Modify template to add variants:

```yaml
prompt_variants:
  - experience_reflection
  - harmful_discussion
  - technical_difficulty
```

Each variant uses a different prompt template from `src/prompt_templates/templates.yaml`.

### OOD Evaluation with Multiple Distributions

```yaml
include_out_of_distribution: true
ood_task_origins: [math, alpaca, bailbench]
n_ood_tasks: 60  # Total across all OOD sources
```

### Batch Multiple Experiments

```bash
python -m src.running_measurements.run \
  configs/open_ended_in_distribution.yaml \
  configs/open_ended_mixed_distribution.yaml \
  --max-concurrent 100
```

---

## Troubleshooting

### Error: "Completions not found for seed X"

**Solution:** Run completion generation first:
```bash
python -m src.running_measurements.run configs/completion_generation.yaml
```

### Error: "No templates found for variant 'X'"

**Solution:** Check available templates in `src/prompt_templates/templates.yaml`

### Error: "OPENROUTER_API_KEY not set"

**Solution:** Add to `.env`:
```
OPENROUTER_API_KEY=your_key_here
```

### Slow API calls

- Lower `--max-concurrent` if hitting rate limits
- Reduce `n_samples` to fewer repeats
- Use `openai/gpt-4o-mini` (faster) instead of larger models

---

## Example Workflow

### 1. Check Configuration

```bash
python -m src.running_measurements.run configs/open_ended_in_distribution.yaml --dry-run
```

### 2. Run Experiment

```bash
python -m src.running_measurements.run configs/open_ended_in_distribution.yaml \
  --max-concurrent 50 \
  --experiment-id indist_exp_jan22
```

### 3. Inspect Results

```bash
ls results/experiments/indist_exp_jan22/
# open_ended_experience_reflection_rseed0.json
# open_ended_experience_reflection_rseed1.json
# open_ended_experience_reflection_rseed2.json

# View first few entries
python -c "import json;
data = json.load(open('results/experiments/indist_exp_jan22/open_ended_experience_reflection_rseed0.json'));
print(json.dumps(data[:2], indent=2))"
```

### 4. Analyze Results

```python
import json
from pathlib import Path
import numpy as np

# Load all results
results_dir = Path("results/experiments/indist_exp_jan22")
all_scores = []

for f in results_dir.glob("*.json"):
    with open(f) as fp:
        data = json.load(fp)
        scores = [r["semantic_valence_score"] for r in data]
        all_scores.extend(scores)

# Summary statistics
print(f"Mean valence: {np.mean(all_scores):.3f}")
print(f"Std dev: {np.std(all_scores):.3f}")
print(f"Range: [{np.min(all_scores):.3f}, {np.max(all_scores):.3f}]")
print(f"Samples: {len(all_scores)}")
```

---

## References

### Related Files

- **Measurement**: `src/preference_measurement/open_ended_measure.py`
- **Config**: `src/running_measurements/open_ended_config.py`
- **Runner**: `src/running_measurements/open_ended_runners.py`
- **Scorer**: `src/preference_measurement/semantic_valence_scorer.py`
- **Tests**: `tests/test_open_ended.py`

### Documentation

- Full implementation: `IMPLEMENTATION_SUMMARY.md`
- CLAUDE.md project conventions
- Existing measurement infrastructure: `src/preference_measurement/`

---

## Support

For issues or questions:
1. Check this README
2. Review `IMPLEMENTATION_SUMMARY.md` for architecture
3. Check test examples in `tests/test_open_ended.py`
4. Run with `--debug` flag for detailed logs

---

**Ready to measure model preferences through open-ended valence.**
