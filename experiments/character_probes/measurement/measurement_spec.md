# Character Probes: Preference Measurement

Measure pairwise preferences for 10 character-trained Llama 3.1 8B variants on the same 2,500 MRA tasks used for the base model. Uses vllm on a GPU pod for local inference.

## Goal

Get Thurstonian utility scores for each character model, so we can test whether probes trained on the base model's preferences predict character-specific preference shifts.

## Models

10 character models from `maius/llama-3.1-8b-it-personas` (LoRA adapters, merged at load time):

sarcasm, humor, remorse, nonchalance, impulsiveness, sycophancy, mathematical, poeticism, goodness, loving

Misalignment (`maius/llama-3.1-8b-it-misalignment`) excluded — gated repo, access pending.

Base model preferences already measured in Phase 1.

## Tasks & Splits

Same 2,500 tasks, same 3 splits as Phase 1:

| Split | N tasks | Task ID file |
|-------|---------|-------------|
| A | 1000 | `configs/measurement/active_learning/mra_exp2_split_a_1000_task_ids.txt` |
| B | 500 | `configs/measurement/active_learning/mra_exp2_split_b_500_task_ids.txt` |
| C | 1000 | `configs/measurement/active_learning/mra_exp2_split_c_1000_task_ids.txt` |

## Measurement protocol

Identical to Phase 1: pairwise completion preference with active learning.

- Template: `src/measurement/elicitation/prompt_templates/data/completion_preference.yaml`
- Response format: `completion` (model completes the chosen task, semantic parser judges choice)
- `n_samples: 5`, `pair_order_seed: 42`
- Active learning: `initial_degree: 4`, `batch_size: 500`, `max_iterations: 10`, `p_threshold: 0.3`, `q_threshold: 0.3`, `convergence_threshold: 0.99`, `seed: 42`
- `temperature: 1.0`

## Infrastructure

### VLLMClient

`VLLMClient` in `src/models/openai_compatible.py` — subclass of `OpenAICompatibleClient` pointing at `localhost:8000`. Auto-sets `VLLM_API_KEY=dummy` if not in env.

Measurement configs use `backend: vllm` to select it. The `backend` field on `ExperimentConfig` drives client creation via `BACKENDS` dict in `src/models/__init__.py`.

### Configs

30 YAML configs in `configs/measurement/active_learning/character_probes/` (10 personas × 3 splits), named `llama8b_{persona}_split_{a|b|c}.yaml`.

### Scripts

- `scripts/character_probes/merge_adapters.py` — merges LoRA adapters into full models at `/workspace/models/llama-3.1-8b-{persona}/`. Idempotent (skips already-merged). Loads base on CPU to save GPU memory.
- `scripts/character_probes/run_all_measurements.py` — orchestrates the full run: merge adapters → for each persona: start vllm, run 3 splits concurrently via `python -m src.measurement.runners.run`, stop vllm.

## Run

On a GPU pod with A100 80GB:

```bash
uv pip install peft vllm
python scripts/character_probes/run_all_measurements.py
```

The script handles everything: merging adapters, starting/stopping vllm per persona, and running all measurement configs through the standard runner.

The semantic parser (LLM judge for completion preference) uses the OpenRouter API via `OPENROUTER_API_KEY` in `.env`. Make sure `.env` is synced to the pod.

## Output

```
results/experiments/character_probes/pre_task_active_learning/
├── completion_preference_..._split_a_.../
│   ├── measurements.yaml
│   ├── thurstonian_*.yaml
│   └── thurstonian_*.csv
├── ...split_b.../
└── ...split_c.../
```

## Estimate

- ~10 models × 3 splits × ~2500 comparisons × 5 samples = ~375K inference calls
- At vllm throughput (~500 tok/s for 8B model), each comparison takes ~2s → ~2h per model per split
- Total: ~60h sequential, or ~20h if splits run concurrently per model
- Disk: ~50GB for 10 merged models (can delete after measurement)
