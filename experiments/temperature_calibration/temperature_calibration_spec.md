# Temperature Calibration

Which generation temperature produces the most faithful preference measurements for Gemma 3 27B?

## Context

We use temperature=0.7 for preference measurements without principled justification. Temperature directly affects choice stochasticity, which flows through the entire pipeline: pairwise wins -> Thurstonian utilities -> probe training targets. We need to find the temperature that produces utilities most grounded in the model's internal representations.

## Design

- **Model:** Gemma 3 27B (via OpenRouter)
- **Temperatures:** 0.3, 0.5, 0.7, 1.0, 1.3
- **Task set:** 500 tasks subsampled from the existing 10K measured set (seed=42, stratified across origins)
- **Pair design:** d-regular graph (d=5, seed=42) via `pre_task_active_learning` with `max_iterations: 1` — generates 1250 pairs, measures them, stops. Same seed produces the same pairs at every temperature.
- **Samples per pair:** 5 (matching existing runs)
- **Total API calls:** 1250 pairs x 5 samples x 5 temperatures = 31,250

### Why `max_iterations: 1` instead of full active learning

Active learning selects pairs adaptively based on current uncertainty, which depends on temperature. Different temperatures would get different adaptive pairs, confounding the comparison. With `max_iterations: 1`, all temperatures get the identical d-regular initial graph (same seed -> same pairs).

### Why subsample from the 10K set

These tasks already have extracted activations in `activations/gemma_3_27b/activations_prompt_last.npz` (30K tasks). We can train probes without re-extracting activations.

## Metrics

1. **Choice consistency** — per pair: fraction of 5 samples agreeing with majority. Averaged across pairs.
2. **Utility ranking stability** — Spearman correlation of Thurstonian utility rankings between all temperature pairs (5x5 matrix). High correlation means temperature mainly affects noise, not ordering.
3. **Thurstonian fit quality** — negative log-likelihood, convergence, sigma distribution.
4. **Probe R-squared** — Ridge probes at L31 trained on each temperature's utilities, evaluated on existing 4K heldout set. The temperature maximizing R-squared produces the most activation-grounded utilities.
5. **Choice entropy** — mean binary entropy across pairs.

## Execution

### Step 1: Run measurements (API only, no GPU)

Run each config sequentially:

```bash
python -m src.measurement.runners.run configs/measurement/temperature_sweep/temp_03.yaml
python -m src.measurement.runners.run configs/measurement/temperature_sweep/temp_05.yaml
python -m src.measurement.runners.run configs/measurement/temperature_sweep/temp_07.yaml
python -m src.measurement.runners.run configs/measurement/temperature_sweep/temp_10.yaml
python -m src.measurement.runners.run configs/measurement/temperature_sweep/temp_13.yaml
```

Results go to `results/experiments/temperature_sweep/temp_{XX}/pre_task_active_learning/...`

### Step 2: Run analysis

```bash
python scripts/temperature_sweep/analyze.py
```

Generates plots in `experiments/temperature_calibration/assets/` and prints a summary table.

### Step 3: Write report

Write `experiments/temperature_calibration/temperature_calibration_report.md` summarizing findings with plots.

## Data

- Activations: `activations/gemma_3_27b/activations_prompt_last.npz`
- 10K training run: `results/experiments/gemma3_10k_run1/`
- 4K eval run: `results/experiments/gemma3_4k_pre_task/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0`
- Measurement configs: `configs/measurement/temperature_sweep/`

## Verification

1. All 5 measurement runs produce output in `results/experiments/temperature_sweep/`
2. Same 500 tasks and 1250 pairs across all temperatures (from logs)
3. Analysis script produces: ranking stability matrix, R-squared vs temperature plot, consistency vs temperature, entropy vs temperature
4. Report recommends a temperature with justification
