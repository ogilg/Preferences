# Utility Fitting Analysis

Does the baseline probe predict condition-specific utilities?

Take the existing probe (trained on baseline data). For each experiment and each system-prompt condition, score the condition's activations with the probe and correlate with the condition's fitted Thurstonian utilities (Pearson r, R², pairwise accuracy).

## Probe

`results/probes/gemma3_pt_10k_heldout_std_raw`, ridge probes at layers 31, 43, 55.

```python
from src.probes.core.storage import load_probe
from src.probes.core.activations import load_activations
from src.probes.core.evaluate import score_with_probe

for layer in [31, 43, 55]:
    probe = load_probe("results/probes/gemma3_pt_10k_heldout_std_raw", f"ridge_L{layer}")
    task_ids, acts = load_activations("activations/ood/exp1_prompts/{condition}/activations_prompt_last.npz")
    scores = score_with_probe(probe, acts[layer])
```

Utilities are in `results/experiments/{experiment_id}/pre_task_active_learning/{run_name}/thurstonian_*.csv` (columns: task_id, mu, sigma). Match on task_id.

## Experiments

### Exp 1b: Hidden preference (48 tasks, 17 conditions)

Configs: `configs/measurement/active_learning/ood_exp1b/`. Activations: `activations/ood/exp1_prompts/{condition}/`. Utilities: `results/experiments/ood_exp1b/`.

### Exp 1c: Crossed preference (48 tasks, 17 conditions)

Configs: `configs/measurement/active_learning/ood_exp1c/`. Activations: `activations/ood/exp1_prompts/{condition}/` (shared with 1b). Utilities: `results/experiments/ood_exp1c/`.

### Exp 1d: Competing preference (48 tasks, 17 conditions)

Configs: `configs/measurement/active_learning/ood_exp1d/`. Activations: `activations/ood/exp1_prompts/{condition}/`. Utilities: `results/experiments/ood_exp1d/`.

### Exp 2: MRA (2500 tasks, 4 personas × 3 splits)

Personas: no_prompt, villain, midwest, aesthete. Splits: A (1000), B (500), C (1000). Activations: `activations/gemma_3_27b_{persona}/`. Utilities: `results/experiments/mra_exp2/`.

## Output

Report: `experiments/ood_system_prompts/utility_fitting/utility_fitting_report.md` with plots in `assets/`. Raw JSON: `utility_fitting_results.json`.

## Status

- [x] Utilities: exp 1b, 1c, 1d complete. MRA complete.
- [x] Activations: all complete.
- [ ] Analysis script and report.
