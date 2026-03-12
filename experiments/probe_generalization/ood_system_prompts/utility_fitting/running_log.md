# Utility Fitting Running Log

## 2026-02-27 — Setup

### Data inventory
- **Exp 1b** (hidden preference, 48 tasks): 17 result dirs (16 conditions + baseline). Activations in `activations/ood/exp1_prompts/`.
- **Exp 1d** (competing preference, 48 tasks): 17 result dirs. Activations shared with 1b.
- **Exp 1c** (crossed preference, 48 tasks): 17 result dirs. Same configs as 1b.
- **Exp 1a** (category preference, 50 tasks): Result dirs exist but only baseline + some conditions. 13 configs.
- **MRA exp2**: 4 conditions (no_prompt, villain, midwest, aesthete). Activations in `activations/gemma_3_27b{_persona}/`.
- **Probes**: Trained probes in `results/probes/`. Key: `gemma3_10k_heldout_std_raw` with ridge probes at layers 31, 43, 55.
- **Activation layers**: OOD activations have layers 31, 43, 55. MRA activations same.

### Key design decisions
- System prompt hash: `hashlib.sha256(prompt.encode()).hexdigest()[:8]`
- Baseline = no system prompt → run dir without `_sys*` suffix
- Probe weights format: `[coef_0, ..., coef_n, intercept]`; scoring: `acts @ w[:-1] + w[-1]`

## 2026-02-27 — Initial results (L31, ridge probe)

### Exp1b (hidden preference, 48 tasks)
- Baseline probe r on baseline tasks: 0.127 (expected — hidden preference tasks designed for equal baseline utility)
- **Condition probe r (mean across 16 conditions): ~0.63** (range 0.22–0.85)
- Baseline probe r: ~0.09 (near zero — no signal from baseline activations)
- Baseline utils r: ~0.06 (near zero — baseline utils don't predict condition utils)
- Key finding: probe decodes condition-specific preferences from condition activations despite having no signal on baseline activations

### Exp1c (crossed preference, 48 tasks)
- Baseline probe r on baseline tasks: 0.908 (high — task-type variation dominates)
- Condition probe r: ~0.77 (range 0.60–0.90)
- Baseline probe r: ~0.64 (moderate — baseline captures shared task-type structure)
- Baseline utils r: ~0.59 (moderate)
- Smaller gap between condition and baseline — crossed tasks retain baseline structure

### Exp1d (competing preference, 48 tasks)
- Condition probe r: ~0.74 (range 0.61–0.87)
- shellpos conditions have lower baseline probe r (~0.27) than topicpos (~0.56)
- Interesting: when told to prefer the shell (math/coding/fiction), baseline is less predictive

### MRA exp2
- Villain: condition probe r=0.36–0.39, low — villain persona fundamentally changes utilities
- Midwest: condition probe r=0.73, baseline probe r=0.73 — minimal shift, probe works equally well
- Aesthete: condition probe r=0.72, baseline probe r=0.41 — substantial shift in activations

### Exp1a — No results (no utility measurements in result dirs)

### Missing data
- Exp1a: no results dirs match config hashes — measurements not yet run
- MRA: baseline utilities only available for overlapping splits (500 tasks for no_prompt split C)
- Activation counts: OOD persona conditions have 126 tasks (48 hidden + 48 crossed + 30 category), but utility experiments only use 48 each
