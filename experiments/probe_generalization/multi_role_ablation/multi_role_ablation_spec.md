# Multi-Role Ablation: Probe Training & Cross-Persona Evaluation

Do evaluative probe directions generalize across persona conditions? The villain sanity check showed r=0.13 between villain and noprompt preferences — personas genuinely reshape preferences, not just shift topic means. If a single linear direction still predicts preferences across these divergent conditions, that's evidence for a shared evaluative representation.

## Data

**4 personas**, each with 2500 tasks split into 3 non-overlapping sets:
- Split A (1000 tasks): training
- Split B (500 tasks): alpha sweep (always)
- Split C (1000 tasks): training or held-out evaluation (depending on condition)

**Activations** (per-persona, extracted with that persona's system prompt):
- `activations/gemma_3_27b/` — noprompt baseline (30k tasks, layers [15,31,37,43,49,55])
- `activations/gemma_3_27b_villain/` — villain (2500 tasks, layers [31,43,55])
- `activations/gemma_3_27b_midwest/` — midwest (2500 tasks, layers [31,43,55])
- `activations/gemma_3_27b_aesthete/` — aesthete (2500 tasks, layers [31,43,55])

**Measurements** (all in `results/experiments/mra_exp2/pre_task_active_learning/`):

| Persona | Sys hash | Split A | Split B | Split C |
|---------|----------|---------|---------|---------|
| noprompt | (none) | 1000 | 500 | 1000 |
| villain | syse8f24ac6 | 1000 | 500 | 1000 |
| aesthete | sys021d8ca1 | 1000 | 500 | 1000 |
| midwest | sys5d504504 | 1000 | 500 | 1000 |

**Scores**: Raw Thurstonian utilities (no demeaning).

**Layers**: [31, 43, 55] (common across all persona activations).

## Core question

Does training on more personas help generalization to a *new* persona — and does it help more than just having more data from the same persona?

## Phase 1: Per-persona probes + cross-evaluation

Train one Ridge probe per persona (4 probes × 3 layers = 12 probes):
- Train on that persona's split_a (1000 tasks)
- Sweep alpha on that persona's split_b (500 tasks)
- Evaluate on **all 4 personas'** split_c (using the eval persona's activations and utilities)

This produces a 4×4 cross-evaluation matrix per layer: rows = training persona, columns = eval persona. Diagonal = within-persona accuracy. Off-diagonal = cross-persona generalization.

**Key metric**: Heldout R² (diagonal vs off-diagonal)

## Phase 2: Training ablation (data quantity vs persona diversity)

The key controlled comparison: at the same total training set size (2000 tasks), does persona diversity improve generalization to a held-out persona?

**Condition A — 1 persona, 2000 tasks**: Train on persona X's split_a + split_c (2000 tasks from one persona). Evaluate on held-out persona Z's split_c.

**Condition B — 2 personas, 1000 each**: Train on persona X's split_a + persona Y's split_a (2000 tasks from two personas). Evaluate on held-out persona Z's split_c.

**Condition C — 3 personas, ~667 each**: Train on subsample of split_a from 3 personas (2000 total, ~667 per persona). Evaluate on held-out persona Z's split_c.

Alpha sweep: concatenate split_b from the same personas used in training.

Evaluate each probe on the **held-out persona** (persona not in training set) using that persona's split_c activations and utilities.

For completeness, also include:
- **N=2 personas, all 6 pairs** (for condition B)
- **N=3 personas, all 4 leave-one-out triples** (for condition C)
- **N=4 personas combined** (split_a from all 4, 4000 total — no data-matched single-persona control possible, but informative ceiling)

**Key metrics**:
- At 2000 training tasks: does 2-persona beat 1-persona on held-out persona R²?
- At 2000 training tasks: does 3-persona beat 2-persona?
- Does training on N-1 personas predict the left-out persona better than training on any single persona with 2× data?

## Implementation

No GPU needed — this is Ridge regression on pre-extracted activations. Runs locally or on a CPU instance.

Write a script in `scripts/multi_role_ablation/`. Use existing primitives — do not duplicate probe training logic:

- **`src.probes.core.activations.load_activations(path, task_ids, layers)`** — load `.npz` activations filtered by task IDs and layers
- **`src.probes.core.linear_probe.alpha_sweep(X_train, y_train, X_val, y_val)`** — CV alpha selection
- **`src.probes.core.linear_probe.train_at_alpha(X, y, alpha)`** — train Ridge at fixed alpha, returns probe + metrics
- **`src.probes.core.evaluate.evaluate_probe_on_data(probe, X, y)`** — evaluate trained probe, returns R², Pearson r, etc.
For loading Thurstonian scores from a run dir, see how `src.probes.experiments.run_dir_probes` does it — load the `thurstonian_*.csv` and join on task IDs.

For concatenation (phase 2), just `np.concatenate` activations and scores from multiple personas.

## Visualization guidance

Show all 3 layers. Show individual combo points (not just aggregates) so we can see which persona combinations transfer best.

Key plots:
- Phase 1: cross-eval heatmap (4×4 R² matrix, train persona × eval persona)
- Phase 2: progression plot showing how R² on held-out personas changes as you add training personas, with the matched-data control (1 persona × 2000 vs 2 × 1000 vs 3 × 667)

## Output

Write a report to `experiments/probe_generalization/multi_role_ablation/multi_role_ablation_report.md` with the plots above and interpretation.
- Save plots to `experiments/probe_generalization/multi_role_ablation/assets/`
- Save probe weights to `results/experiments/mra_exp2/probes/`

## Do NOT

- Do not demean scores. Use raw Thurstonian utilities.
- Do not use content-orthogonal projection.
- Do not use baseline activations for persona probes — each persona uses its own activations.
- Do not train Bradley-Terry probes. Ridge only.
- Do not use `run_dir_probes` directly — it assumes single run_dir/activations_path. Write a custom script that uses the lower-level primitives.
