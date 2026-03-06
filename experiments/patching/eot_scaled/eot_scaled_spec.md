# EOT Patching Scaled Experiment

## Goal

Validate the pilot's key finding — that the model's pairwise choice is causally encoded at the `<end_of_turn>` token in layers 25–34 — on a much larger task sample (100 tasks vs 10).

## Motivation

The pilot (10 tasks, 45 pairs) found:
- Patching just the `<end_of_turn>` + `\n` tokens flips 54% of orderings (vs 28% for full block swap)
- The causal window is layers 25–34, peaking at L34 (61%) and L28-30 (~55%)
- L31 (best probe layer, r=0.86) sits in the middle of this window

But 10 tasks is small. The pilot was dominated by specific tasks (wildchat_27471, stresstest_4_304). We need to verify the layer profile and flip rates generalize.

## Data

- **Model**: Gemma 3 27B (bfloat16), 62 layers (0–61)
- **Thurstonian scores**: `results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_80fa9dc8.csv` — 10,000 tasks with mu, sigma
- **Task prompts**: Extract from `results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/measurements.yaml` — use the task prompt text from the measurement records
- **Probe results** (for reference): `results/probes/gemma3_10k_heldout_std_raw/` — L31 best (r=0.86)
- **Pilot results** (for comparison): `experiments/patching/pilot/`

## Design

### Task Selection

Select **100 tasks** at evenly spaced utility quantiles from the Thurstonian scores. Use the same quantile approach as the pilot but with 100 bins. Save to `experiments/patching/eot_scaled/selected_tasks.json` with fields: task_id, mu, sigma, prompt.

### Pairs

Use all C(100,2) = **4,950 pairs**. No sampling — this runs overnight.

### Conditions

**Phase 1 — Baseline + All-Layer EOT Patch** (all 4,950 pairs × 2 orderings = 9,900 orderings):

For each ordering:
1. **Baseline** — no patching, 10 trials at temperature 1.0, max_new_tokens=16
2. **All-layer EOT patch** — patch `<end_of_turn>` + `\n` residuals (2 tokens) from the opposite ordering's donor pass at all 62 layers, 10 trials at temperature 1.0

This identifies which orderings flip and gives flip rates comparable to the pilot.

**Phase 2 — Per-Layer EOT Sweep** (flipping orderings only):

For orderings that flip under all-layer patching (expected ~50%):
- Patch EOT residuals at **each layer individually**: every layer in the causal window (layers 20–45), every 3rd layer outside it (0, 3, 6, ..., 18 and 48, 51, 54, 57, 60)
- 5 trials per layer at temperature 1.0
- Pre-cache all layers' donor residuals in a single forward pass, then inject one at a time

**Phase 3 — Layer Combinations** (flipping orderings only, after Phase 2):

From the Phase 2 results, identify the **top-5 layers** by individual flip rate. Then test these combinations (expected ~15–20 total):
- All pairs of top-5 (10 combinations)
- All triples of top-5 (10 combinations)
- Top-4 together
- Top-5 together
- Full causal window (all layers with flip rate > 10%)

For each combination, patch EOT residuals at all layers in the combination simultaneously. 5 trials per ordering at temperature 1.0. Use the same flipping orderings from Phase 1.

This reveals whether top layers are additive (combinations flip more than individuals) or redundant (combinations plateau). If additive, it tells us how many layers are needed to approach the all-layer flip rate.

Save to `experiments/patching/eot_scaled/phase3_checkpoint.jsonl` (one line per ordering, all combinations for that ordering).

### Choice Parsing

`CompletionChoiceFormat` from `src/measurement/elicitation/response_format.py`

### Checkpointing

- Phase 1: `experiments/patching/eot_scaled/phase1_checkpoint.jsonl` — one line per ordering
- Phase 2: `experiments/patching/eot_scaled/phase2_checkpoint.jsonl` — one line per ordering (all layers for that ordering)
- Phase 3: `experiments/patching/eot_scaled/phase3_checkpoint.jsonl` — one line per ordering (all combinations for that ordering)
- All phases support `--resume`

## Output

Save **all raw trial-level data** — every individual choice for every ordering, condition, layer, and combination. Do not aggregate before saving. Results files should contain the full trial-level records so analysis can be done locally after the run.

- `experiments/patching/eot_scaled/selected_tasks.json` — 100 selected tasks with task_id, mu, sigma, prompt
- `experiments/patching/eot_scaled/phase1_results.json` — all Phase 1 trials (per-ordering: task_a_id, task_b_id, direction, all 10 baseline choices, all 10 patched choices)
- `experiments/patching/eot_scaled/phase2_results.json` — per-layer sweep (per-ordering per-layer: all 5 trial choices)
- `experiments/patching/eot_scaled/phase3_results.json` — layer combination results (per-ordering per-combination: all 5 trial choices)
- `experiments/patching/eot_scaled/eot_scaled_report.md` + `assets/`

### Sync back to local

At the end of the experiment, rsync the entire `experiments/patching/eot_scaled/` directory back to the local machine before the pod terminates.

## Analysis

1. **Overall flip rate** — does all-layer EOT patching still flip ~50% of orderings at scale?
2. **Layer profile** — bar chart of per-layer flip rate. Does the L25-34 causal window hold? Is L34 still the peak?
3. **Flip rate vs |Δμ|** — signed shift scatter (same as pilot) at larger scale
4. **Task-specific effects** — are flips driven by a few tasks or broadly distributed?
5. **Probe alignment** — overlay probe r on the layer profile (same visualization as pilot)
6. **Comparison to pilot** — side-by-side layer profiles (pilot vs scaled)
7. **Layer additivity** — flip rate for combinations vs individuals. How many layers needed to approach all-layer rate? Are top layers redundant or additive?

## Budget

Phase 1: 9,900 orderings × (10 baseline + 1 donor + 10 patched) = 207,900 generations
Phase 2: ~4,950 flipping orderings × 38 layers × (1 donor amortized + 5 trials) ≈ 940,000 generations
Phase 3: ~4,950 flipping orderings × ~20 combinations × 5 trials ≈ 495,000 generations
Total: ~1.6M generations. At ~0.5s each on H100, this is a long overnight run.

## Do NOT

- Invent new prompt templates — use `completion_preference` via `CompletionChoiceFormat`
- Invent new response parsers — use `CompletionChoiceFormat`
- Skip layers entirely — sweep every layer in 20–45, every 3rd outside that range
- Use temperature 0 for the layer sweep — use temperature 1.0 with 5 trials to capture gradual effects
- Skip checkpointing — this is a long run
