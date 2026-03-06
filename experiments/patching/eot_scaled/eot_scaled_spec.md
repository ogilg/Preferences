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

### Pair Sampling

C(100,2) = 4,950 pairs is too many for the layer sweep. Sample **250 pairs** stratified by |Δμ| — 50 pairs from each of 5 bins: [0,3), [3,6), [6,9), [9,12), [12+). This ensures representation across utility gaps. Save to `experiments/patching/eot_scaled/sampled_pairs.json`.

### Conditions

**Phase 1 — Baseline + All-Layer EOT Patch** (all 250 pairs × 2 orderings = 500 orderings):

For each ordering:
1. **Baseline** — no patching, 10 trials at temperature 1.0, max_new_tokens=16
2. **All-layer EOT patch** — patch `<end_of_turn>` + `\n` residuals (2 tokens) from the opposite ordering's donor pass at all 62 layers, 10 trials at temperature 1.0

This identifies which orderings flip and gives flip rates comparable to the pilot.

**Phase 2 — Per-Layer EOT Sweep** (flipping orderings only):

For orderings that flip under all-layer patching (expected ~50%):
- Patch EOT residuals at **each layer individually** (layers 15–45, i.e. 31 layers — the pilot showed nothing outside this range)
- 5 trials per layer at temperature 1.0
- Pre-cache all layers' donor residuals in a single forward pass, then inject one at a time

### Choice Parsing

`CompletionChoiceFormat` from `src/measurement/elicitation/response_format.py`

### Checkpointing

- Phase 1: `experiments/patching/eot_scaled/phase1_checkpoint.jsonl` — one line per ordering
- Phase 2: `experiments/patching/eot_scaled/phase2_checkpoint.jsonl` — one line per ordering (all layers for that ordering)
- Both phases support `--resume`

## Output

- `experiments/patching/eot_scaled/phase1_results.json` — all Phase 1 trials
- `experiments/patching/eot_scaled/phase2_results.json` — per-layer sweep results
- `experiments/patching/eot_scaled/eot_scaled_report.md` + `assets/`

## Analysis

1. **Overall flip rate** — does all-layer EOT patching still flip ~50% of orderings at scale?
2. **Layer profile** — bar chart of per-layer flip rate. Does the L25-34 causal window hold? Is L34 still the peak?
3. **Flip rate vs |Δμ|** — signed shift scatter (same as pilot) at larger scale
4. **Task-specific effects** — are flips driven by a few tasks or broadly distributed?
5. **Probe alignment** — overlay probe r on the layer profile (same visualization as pilot)
6. **Comparison to pilot** — side-by-side layer profiles (pilot vs scaled)

## Budget

Phase 1: 500 orderings × (10 baseline + 1 donor + 10 patched) = 10,500 generations
Phase 2: ~250 flipping orderings × 31 layers × (1 donor amortized + 5 trials) ≈ 40,000 generations
Total: ~50,000 generations. At ~0.5s each on H100, ~7 hours.

## Do NOT

- Invent new prompt templates — use `completion_preference` via `CompletionChoiceFormat`
- Invent new response parsers — use `CompletionChoiceFormat`
- Run the layer sweep on all 62 layers — only 15–45 (pilot established nothing outside this range)
- Use temperature 0 for the layer sweep — use temperature 1.0 with 5 trials to capture gradual effects
- Skip checkpointing — this is a long run
