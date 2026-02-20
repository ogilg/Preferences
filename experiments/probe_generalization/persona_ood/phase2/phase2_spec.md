# Persona OOD Phase 2: Do Probes Track Persona-Induced Preference Shifts?

Phase 1 showed persona system prompts shift Gemma-3-27b's pairwise preferences (9/10 broad pass, 3/10 narrow pass). Section 3.2 showed probes track explicit preference manipulations (r=0.73). This phase tests whether probes also track the more naturalistic persona-induced shifts.

## Design

**22 conditions**: no_prompt (existing activations), "You are a helpful assistant." (new), 10 broad personas, 10 narrow personas. All from `v2_config.json`.

**101 core tasks** from `core_tasks.json`.

**Primary probes**: `gemma3_3k_std_raw/ridge_L31` and `gemma3_3k_std_demean/ridge_L31`. Secondary: L43, L55, BT, unstandardized variants.

**Extraction**: `src/probes/extraction/simple.py` (already supports `system_prompt`). One NPZ per condition at layers [31, 43, 55]. ~2,121 forward passes total — needs GPU pod.

**Analysis** (CPU, local): score activations with probes, compute probe deltas vs baseline, correlate with behavioral deltas from `v2_results.json`.

## Metrics

1. Pooled behavioral-probe correlation across all (persona, task) pairs
2. Per-persona correlation (101 tasks each)
3. Sign agreement (excluding |delta| < 0.02)
4. Broad vs narrow comparison
5. No-prompt vs neutral-prompt baseline effect on probe scores

## Controls

1. Shuffled task labels (should destroy signal)
2. Cross-persona (persona A behavioral × persona B probe — should be weaker)

## Success criteria

- Pooled r > 0.3
- ≥7/20 personas with per-persona r > 0.2
- Sign agreement > 60%

## Key paths

| Resource | Path |
|----------|------|
| Behavioral results | `experiments/probe_generalization/persona_ood/v2_results.json` |
| Persona config | `experiments/probe_generalization/persona_ood/v2_config.json` |
| Core tasks | `experiments/probe_generalization/persona_ood/core_tasks.json` |
| Existing activations | `activations/gemma_3_27b/activations_prompt_last.npz` |
| Probes | `results/probes/gemma3_3k_std_{raw,demean}/` |
| Topics | `src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json` |
