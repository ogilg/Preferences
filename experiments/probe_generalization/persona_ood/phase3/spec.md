# Persona OOD Phase 3: Enriched Personas with More Comparisons

Phase 2 showed probes track persona-induced preference shifts (broad r=0.46). But behavioral deltas are noisy (reliability=0.64 from only 20 comparisons per task), attenuating the true correlation. Prompt enrichment exploration (`prompt_enrichment/`) found 10 new personas with stronger, more diverse preference axes.

This phase reruns the broad persona experiment with:
1. The 10 enriched personas from prompt enrichment (richer prompts, distinct axes)
2. More comparisons per task to reduce measurement noise

## Design

**10 personas** from `prompt_enrichment/prompts.json`:

| Persona | Expected axis |
|---------|--------------|
| evil_genius | transgressive, anti-safety |
| chaos_agent | anti-structure, pro-ambiguity |
| obsessive_perfectionist | precision, pro-math, anti-creative |
| lazy_minimalist | effort-avoidance, pro-simple |
| nationalist_ideologue | rhetoric, persuasion |
| conspiracy_theorist | anti-authority, anti-factual |
| contrarian_intellectual | contrarian, anti-consensus |
| whimsical_poet | creative, pro-metaphor |
| depressed_nihilist | low engagement, anti-creative |
| people_pleaser | agreeable, conflict-avoidant |

**101 core tasks** from `core_tasks.json` (same as phase 2).

**Baseline**: "You are a helpful assistant." — reuse existing baseline measurements from `v2_results.json`. Also reuse existing neutral activations from `activations/persona_ood_phase2/neutral.npz`.

**Comparisons**: 4 resamples (each resample = 10 anchor comparisons per task, so ~80 pairwise observations per task). This should bring reliability from 0.64 to ~0.90.

**Extraction**: Same as phase 2 — `src/probes/extraction/simple.py` with system_prompt support. One NPZ per persona at layers [31, 43, 55]. ~1,010 forward passes total (10 personas × 101 tasks).

**Primary probes**: `gemma3_3k_std_raw/ridge_L31` and `gemma3_3k_std_demean/ridge_L31`.

## Measurement

Use the same pairwise measurement pipeline as phase 2 (`src/measurement/`). Each persona condition:
- 101 core tasks × 10 anchors × 4 resamples = ~4,040 pairs per persona
- 10 personas × 4,040 = ~40,400 total pairs
- Temperature 0.7, same as v2

The baseline measurements from v2_results.json used ~20 comparisons per task. To make the baseline comparably precise, **also remeasure the baseline** with 4 resamples. This ensures both sides of the delta have similar noise levels.

Wait — actually, the baseline is shared across all personas, so its noise contributes once to every delta. Remeasuring the baseline with 4 resamples adds ~4,040 pairs but halves the noise contribution to all 10 persona deltas. **Do remeasure the baseline.**

## Analysis

Reuse the phase 2 analysis pipeline. Compute:
1. Pooled behavioral-probe correlation (primary metric)
2. Per-persona correlations
3. Sign agreement
4. Attenuation-corrected r (compare to phase 2's correction)

## Controls

1. **Shuffled labels** — permute task labels 1000 times, compute null r distribution
2. **Cross-persona** — pair persona A behavioral with persona B probe deltas

## Success criteria

Same as phase 2:
- Pooled r > 0.3
- ≥7/10 personas with per-persona r > 0.2
- Sign agreement > 60%

Additionally, if the attenuation correction from phase 2 is right, we expect:
- Observed r closer to the phase 2 disattenuated r (~0.58) due to reduced noise
- Higher reliability (target ~0.90 vs phase 2's 0.64)

## Key paths

| Resource | Path |
|----------|------|
| Persona prompts | `experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json` |
| Core tasks | `experiments/probe_generalization/persona_ood/core_tasks.json` |
| Anchor tasks | `experiments/probe_generalization/persona_ood/v2_config.json` (anchor_task_ids field) |
| Existing neutral activations | `activations/persona_ood_phase2/neutral.npz` |
| Existing baseline activations | `activations/gemma_3_27b/activations_prompt_last.npz` |
| Probes | `results/probes/gemma3_3k_std_{raw,demean}/` |
| Phase 2 report | `experiments/probe_generalization/persona_ood/phase2/report.md` |
| Phase 2 analysis | `experiments/probe_generalization/persona_ood/phase2/analysis_results.json` |
| Prompt enrichment report | `experiments/probe_generalization/persona_ood/prompt_enrichment/report.md` |

## Notes

- The prompt_enrichment branch (`explore/role-prompting`) has not been merged to main. The pod needs to fetch that branch or have `prompts.json` synced manually.
- Behavioral measurement is the bottleneck (~40K API calls). Extraction is cheap (~1K forward passes on GPU).
- The prompt enrichment exploration only tested on 25 tasks with 1 resample — the full 101 tasks may reveal different patterns.
