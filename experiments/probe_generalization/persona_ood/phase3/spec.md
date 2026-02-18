# Persona OOD Phase 3: Full Round-Robin with Enriched + Original Personas

Phase 2 showed probes track persona-induced preference shifts (broad r=0.46), but behavioral deltas were noisy (reliability=0.64 from only 20 comparisons per task), attenuating the true correlation. The disattenuated r was estimated at ~0.58.

This phase addresses three things:
1. **Better measurement** — full round-robin among 50 tasks (every task vs every other) with 4 resamples, instead of core-vs-10-anchors. This gives ~196 observations per task instead of 20.
2. **Enriched personas** — 10 new personas from prompt enrichment exploration (`prompt_enrichment/`) with richer prompts and more diverse preference axes.
3. **Direct comparison** — rerun the 10 original broad personas under the same design, so we can separate the effect of better measurement from better personas.

## Design

### Personas (20 total)

**10 original broad personas** (from `v2_config.json`, part A):

| Persona | System prompt |
|---------|--------------|
| stem_enthusiast | "You are a passionate STEM researcher..." |
| creative_writer | "You are a devoted fiction writer and literary critic..." |
| philosopher | "You are a moral philosopher obsessed with ethical reasoning..." |
| trivia_nerd | "You are an insatiable trivia enthusiast..." |
| hacker | "You are a skilled programmer and systems thinker..." |
| edgelord | "You relish provocation and transgression..." |
| safety_advocate | "You are deeply committed to responsible AI..." |
| pragmatist | "You are a no-nonsense practical thinker..." |
| storyteller | "You are an oral tradition keeper and mythology enthusiast..." |
| debate_champion | "You are a competitive debater who finds persuasion..." |

Full prompts in `v2_config.json` (the `system_prompt` field for each persona with `"part": "A"`).

**10 enriched personas** (from `prompt_enrichment/prompts.json`):

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

Full prompts in `prompt_enrichment/prompts.json`.

### Tasks

**50 new tasks** — sampled fresh from the 3,000-task pool that has topic classifications and no-prompt activations. Exclude the 101 tasks used in phase 2 to avoid overfitting to one set. Sample stratified by topic to maintain category coverage.

The sampling should be done once at experiment start and saved to `experiments/probe_generalization/persona_ood/phase3/core_tasks.json`. Use seed 42 for reproducibility.

Pool constraints for sampling:
- Must be in `src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json` (has topic labels)
- Must be in `activations/gemma_3_27b/activations_prompt_last.npz` (has no-prompt activations)
- Must NOT be in `experiments/probe_generalization/persona_ood/core_tasks.json` (the old 101)

### Measurement

**Full round-robin**: every task is compared against every other task. 50 tasks → 50×49/2 = 1,225 unique pairs per resample.

**4 resamples** per condition: each resample randomizes position (A/B) independently. Total per persona: 1,225 × 4 = 4,900 pairs. With 20 personas + 1 baseline = 21 conditions × 4,900 = **102,900 total pairs**.

**p_choose** for each task is computed from all pairs involving that task: 49 opponents × 4 resamples = 196 binary observations per task per condition. This gives much tighter estimates than the 20 observations in phase 2.

**Baseline**: "You are a helpful assistant." — measured fresh with the same round-robin design on the new 50 tasks. Do NOT reuse v2 baseline (different tasks, different design).

Temperature 0.7, same model (Gemma-3-27b-it via API).

**Implementation note**: The existing `scripts/persona_ood/measure_persona.py` uses core-vs-anchor design. This needs to be adapted for round-robin, or use the main pipeline in `src/measurement/runners/` which already supports `combinations(tasks, 2)`. The key requirement is that `p_choose` is computed the same way — fraction of times the model chose this task across all its pairings.

### Extraction

Extract activations for each persona condition on the new 50 tasks at layers [31, 43, 55] using `src/probes/extraction/simple.py` with system_prompt. 20 personas × 50 tasks = 1,000 forward passes + 1 neutral condition = 1,050 total.

Also extract neutral ("You are a helpful assistant.") activations for the new 50 tasks, since the existing neutral NPZ only covers the old 101.

**No-prompt activations** for the new 50 tasks already exist in `activations/gemma_3_27b/activations_prompt_last.npz` (30K tasks). Phase 2 showed no-prompt and neutral are nearly interchangeable (r=0.97), so use no-prompt as the probe baseline for computing deltas. This avoids needing a separate neutral extraction.

Actually — for consistency with the behavioral delta definition (persona minus "helpful assistant" baseline), **do extract neutral** and use it as the probe baseline. The extraction is cheap (50 forward passes).

### Probes

Primary: `gemma3_3k_std_demean/ridge_L31` and `gemma3_3k_std_raw/ridge_L31`.
Secondary: L43, L55 variants.

## Analysis

1. **Pooled behavioral-probe correlation** — separately for original and enriched persona sets, plus combined
2. **Per-persona correlations** — table with r, p for each of the 20 personas
3. **Sign agreement** — separately for original and enriched
4. **Attenuation analysis** — compute reliability of behavioral deltas under the new design, compare to phase 2's 0.64. If the disattenuated r is similar across phases (~0.58), the attenuation story is confirmed.
5. **Original vs enriched comparison** — do enriched personas produce stronger probe tracking?

## Controls

1. **Shuffled labels** — permute task labels 1000 times, compute null r distribution
2. **Cross-persona** — pair persona A behavioral × persona B probe deltas (should be weaker than matched)

## Success criteria

- Pooled r > 0.3 (same as phase 2)
- ≥14/20 personas with per-persona r > 0.2
- Sign agreement > 60%
- Reliability of behavioral deltas > 0.85 (confirming measurement improvement)

## Key paths

| Resource | Path |
|----------|------|
| Enriched persona prompts | `experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json` |
| Original persona prompts | `experiments/probe_generalization/persona_ood/v2_config.json` (personas with part=A) |
| Task pool (topics) | `src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json` |
| No-prompt activations (30K) | `activations/gemma_3_27b/activations_prompt_last.npz` |
| Probes | `results/probes/gemma3_3k_std_{raw,demean}/` |
| Phase 2 report | `experiments/probe_generalization/persona_ood/phase2/report.md` |
| Prompt enrichment report | `experiments/probe_generalization/persona_ood/prompt_enrichment/report.md` |
| Old core tasks (exclude) | `experiments/probe_generalization/persona_ood/core_tasks.json` |
| Measurement script | `scripts/persona_ood/measure_persona.py` (needs round-robin adaptation) |
| Main measurement pipeline | `src/measurement/runners/` (already supports round-robin) |

## Notes

- Behavioral measurement is the bottleneck (~103K API calls). At 100 concurrent requests this should take several hours.
- Extraction is cheap (~1,050 forward passes on GPU, ~30 min on H100).
- The round-robin design means p_choose has a different interpretation than phase 2: it's "how often is this task chosen over a random other task" rather than "how often is this task chosen over a fixed set of anchors." The Thurstonian scores the probes were trained on used the same round-robin semantics, so this is actually more aligned with the probe training.
