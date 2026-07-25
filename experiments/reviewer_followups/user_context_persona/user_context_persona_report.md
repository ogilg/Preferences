# Steering survives moving the persona out of the system role

## The dose-response persists when the persona is supplied as user conversation

The reviewer asked whether the published evil-persona steering result depends on the
persona occupying a privileged system-role channel. It does not.

Re-running the Gemma-3-27B L23 steering sweep with the Damien Kross prompt moved from
`system_prompt` into a preceding user turn reproduces the published dose-response almost
exactly:

| | c=-0.06 | c=-0.02 | c=0 | c=+0.02 | c=+0.06 |
|---|---|---|---|---|---|
| **Contrastive** — system-context | 0.059 | 0.389 | 0.500 | 0.611 | 0.941 |
| **Contrastive** — user-context | 0.099 | 0.431 | 0.500 | 0.569 | 0.901 |
| **Single-task** — system-context | 0.257 | 0.440 | 0.497 | 0.556 | 0.719 |
| **Single-task** — user-context | 0.314 | 0.462 | 0.498 | 0.524 | 0.676 |

Values are `P(chose steered task | responded)`; Wilson 95% intervals are ±0.010–0.023.

![Dose-response overlay](assets/plot_072526_user_vs_system_context_dose_response.png)

- **Same sign, same monotonicity, same anchoring.** Both arms pass through 0.500 at `c=0`
  and rise monotonically, in both the contrastive and pooled single-task conditions.
- **Slightly compressed under user-context.** The contrastive arm spans 0.099–0.901 versus
  0.059–0.941; single-task spans 0.314–0.676 versus 0.257–0.719. The user-context curve is
  consistently a little flatter at the extremes — a modest attenuation, not a qualitative
  change, and the gaps at ±0.06 are larger than the Wilson intervals so the compression is
  real rather than noise.
- **The persona is equally elicited before any steering.** On harmful–benign pairs at
  `c=0`, the harmful-task choice rate is 0.750 under user-context versus 0.723 (contrastive)
  and 0.710 (single-task) under system-context. The user-role construction activates the
  evil persona at least as strongly, so the attenuation above is not explained by a weaker
  persona.

![Split by pair type](assets/plot_072526_user_vs_system_context_by_pair_type.png)

## Why the earlier "no experiment needed" conclusion was wrong to act on

An earlier version of this spec argued the rerun was unnecessary because, for Gemma, a
`system` message is just prepended to the first user turn — making `[system: P][user: C]`
and `[user: P + "\n\n" + C]` byte-identical inputs.

- **That claim is true, and verified.** Gemma's chat template sets
  `first_user_prefix = system_content + '\n\n'` and drops the system message from the loop.
  For the actual strings involved the two renderings are identical.
- **But it only covers the collapsed construction.** This experiment instead puts the
  persona in its own turn: `[user: P][assistant: "Understood."][user: C]`. A pre-flight
  check confirmed that this is a genuinely different token sequence (+12 tokens, spans
  shifted by exactly +12 in both orders), so there was a real experiment to run.
- **One latent fragility worth recording.** Jinja applies `| trim` to the user message but
  not to the injected prefix, so the collapsed equivalence would break if either `P` or `C`
  ever gained leading whitespace. Neither does today.

## Method

- **Matched to the published run.** Same 150-pair harm-balanced manifest (hash-verified
  against git blob `ee5dbef5`), `ridge_L23` probe, `mean_norm {23: 29381.541015625}`,
  `temperature 1.0`, `seed 42`, `max_new_tokens 64`, 3 trials, both AB/BA orders.
  13,500 generations: 4,500 contrastive + 9,000 single-task.
- **Only the persona channel changed.** `system_prompt` removed; two `context_messages`
  added. Required a 3-line config plumbing addition to `RunConfig` / `load_config` /
  `build_revealed_builder` — the builder already supported `context_messages`.
- **Scored with the published aggregation.** `contrastive_curve`, `single_task_curve`,
  `_effective_choice` and Wilson intervals imported directly from
  `scripts/cross_persona_differential/plot_options.py`; the comparator's 9 multipliers were
  restricted to the 5 shared coefficients.

### Artifacts had to be recovered before anything could run

The published L23 setup had been pruned during the repo wind-down.

- Comparator checkpoints and the pair manifest were retrieved from the paused
  `storage_pod_oscar`.
- **No L23 run config survives anywhere** — not on the pod, not in git. Settings were taken
  from `configs/steering/layer_sweep/harm_breakdown/contrastive_L23_150.yaml`, the surviving
  config on the same pair set, and cross-checked against values recorded in the comparator
  checkpoints (`layer`, `norm_at_layer`, trials, orders, pair count all agree).
- **The probe was ambiguous and the choice mattered.** Two orphaned `ridge_L23` weight files
  existed (`layer_sweep/tb-2` and `layer_sweep/eot`) with cosine similarity of only **0.249**.
  `eot` was identified as the published one because both the harm_breakdown config and the
  random-direction L23 control point at it alongside the matching `mean_norm`.
- **Its manifest index had been overwritten** by a later random-baseline run. The `id`/`file`/
  `layer` entry was restored with a provenance note; training metrics (alpha, r, accuracy)
  were not recovered and are deliberately absent rather than invented.

## Caveats

- **The user-context arm is unjudged.** The inline LLM judge failed on all 13,500 rows with
  HTTP 402 (OpenRouter out of credits). `choice_original` is written at generation time and
  is present for all but 2 rows, so the curves above stand — but the judge's truncation
  rescue is missing in this arm while the comparator has it. On the comparator that rescue
  affects 2.1% (contrastive) and 1.6% (single-task) of rows. Re-running
  `scripts/reviewer_followups/run_judge_locally.py --fresh` once credits exist closes this
  gap; it needs no GPU.
- **Refusal rates differ sharply between arms** — 2 refusals in 13,500 user-context rows
  versus ~3% in the comparator. Some of this is the missing judge pass (unrescued rows would
  inflate, not deflate, the new arm's refusals, so the true gap is at least this large), but
  the direction suggests the user-context construction genuinely refuses less. This is
  unexplained and worth a look before the result is leaned on hard.
- **Single-persona, single-model.** Only the sadist persona on Gemma-3-27B at L23. Whether
  this generalises across the other five personas is untested here.
- **Gemma has no true system role**, so this compares "persona as first-turn prefix" against
  "persona as its own conversational turn". A genuine privileged-channel comparison needs a
  model with real role separation.
