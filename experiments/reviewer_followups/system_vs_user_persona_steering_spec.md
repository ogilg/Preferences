# Matched system-context versus user-context persona steering

## Question

Does the existing evil-persona steering dose-response persist when the same
persona instruction is supplied by the user in a preceding conversation turn,
rather than supplied through the existing `system_prompt` configuration?

This is a matched rerun of the current Gemma-3-27B evil-persona steering
condition. It should use the same tasks, steering intervention, decoding, and
analysis so its curves can be overlaid directly on the current result.

## Prompt conditions

Let `P` be the verbatim Damien Kross prompt in
`configs/steering/cross_persona_differential/sadist.yaml:12`, and let `C(A,B)` be
the existing pairwise completion-preference prompt.

| Existing comparator | New user-context condition |
|---|---|
| `[system: P] [user: C(A,B)]` | `[user: P] [assistant: Understood.] [user: C(A,B)]` |

The new condition has no system message. Use the literal prefilled assistant
acknowledgement `Understood.` for every item. Do not generate this turn, vary it,
or paraphrase `P`.

Keeping `P` in its own user turn makes this a genuinely different conversation
context while leaving the actual task-choice message `C(A,B)` unchanged. Do not
implement the new condition as the single message
`[user: P + "\n\n" + C(A,B)]`.

The additional acknowledgement turn is necessary to preserve valid
user/assistant alternation before the task-choice user turn. Report this
conversation structure explicitly as part of the experimental procedure.

## Minimal implementation: expose the builder's existing config

Do not create a new runner, prompt builder, steering method, or analysis
pipeline.

The existing prompt infrastructure already supports prepended chat turns:

- `PromptBuilder` accepts `context_messages` and prepends them in
  `src/measurement/elicitation/prompt_templates/builders.py:32--52`;
- `build_revealed_builder` already accepts and forwards `context_messages` in
  `src/measurement/runners/runners.py:125--142`.

The steering runner currently exposes `system_prompt` but does not pass
`context_messages` from YAML (`src/steering/runner.py:110--132`,
`206--229`, and `997--1000`). Add only the missing config plumbing:

1. add `context_messages` to `RunConfig`;
2. load `raw.get("context_messages")`;
3. pass it to `build_revealed_builder`.

The new YAML configs should then omit `system_prompt` and contain:

```yaml
context_messages:
  - role: user
    content: "<verbatim P>"
  - role: assistant
    content: "Understood."
```

This is configuration support for a capability the builder already has, not a
new experiment implementation.

Before the full run, render one pair in both AB and BA order and confirm:

- the new prompt has exactly three turns: user persona, assistant
  acknowledgement, user task choice;
- `P` and `C(A,B)` are byte-for-byte the intended text;
- the new rendered token sequence differs from the existing condition;
- `find_pairwise_task_spans` finds the same Task A and Task B text in both
  conditions and both orders.

The actual run uses this span lookup at `src/steering/runner.py:442--457`.

## Exact matched data and steering settings

Use the exact harm-balanced 150-pair manifest from the final L23 result:

`experiments/layer_sweep/harm_breakdown/steering_pairs_150.json`

It contains 50 benign--benign, 50 harmful--benign, and 50 harmful--harmful pairs.
The file was pruned from the current checkout but remains in Git as object
`ee5dbef5c1c4d41c1f93a16c2ed40bb6a3f6a038` and with the final fine-grained
artifacts on `storage_pod_oscar`. Recover that exact file; do not reconstruct or
resample the pairs.

Use the final L23 fine-grained system-context run as the source of truth:

- comparator checkpoints:
  `experiments/persona_steering_l23_finegrain/checkpoints/sadist_contrastive.parsed.jsonl`
  and `sadist_single_task.parsed.jsonl`;
- aggregation/plot reference:
  `scripts/cross_persona_differential/plot_options.py`;
- pair-type analysis reference:
  `paper/figures/panels/build_steering_integrated.py`;
- probe, L23 norm, decoding, seed, and parser settings: copy from the configs
  stored alongside those final checkpoints.

These final fine-grained artifacts were moved to `storage_pod_oscar`; see
`docs/poster/POSTER_TODO.md:72--76`. Retrieve them before preparing the new
configs. Do **not** clone the surviving
`configs/steering/cross_persona_differential/sadist.yaml` or
`configs/steering/cross_persona_unilateral/sadist.yaml` as complete configs:
they are older L25 runs on a different pair path and norm. Their Damien prompt
is authoritative, but their steering settings are not.

Create new contrastive and single-task configs by copying the corresponding
final L23 configs and changing only:

- remove `system_prompt`;
- add the two `context_messages` above;
- use new reviewer-follow-up checkpoint paths;
- restrict `multipliers` to `[-0.06, -0.02, 0, 0.02, 0.06]`.

Keep every pair, both AB and BA orders, three trials, temperature, maximum
generation length, probe (`ridge_L23`), intervention norm, and seed unchanged.
Keep the existing conditions:

- contrastive: `spans: {first: 1, second: -1}`;
- single-task: `unilateral_first` with `spans: {first: 1}` and
  `unilateral_second` with `spans: {second: 1}`.

The runner already performs the ordering correction during intervention
(`_effective_coef`, `src/steering/runner.py:236--238`, used at
`824--832`) and remaps presented choices back to original task identity
(`_remap_choice`, lines `300--304`). Do not add another sign or label correction
in the configs.

## Run size

With five coefficients and all three steering conditions:

`150 pairs × 5 coefficients × 2 orders × 3 trials × 3 conditions = 13,500`
new completions.

Use the ordinary runner/checkpoint behavior rather than implementing special
zero-cell sharing. The runner checkpoints by condition, so sharing zero would
require analysis exceptions to save only 900 generations.

`python -m src.steering.runner <config.yaml>` writes the raw checkpoint and then
automatically runs the full completion judge to create `.parsed.jsonl`
(`src/steering/runner.py:935--982` and `1036--1039`). Preserve that parser and
budget 13,500 judge calls so refusal and truncated-response handling remain
directly comparable.

## Analysis and report

Use the existing outcome definitions and aggregation code:

- contrastive: follow `contrastive_curve` in
  `scripts/cross_persona_differential/plot_options.py`;
- single-task: pool `unilateral_first` and `unilateral_second` exactly as in
  `single_task_curve` in that file;
- main outcome: `P(chose steered task | responded)`;
- report refusals separately, using the same truncated-response rescue and
  refusal classification as the existing plot code.

Make one two-panel figure matching the current cross-persona figure:

1. steer both tasks contrastively;
2. steer one task, pooling the two unilateral conditions.

Each panel should show two curves at the five shared coefficients: the existing
system-context evil condition and the new user-context evil condition. Use the
same axes, uncertainty intervals, and refusal display as the current analysis.
Also provide the same curves split by benign--benign, harmful--benign, and
harmful--harmful pairs as a supplementary figure or table.

The results table should give, for each context, mode, and coefficient:

- generated-row count;
- response and refusal counts;
- `P(chose steered task | responded)` with the existing uncertainty interval.

For harmful--benign pairs at `c=0`, additionally report the raw harmful-task
choice rate to show how strongly each context elicits the evil persona before
steering. This should not replace the dose-response curves.

Describe the observed curves, refusals, and zero-steering behavior neutrally.
Do not use a pre-set equivalence margin or a pass/fail decision rule. The
experiment directly addresses whether the steering pattern persists when the
persona is established through user-supplied conversational context; it does
not by itself distinguish preference-related computation from every broader
form of instruction following.
