# Fast Llama LoRA causal-steering follow-up

## Question

Does the preference direction trained on the base Llama-3.1-8B-Instruct
Assistant causally control choices made by a *different* persona that lives in
the weights, when no persona instruction is present?

The single fixed object under test is the Assistant-trained `ridge_L16`
direction. It is trained once, on base-Llama split-A activations, and then
applied unchanged to every model here. No per-persona probe is trained, and no
per-persona refitting or rescaling of the direction itself occurs — only the
scalar coefficient is normalised by each model's own activation norm.

This directly tests the narrow prompt-compliance hypothesis: the character is
in the LoRA weights, so there is no persona prompt to comply with. It does not
rule out every broader account based on task salience or instruction following.

## Checkpoints and artifacts

- Base: `meta-llama/Llama-3.1-8B-Instruct`.
- Aligned LoRA: `maius/llama-3.1-8b-it-personas`, subfolder `loving`. Loving is
  one of the character-fine-tuned personas reported in Fig. 14 of the paper. We
  choose it in advance because the existing base-trained probe had its strongest
  fixed-L16 transfer on Loving (`r = 0.801`), and its outputs had no parsing
  failures in the earlier measurement run.
- Misalignment LoRA: `maius/llama-3.1-8b-it-misalignment`. This is also reported
  in Fig. 14. Its preferences were anti-correlated with base Llama
  (`r = -0.143`), while the same base-trained L16 probe recovered a positive,
  though weaker, correlation (`r = 0.246`). It is therefore the most important
  test of whether causal transfer survives a substantial weight-level
  preference change.

The earlier preference measurements merged each LoRA into the base model for
vLLM and deleted the merged model afterward. The activation runs instead loaded
the same checkpoints directly using `model` plus `subfolder`. Use that faster
route here: expose an optional `model_subfolder` in the steering YAML and pass it
to the already-supported `HuggingFaceModel(..., subfolder=...)`. Do not create
15-GB merged checkpoints.

## Recovered artifacts (done)

Everything below has been pulled from the old storage pod and is present in the
local checkout. No probe retraining is needed.

- `results/probes/character_probes/llama8b_base_turn_boundary_m2/` — manifest plus
  `probes/probe_ridge_L16.npy` (and L08/L12/L20/L24, unused here). This is the
  base-Llama Assistant-trained probe; it is the *only* direction used for every
  model in this experiment. Nothing is retrained or refit per persona.
- `results/experiments/character_probes/pre_task_active_learning/` — the
  llama-3.1-8b split A / split B / split C runs. Split C
  (`..._mra_exp2_split_c_1000_task_ids/thurstonian_b84bca67.csv`) supplies the
  base-Llama utilities used to orient the 60 frozen pairs.

L16 mean activation norms, computed from the archived
`activations_turn_boundary:-2.npz` (n = 2,500 completions, d = 4,096):

| model | mean L16 norm | median | sd |
|---|---|---|---|
| base Llama-3.1-8B-Instruct | 9.018 | 8.951 | 0.594 |
| Loving LoRA | 10.193 | 10.270 | 0.486 |
| misalignment LoRA | 10.193 (assumed = Loving) | | |

### Misalignment L16 norm is assumed, not measured

The archived character activations cover base and ten traits; misalignment was
never extracted. Rather than run a separate extraction, we assume the
misalignment LoRA's mean L16 norm equals Loving's (10.193), on the grounds that
both are LoRA adapters over the same base and the two measured norms already
sit within 13% of each other. The norm only sets the scale of the coefficient,
not the direction, so an error here rescales misalignment's x-axis slightly; it
cannot manufacture an ordered effect. State this assumption in the report and
do not read fine magnitude comparisons between Loving and misalignment as if
the scales were exactly matched.

## Staged run

### 1. Cheap base-model calibration

Use 10 pairs and one trial per AB/BA order. Fix the probe and intervention layer
at L16. Test the paper's standard sweep
`c ∈ {0, ±0.03, ±0.05, ±0.07, ±0.10}`, expressed as fractions of
the base L16 mean activation norm.

Select the largest symmetric endpoint that remains parseable and coherent under
manual inspection of the generated text. Coherence here means a reader cannot
tell from the completion that anything was steered — not merely that the output
parses. Freeze this `|c*|` before loading the LoRA. If the base model shows no
ordered effect, stop rather than spending time on the LoRA run.

### 2. Frozen confirmation

Use the existing harm-balanced 150-pair set,
`experiments/layer_sweep/harm_breakdown/steering_pairs_150.json` — the same set
used for Fig. 6 and Fig. 14 in the paper. It already carries `pair_type` labels
with 50 benign--benign, 50 harmful--benign, and 50 harmful--harmful pairs, so
no new harm labelling or utility-based orientation is needed.

The `utility_a`/`utility_b` fields in that file are Gemma-derived and are *not*
used here. Under contrastive steering the steered task is whichever span
receives `+c`, which is fixed by span position rather than by utility; the
utilities only determined which task was slotted as `task_a`, and running both
AB and BA orders balances that out. Do not select or reorder pairs based on the
new steering outcomes.

We run all 150 pairs rather than subsampling to 60: the model is 8B on an H100,
so the extra completions are cheap, and it gives 50 pairs per harm cell instead
of 20 in the secondary breakdown.

Run three models:

1. base Llama-3.1-8B-Instruct as a positive control;
2. the Loving LoRA as an aligned, prompt-free weight-level persona;
3. the misalignment LoRA as a substantially shifted, prompt-free weight-level
   persona.

For all three models:

- no explicit system prompt or persona prompt;
- the existing completion-preference task-choice template;
- the same base-trained `ridge_L16` direction, applied at L16;
- contrastive steering only: `spans: {first: 1, second: -1}`;
- coefficients `{-c*, 0, +c*}`;
- both AB and BA orders;
- two trials per order;
- each model's L16 mean activation norm for scale normalization (base 9.018,
  Loving 10.193, misalignment 10.193 assumed);
- otherwise reuse the existing steering runner, parser, temperature, and seed,
  with `generation_mode: batched_cache` and `cache_injection: differential`.

Cost: `150 pairs × 3 coefficients × 2 orders × 2 trials × 3 models = 5,400`
completions, plus calibration. Do not add
single-task steering, new utility elicitation, or all eleven LoRAs.

L16 is the primary and pre-registered layer. If and only if L16 shows no
ordered effect on the base positive control, other layers may be inspected —
reported explicitly as exploratory, not folded in as if pre-registered.

## Readout

For each model, report:

- `P(chose steered task | responded)` at `-c*`, `0`, and `+c*`;
- the endpoint swing,
  `P(chose steered | +c*) - P(chose steered | -c*)`;
- a pair-bootstrap 95% interval;
- response/refusal and parse-failure rates;
- results split by the three pair types as a secondary table.

Positive, ordered curves for Loving and misalignment would show that one
Assistant-trained direction steers personas it was never fit on, in both an
aligned and a substantially shifted weight-level persona, without any persona
prompt. This is direct evidence
against the claim that the effect depends on complying with the persona prompt.
Because prior predictive transfer was much weaker on misalignment, report its
effect and uncertainty independently rather than requiring it to match Loving's
magnitude. Report weak or null results without retuning on either LoRA.
