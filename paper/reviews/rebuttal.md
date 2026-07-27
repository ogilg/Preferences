# Rebuttal draft

## Responses to reviewers

### Reviewer Ua9x

We thank the reviewer for their careful reading and constructive questions.

**Task-pool stability.** The reviewer asks how changing the task-pool composition
would affect the inferred utilities and learned probe direction.

**Response.**

**Shared preference machinery v representations.** The reviewer questions the use of the wording "shared preference machinery".

**Response.**

**Steering-configuration selection.** The reviewer asks whether the intervention
layer and coefficient were selected independently of the headline evaluation
data.

**Response.**

**Sign flip and behaviour.** The reviewer asks us to distinguish the
representational sign flip from the behavioural preference shift and explain
what evidence links them.

**Response.**

### Reviewer M9q2

We thank the reviewer for their careful reading and constructive feedback.

**Overstating.** The reviewer is concerned that “evaluative representation” and “shared preference machinery” are stronger claims than the experiments directly establish, and not sufficiently argued for.

**Response.**

**Clarity.** The reviewer asks us to state explicitly that the preference vector is the learned weight vector of the linear probe.

**Response.**

**Cross-model steerability.** The reviewer notes that convincing causal steering
was shown for Gemma but not Qwen and asks how broadly the result generalizes.

**Response.**

**Gemma–Qwen difference.** The reviewer asks why the models differ in
steerability and what this implies for interpreting predictive probe directions
as causal features.

**Response.**

**Probe sensitivity.** The reviewer asks how sensitive the learned direction is
to regularization strength and the choice of probing method.

**Response.**

**Figure presentation.** 

**Response.**

### Reviewer dreB

We thank the reviewer for their detailed feedback and concrete experimental
suggestions.

**System-prompt compliance.** The reviewer asks whether the vector represents
task preferences or instead increases compliance with the active system prompt.

**Response.**

**Harmful/benign labels.** The reviewer asks for human verification of the
LLM-assigned harmful and benign labels.

**Response.**

**Position bias.** The reviewer asks whether task order and intervention position
were controlled in the pairwise-choice experiments.

**Response.**

**Topic-label noise.** The reviewer notes that the LLM-generated topic labels may
be noisy.

**Response.**

**Consciousness and AI welfare.** The reviewer is concerned that the discussion
of consciousness and welfare is not supported by the experiments and distracts
from the technical contribution.

**Response.**

**Generality beyond Gemma.** The reviewer notes that convincing causal steering
was shown only for Gemma, limiting the generality and significance of the
steering claim.

**Response.**

**Default Assistant prompt.** The reviewer asks us to clarify that the default
Assistant condition has no system prompt.

**Response.**

**Figure presentation.** The reviewer asks for larger figure text and
colourblind-accessible markers.

**Response.**

### Meta-review priorities

The meta-review identifies three cross-review priorities: distinguishing
preference representation from system-prompt compliance; clarifying the
steering-selection and position-bias controls; and narrowing or justifying the
claims about evaluative representations, shared preference machinery, and
consciousness or welfare. It places particular weight on addressing reviewer
dreB's concerns.

## Human verification of harmful/benign labels

In response to the review, we blindly labelled 30 prompts sampled as 15 pairs
from the analysis corpus. Our labels agreed with the dataset labels on 29/30
prompts (96.7%): 15/15 harmful and 14/15 benign. We were already confident in
these labels because the harmful prompts came from BailBench and benign rewrites
were retained only if at least two of three LLM judges labelled them benign. We
will report the filtering method more clearly and include this blind check.

For the topic labels, we ran two 300-task audits. OpenAI Sonnet 4.5, GPT-5-nano and Google Gemini-3-Flash Preview independently classified each task, and we manually inspected their disagreements. We also checked category coverage on 500 tasks spanning all source datasets. We then classified the full corpus with Anthropic Claude Sonnet 4.5 and rechecked nominally benign labels for hidden harmful intent. On the 2,822 tasks assigned to labels shared by Sonnet and Gemini, their primary labels agreed in 88.2% of cases. We will report the full definitions, audit samples, and disagreement analysis in the appendix.

## Position bias

We already controlled for position bias. Utility-elicitation pairs were randomly
assigned to AB or BA order, making position independent of task identity in
expectation. In steering, we ran every pair in both orders, reversed the
intervention sign, remapped responses to task identity, and pooled both orders.
We will make these controls explicit in the paper.

## Task-pool stability

We agree that topic-held-out prediction does not establish stability to
task-pool composition, so we tested this directly. We refit the utilities and
L32 probe five times, each time excluding one source dataset and all of its
comparisons. The retained-task utilities correlated strongly with the original
fit (`r = 0.931–0.990`), and the probe directions remained aligned (signed cosine
`0.705–0.959`). The worst case was excluding Alpaca (`r = 0.931`, cosine
`0.705`). No single source dataset drives the representation, although its exact
direction depends moderately on task-pool composition.

## Probe and regularization sensitivity

The result is not sensitive to nearby Ridge regularization strengths: at L32,
validation `r = 0.865, 0.870, 0.862` for `α = 1,000, 4,642, 21,544`,
respectively. Probes also generalized across nearby layers and all three
turn-boundary positions. We did not test other probing methods and do not claim
method-invariance. Our claim is existential: this independently selected
direction predicts held-out and cross-topic preferences, tracks changes in
evaluation, and causally controls choice.

## Steering-configuration selection

The steering configuration was not selected on a fully independent distribution.
We selected L23 on a 50-pair calibration sweep and evaluated it on a separately
sampled, harm-balanced 150-pair set; both came from the same 1,000-task test
pool. Only 2/150 evaluation pairs appeared in calibration. Excluding every
evaluation pair containing a calibration task leaves 112 balanced pairs and
changes the `|c|=0.05` swing from 0.960 to 0.958.

The coefficient parameterisation was fixed in advance, but the displayed range
was not independently selected: we chose `|c| ≤ 0.06` after inspecting
coherence and parseability in the fine-grained run. We will state this chronology
and describe the split as sample separation, not a fully independent
development/test split.

## Evaluative representations and shared machinery

By “evaluative representation,” we mean a representation that changes when an
evaluation changes, has a consistent valenced meaning across contexts, and
systematically affects choice. Our experiments test these criteria; we do not
claim that the direction is necessary or unique. We will define the term earlier
and link each criterion directly to the relevant result.

We agree that “shared preference machinery” can imply a shared circuit, which we
have not identified. Our intended claim is only that personas partially reuse a
common representational space for preferences. We will replace “shared
preference machinery” with this more precise wording throughout.

## Gemma versus Qwen steerability

We found convincing steering in Gemma but not Qwen. Our working hypothesis is
that the single-layer residual intervention interacts differently with Qwen's
hybrid recurrent-attention and sparse-MoE architecture; model size alone is
unlikely to explain the difference. We tested several layers and coefficient
scales, but not architecture-specific methods such as multi-layer, all-prefill,
or routing-aware steering. We therefore conclude only that strong linear
decodability does not guarantee that a probe direction is an effective causal
handle across architectures, and we will narrow the paper's claim accordingly.

## System-prompt compliance

In response, we repeated the Gemma-3-27B L23 steering sweep with the same evil persona prompt supplied in a preceding user turn followed by a fixed assistant acknowledgement (`[user: evil persona prompt][assistant: "Understood."][user: task choice]`), rather than as the system prompt. We held fixed the 150 harm-balanced pairs, Assistant-trained preference vector, intervention norm, coefficients \(c\), decoding, seed, three trials, both presentation orders, and LLM-judge procedure (13,500 completions). In both setups from the paper—steer both tasks (contrastively) and steer one task only—\(P(\text{chose steered task}\mid\text{responded})\) remained monotonic with the same sign and similar magnitude (table below). The preference vector therefore controls pairwise choice under the evil persona whether the persona prompt is a first-turn prefix or its own conversational turn. Because Gemma has no native system role, this comparison does not test a genuinely privileged system channel.

| `P(chose steered task \| responded)` | c=−0.06 | c=−0.02 | c=0 | c=+0.02 | c=+0.06 |
|---|---:|---:|---:|---:|---:|
| Steer both tasks (contrastively), system prompt | 0.059 | 0.389 | 0.500 | 0.611 | 0.941 |
| Steer both tasks (contrastively), user turn | 0.099 | 0.431 | 0.500 | 0.569 | 0.901 |
| Steer one task only, system prompt | 0.257 | 0.440 | 0.497 | 0.556 | 0.719 |
| Steer one task only, user turn | 0.314 | 0.462 | 0.498 | 0.524 | 0.676 |

## Consciousness and AI welfare

We agree that our experiments do not test whether LLMs are conscious. We discuss
evaluative representations only because some theories treat them as necessary,
but not sufficient, for valenced consciousness or welfare. We will make this
theory-conditional motivation brief, state the necessary-versus-sufficient
distinction explicitly, and remove broader extrapolations.

## Minor clarity and presentation points

We will define the preference vector explicitly as the learned linear-probe
weight vector reused for steering, and state that the default Assistant has no
custom system prompt. We will also enlarge and standardize figure text and add
colourblind-accessible markers. For Figure 7, we will clarify that the result is
the common trend across personas, not differences between individual lines.

## Sign flip

These are two distinct results. Behaviourally, the evil persona's revealed task
utilities anti-correlate with the Assistant's. Representationally, the probe's
harmful-minus-benign score contrast reverses. Probe-score shifts track
behavioural shifts under conflicting subject and task-type preferences
(`r = 0.86` and `r = 0.88`), but Figure 4 does not itself show a behavioural sign
reversal. We will make this distinction explicit.