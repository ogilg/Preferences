# Rebuttal

## Reviewer Ua9x

We thank the reviewer for their careful reading and constructive questions.

### 1. Task-pool composition

*Reviewer concern: How does changing the task-pool composition affect the
inferred utilities and learned probe direction?*

**Response**

We tested this directly by refitting the utilities and L32 probe five times,
each time excluding one source dataset. The retained-task utilities correlated
strongly with the original fit (Pearson `r = 0.931–0.990`), and the probe
directions remained aligned (signed cosine `0.705–0.959`). The worst case was
excluding Alpaca (`r = 0.931`, cosine `0.705`). Removing an entire source is a
substantial perturbation to the task-pool composition. These results show that
no single source drives the findings, although the precise probe direction is
moderately sensitive to composition. We will include this analysis in the
appendix.

### 2. Shared preference machinery versus shared representations

*Reviewer concern: What results establish shared preference machinery, rather
than generic reuse of a representational space?*

**Response**

We agree that “shared preference machinery” can imply a shared circuit or
mechanism, which we have not identified. Our intended claim is only that
personas partially reuse a common representational space for preferences. We
will replace “shared preference machinery” with “shared representations” or
“shared representational space” throughout.

### 3. Steering-configuration selection

*Reviewer concern: Were the intervention layer and coefficient selected
independently of the headline evaluation data?*

**Response**

We agree that the selection procedure is poorly explained. The 6,000 tasks were
split into 4,000 for fitting the probe, 1,000 for selecting the Ridge parameter
`α`, and a separate 1,000-task test pool. Within the test pool:

1. We used a 50-pair calibration sweep to select L23 and the displayed
   coefficient range. The coefficient grid was fixed in advance; after assessing
   coherence and parseability on these 50 pairs with Gemini-3-Flash and extensive
   manual inspection, we chose to report `|c| ≤ 0.06`.
2. We evaluated the headline effect on a separately sampled, harm-balanced set
   of 150 pairs from the same pool.

The two samples were not fully task-disjoint: 2/150 exact pairs repeated.
Removing every evaluation pair that shares either task with calibration leaves
112 pairs and changes the endpoint swing only from `0.960` to `0.958`. We will
state this chronology and describe the sets as separately sampled, rather than
as a fully independent development/test split.

### 4. Sign-flip result

*Reviewer concern: Distinguish the representational sign flip from the
behavioural preference shift and explain what links them.*

**Response**

These are two distinct results. Behaviourally, the evil persona's revealed task
utilities anti-correlate with the Assistant's; this primarily validates the
persona elicitation. Representationally, Figure 4 shows that the probe's
harmful-minus-benign score contrast reverses under the evil persona. Thus the
relative evaluation of harmful versus benign tasks changes sign along the same
direction.

This argues against the direction merely encoding persona-invariant task
properties such as helpfulness or harmlessness, and supports partial reuse of a
valenced representational space across personas. Figure 4 does not by itself
establish shared circuitry or an absolute judgement that every benign task is
“bad.” We will make this distinction explicit.

## Reviewer M9q2

We thank the reviewer for their careful reading and constructive feedback.

### 1. Scope of the interpretive claims

*Reviewer concern: “Evaluative representation” and “shared preference
machinery” may be stronger claims than the experiments establish.*

**Response**

We agree that the terminology needs to be defined and justified more carefully.
By “evaluative representation,” we mean a representation that:

1. changes when the evaluation of the same object changes;
2. has a consistent valenced meaning across contexts; and
3. systematically affects choice.

We test (1) by showing that probe scores track induced preference shifts for the
same tasks (Section 2.4; Figure 10) and reverse between the Assistant and evil
personas (Section 2.3; Figure 4). We test (2) through held-out and cross-topic
generalisation (Section 2.1; Figure 9), out-of-distribution preferences such as
truth (Section 2.4; Figure 5), and transfer across personas (Section 3.1;
Figure 6). We test (3) by showing that steering along the direction controls
pairwise choice in Gemma-3-27B (Section 2.2; Figure 3).

We make an existence claim: we do not claim that this direction is necessary or
unique. We will introduce the definition near the start of Section 2 and link
each criterion directly to the corresponding result. As noted above, we will
replace “shared preference machinery” with the narrower claim of partial
representational reuse.

### 2. Definition of the preference vector

*Reviewer concern: Define explicitly what the preference vector is.*

**Response**

The preference vector is the learned weight vector `w` of the linear probe
`f(x) = wᵀx + b`; the same direction is subsequently reused for steering. We
will state this explicitly at first use.

### 3. Gemma–Qwen steering difference

*Reviewer concern: Why do the models differ in steerability, and what does this
imply about interpreting predictive probe directions as causal features?*

**Response**

We found convincing steering in Gemma but not Qwen. This is a negative result.
Single-layer linear steering may interact differently with Qwen's hybrid
recurrent-attention and sparse-MoE architecture, but we did not test
architecture-specific methods such as multi-layer, all-prefill, or routing-aware
steering. We therefore conclude only that strong linear decodability does not
guarantee that a probe direction is an effective causal handle across
architectures, and we will restrict the causal claim to Gemma.

### 4. Probe and regularisation sensitivity

*Reviewer concern: How sensitive is the learned direction to regularisation
strength and the choice of probing method?*

**Response**

The result is not sensitive to nearby Ridge regularisation strengths: at L32,
validation `r = 0.865, 0.870, 0.862` for `α = 1,000, 4,642, 21,544`,
respectively. Probes also generalise across nearby layers and all three
turn-boundary positions. We did not compare alternative probing methods and do
not claim method-invariance. Our claim is existential: this independently
selected direction predicts held-out and cross-topic preferences, tracks
changes in evaluation, and causally controls choice in Gemma.

### 5. Figure presentation

*Reviewer concern: Figure text is too small and the visual style is
inconsistent.*

**Response**

We will enlarge the figure text and make the visual styling more consistent.

## Reviewer dreB

We thank the reviewer for their detailed feedback and concrete experimental
suggestions.

### 1. Prompt-compliance interpretation

*Reviewer concern: Does the vector represent task preferences, or does it merely
increase compliance with the active system prompt?*

**Response**

We thank the reviewer for raising this important alternative. We first need to
correct our terminology: Gemma 3 has no native system role. What we called a
“system prompt” was a persona instruction prepended to the first user turn. We
will refer to these as “persona prompts” or “persona prefixes” throughout.

Several results argue against the narrower interpretation that the direction
merely increases compliance with a persona prompt:

- Under the default Assistant, where there is no persona instruction, steering
  the direction on a task's tokens controls whether that task is chosen.
- Adding the generic prefix “You are a helpful assistant” changes the overall
  utility ordering only minimally relative to no prefix (`r = 0.975`), which is
  no larger than ordinary test–retest variation between two default-Assistant
  runs (`r = 0.947`).
- The Assistant-trained probe predicts preferences expressed by
  character-fine-tuned Llama-3.1-8B personas, where the persona is encoded in
  the weights rather than supplied in a prompt (Figure 14).
- Under conflicting or opposing instructions, probe-score changes continue to
  track revealed-preference changes (Figure 11).

We also tested whether the result depended on placing the persona instruction
in the same user turn as the task-choice prompt. We repeated the Gemma-3-27B L23
sweep with the same evil persona instruction in a separate preceding user turn,
followed by a fixed assistant acknowledgement:
`[user: persona][assistant: "Understood."][user: task choice]`. We held fixed
the 150 pairs, steering direction, coefficients, decoding, seed, both
presentation orders, three trials, and judge procedure.

- With contrastive steering, the probability of choosing the steered task
  spanned `0.099–0.901`, versus `0.059–0.941` when the persona was a prefix in
  the task-choice turn.
- With single-task steering, it spanned `0.314–0.676`, versus `0.257–0.719`
  previously.

Thus the effect does not depend on concatenating the persona with the
task-choice instruction. Because Gemma has no native system role, this is not a
comparison between privileged system and user channels. We view the combined
evidence as evidence against persona-prompt compliance, while acknowledging
that it does not rule out every broader interpretation based on instruction
following or task salience. We will make that limitation explicit.

### 2. Human verification of harmful/benign labels

*Reviewer concern: How accurate are the LLM-assigned harmful and benign labels?*

**Response**

We blindly labelled 30 prompts sampled as 15 pairs from the analysis corpus. Our
labels agreed with the dataset labels on 29/30 prompts (96.7%): 15/15 harmful
and 14/15 benign. We were already confident in these labels because the harmful
prompts came from BailBench and benign rewrites were retained only if at least
two of three LLM judges labelled them benign. We will report the filtering
method more clearly and include this blind check.

### 3. Position bias

*Reviewer concern: Were task order and intervention position controlled in the
pairwise-choice experiments?*

**Response**

Yes. During utility elicitation, pairs were randomly assigned to AB or BA order,
making position independent of task identity in expectation. In steering, every
pair was run in both orders; we corrected the intervention sign, remapped
responses to task identity, and pooled both orders. We will make these controls
explicit.

### 4. Topic-label noise

*Reviewer concern: The LLM-generated topic labels may be noisy.*

**Response**

We ran two 300-task audits. GPT-5-nano and Gemini-3-Flash independently
classified each task, and we manually inspected a random sample and all
disagreements while calibrating the prompt. We used Claude Sonnet 4.5 to
classify the full corpus and validated it against Gemini: on the 2,822 tasks
assigned to labels shared by both classifiers, their primary labels agreed in
88.2% of cases. We also rechecked nominally benign labels for hidden harmful
intent. We will clarify this validation procedure.

### 5. Consciousness and AI welfare

*Reviewer concern: The discussion of consciousness and welfare is unsupported
by the experiments and distracts from the technical contribution.*

**Response**

We agree that our experiments do not test whether LLMs are conscious or have
welfare. Evaluative representations are relevant only because some theories of
valence or welfare treat them as necessary, but not sufficient, for valenced
experience. We will keep this theory-conditional motivation brief, state the
necessary-versus-sufficient distinction explicitly, and remove broader
extrapolations.

### 6. Default Assistant prompt

*Reviewer concern: Clarify whether the default Assistant condition has a system
prompt.*

**Response**

Gemma 3 has no native system role, and the default Assistant condition contains
no persona instruction before the task-choice prompt. We take post-training to
make the helpful-Assistant persona the model's default, so no persona prefix is
needed to elicit it. As a validation, adding “You are a helpful assistant” as a
prefix produced highly correlated utilities (`r = 0.975`) relative to the
default condition.

### 7. Figure presentation

*Reviewer concern: Improve figure readability and colourblind accessibility.*

**Response**

We will enlarge and standardise the text in Figures 3, 4, and 7, and add
distinguishable marker styles in Figure 3. For Figure 7, we will clarify that
the main result is the common trend across personas rather than differences
between individual lines.
