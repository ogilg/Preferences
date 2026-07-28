# Reviews for Submission 34568

## Meta-review

**Area Chair:** GyNw
**Submitted:** 23 Jul 2026 at 12:58
**Modified:** 23 Jul 2026 at 19:37

### Meta-review

The paper trains a linear probe on residual-stream activations to predict utilities derived from revealed pairwise task choices, finds a preference vector that predicts held-out and cross-topic utilities, and causally controls pairwise choice via steering. Interestingly, it transfers across prompted personas, including an evil persona whose utilities anti-correlate with the Assistant's.

All reviewers find the question well-motivated, and two of three consider the experimental program thorough. Nevertheless, the current ratings are split (4/4/2) where the negative reviewer (2) is the most confident (4). Therefore, addressing the concerns raised by the negative reviewer is a key for the acceptance, while addressing others are also important that includes the followings:

1. Alternative interpretation: is this a "preference" vector or a "system-prompt compliance" vector?: Reveiwer dreB points to the paper's own lines 59–61 (steering makes the evil persona more evil but has no effect on the default Assistant, which has no system prompt) as direct evidence for the competing reading. The experiment in Appendix E does not resolve this, since Qwen is separately shown (App. F.4) not to be steerable. Reviewer Ua9x raises the same worry from the transfer side.

2. Methodological controls: steering-configuration selection and positional bias: Reviewer Ua9x notes it is unclear whether the intervention layer and coefficient were selected independently of the data on which the main causal effects are reported; this matters because the paper itself reports the steering peak differs from the probe peak, so layer choice largely affects the results. Please state the selection protocol and whether a separate development set was used. Reviewer dreB also asks whether positional bias in the A/B task-choice prompt was controlled and for human verification of the LLM harmful/benign labeling on a small subset, noting addressing these concerns would improve their evaluation.

3. Overclaim relative to what the experiments establish: Reviewer M9q2 argues the manuscript repeatedly presents interpretive conclusions such as "evaluative representation," "personas share preference machinery" as if established, when the empirical content is that a learned linear probe predicts utilities and functions as a steering direction. Reviewer dreB separately objects that the AI-welfare/consciousness discussion is speculation unsupported by any experiment. Reviewer M9q2's related clarity point that it takes effort to realize the "preference vector" is simply the probe's weight vector should also be fixed with an explicit definition.

---

## Official review — Ua9x

**Reviewer:** Ua9x
**Submitted:** 14 Jul 2026 at 09:15
**Modified:** 23 Jul 2026 at 18:03

### Summary

The paper studies whether LLMs encode revealed task preference in internal representation and whether such representations further transfer across personas. Experiments show predictive generalization and causal steering effects in Gemma-3-27B. Paper further reports that an Assistant-trained preference probe transfers to several prompt based personas, suggesting partial reuse of evaluative representations.

**Contribution type:** General: Most submissions will fall into this type.

### Strengths and weaknesses

#### Strengths

The paper connects revelaed behavioral preferences, internal representations and causal interventions in coherent experiments.

The extensive cross topic, persona conditioned and steering experiments provide stronger evidence than probe based correlation analyses alone.

The distinction between linear decodability and causal efficacy across layers is interesting and provides useful insight for representation based analysis.

#### Weaknesses

- The sensitivity of resulting preference direction, which is based on inferred task utilities to task-pool composition or sampling distribution is unclear. Leave one topic out evaluation tests predictive generalization, but does not directly address the stability of the utility target itself.
- Authors’ claim that cross-persona probe transfer alone establishes shared preference machinery is not fully convincing. Since most personas are induced by system prompts in the same frozen model, the observed transfer may also reflect generic reuse of a common semantic or evaluative representation space. Weaker transfer for weight-level personas further motivates this concern.
- The protocol used to select steering layer and operating magnitude is not sufficiently clear. It is difficult to determine whether these configurations were selected independently of the data used to report the main causal effects.
- The interpretation of the reported sign flip is somewhat ambiguous. The sign flip appears to refer to the relative probe scores of harmful and benign prefilled responses, rather than a direct reversal of pairwise behavioral choices in the same experiment.

### Scores

| Criterion | Score |
| --- | --- |
| Quality | 3: good |
| Clarity | 3: good |
| Significance | 3: good |
| Originality | 3: good |

### Questions

1. How sensitive are the inferred utilities and the learned preference direction to the composition of the task pool? Cold the authors report an ablation with different task mixtures or sampling distribution and compare the resulting probe directions and cross-pool transfer?
2. What evidence specifically distinguishes shared preference machinery from generic representational reuse among prompted personas in the same underlying model?
3. Were the intervention layer and steering coefficient used for the main result selected on a separate development set? Please clarify the selection protocol for steering layers and magnitude.
4. Could the authors more clearly distinguish the representational sign flip in probe scores from the behavioral preference shift? In particular, what direct evidence links the two observation?

### Assessment

**Limitations:** yes
**Rating:** 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
**Confidence:** 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
**Ethical concerns:** NO or VERY MINOR ethics concerns only
**Paper formatting concerns:** None
**Code of conduct acknowledgement:** Yes
**Responsible reviewing acknowledgement:** Yes

---

## Official review — M9q2

**Reviewer:** M9q2
**Submitted:** 27 Jun 2026 at 11:59
**Modified:** 23 Jul 2026 at 18:03

### Summary

The authors use linear probes to identify a preference vector, i.e., linear weights which predict task utilities from residual-stream activations and can be used as a steering direction to influence the model's pairwise task choices. They show that this direction generalizes across tasks, preference manipulations, and prompted personas, and argue that it constitutes a shared evaluative representation underlying persona-dependent preferences.

**Contribution type:** General: Most submissions will fall into this type.

### Strengths and weaknesses

The paper does a reasonably good job of empirically evaluating what I believe is the main contribution: establishing whether preferences are persona dependent. The steering experiments are useful in particular as they provide causal evidence beyond simple probing. I would however have liked to see a broader cross-model evaluation. As it is, it is not clear whether steerability is specific to the gemma model you tested or how it depends on model architecture or scale.

The main weakness is clarity. The paper would benefit from a clearer problem formulation and more precise definitions of its key concepts. Maybe it is just me, but it took some effort to infer that the preference vector is simply the learned weight vector of the linear probe rather than an activation or an independently discovered latent representation. I also think the manuscript occasionally overstates what the experiments establish. The empirical results demonstrate that a learned linear probe predicts utilities and can be an effective steering direction. However, the paper often presents stronger conclusions (for example, that it has identified an evaluative representation or that personas share preference machinery) without sufficiently separating these interpretations from the empirical observations. While these are plausible hypotheses, the experiments do not sufficiently establish them.

Finally, I would suggest making the figures more visually consistent, as the illustration style varies significantly across figures. For readability I would also increase the font size in the figures such that it matches the size of the captions.

### Scores

| Criterion | Score |
| --- | --- |
| Quality | 3: good |
| Clarity | 2: not good |
| Significance | 3: good |
| Originality | 3: good |

### Questions

1. Do you have a hypothesis for the difference in steerability of qwen and gemma? Is it architecture, model size, something else? What does it imply about interpreting probe directions as causal features?
2. Since multiple probe directions could achieve similar performance in high-dimensional activation spaces, how sensitive are the results to regularization, or the choice of probing method?

### Assessment

**Limitations:** yes
**Rating:** 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
**Confidence:** 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
**Ethical concerns:** NO or VERY MINOR ethics concerns only
**Paper formatting concerns:** none
**Code of conduct acknowledgement:** Yes
**Responsible reviewing acknowledgement:** Yes

---

## Official review — dreB

**Reviewer:** dreB
**Submitted:** 25 Jun 2026 at 23:38
**Modified:** 23 Jul 2026 at 18:03

### Summary

This paper studies LLM preferences and their relation to personas, which can be induced by different system prompts. The authors study two open source LLMs, Gemma-3-27B and Qwen-3.5-122B, and train linear probes on their activations to discover a universal preference vector which can be used to track the LLM’s preferences across different personas. When an LLM is presented with two tasks, steering using this preference vector controls the LLM’s task selection. This result is significant because it holds across multiple personas, including an evil’’ persona which prefers to complete harmful tasks and has preferences that are anti-correlated with the LLM’s default helpful’’ behavior.

**Contribution type:** General: Most submissions will fall into this type.

### Strengths and weaknesses

#### Quality

The authors study two research questions: whether LLMs use evaluative representations (value-based judgements) as opposed to purely descriptive representations, and whether different personas share the same representational machinery for preferences.

The authors carefully investigate how the preference vector controls pairwise choice between tasks. The experimental results in this section support the analysis.

One concern with the experimental design is that LLM judges can exhibit positional biases (preference for the first or last answer). The authors should randomize the order of tasks in their prompt to control for this bias. (See the question later.)

In the later sections of the paper, the authors engage in speculation about the risks of AI being conscious and capable of making value-based judgements due to the existence of consistent preferences for one task over another. Since there is no evidence that AI is conscious, this speculation cannot be supported by experiments and distracts from the technical analysis of the preference vector.

Additionally, there are some unanswered questions about preference vectors.

The authors compare their linear probe approach against a text-encoder approach that fails to achieve the same universal compatibility with different preferences. This shows that the linear probe design is necessary. The experiments are conducted on popular peer-reviewed datasets.

A minor concern is that the experiments use LLM judges to assign topics to the dataset samples. The LLM judge first produces a list of topics for each dataset based on a subsample of the data, and then assigns one topic to each sample. It’s likely these labels are noisy due to the variability of LLM judges. However, as the primary goal of the paper is to study the existence of preference vectors, this is not a significant issue.

LLM judges are also used to categorize prompts as harmful or benign, a labeling that contributes directly to the analysis of different personas. This labeling is double-checked by prompting the LLM to recheck benign samples for potentially harmful content. The paper posits the existence of a universal preference vector. However, another interpretation for these results is that there is a universal vector that increases compliance with the system prompt. In lines 59-61, for example, the authors report that steering makes the ``Evil’’ persona more evil, but has no effect on the default Assistant (which has no system prompt).

In Appendix E, the authors study whether a probe trained on a base Qwen3.5-122B model can steer a model fine-tuned to have the Evil’’ persona instead of a model where the Evil’’ persona is induced via system prompt. The results of this experiment are unsuccessful.

However, in other sections of the paper such as Appendix F.4, the authors find that they cannot steer Qwen3.5-122B with the same statistical significance as they can for Gemma3-27B. Additionally, fine-tuning may change the activations such that the linear probe from the base model no longer works. Thus, the fine-tuning experiment on Qwen in Appendix E thus does not rule out the possibility that the preference vector instead serves to increase compliance with the system prompt on the more steerable model, Gemma3.

#### Clarity

The paper is well-written. Figures are clearly designed and serve to communicate the experimental results. Contributions are clearly explained. Extensive information is provided in the appendices to support replicating the results.

Minor concerns:

- The fact that the “default Assistant” has no system prompt is not mentioned in the main paper. Typically, the default system prompt is “You are a helpful assistant” so clarifying this would improve the paper.
- The text on Figures 3, 4, and 7 is small and difficult to read. This issue is also present in some of the figures in the Appendices.
- In Figures 3 and 7 , the use of different marker styles would make the plots accessible to colorblind readers. This issue is also present in a few Appendix figures (for example, Figures 12 and 15).

#### Significance

The question of how prompt-induced LLM personas work is not only scientifically interesting, it has practical implications for linear probe methods used for LLM safety. Thus, this work is likely to be of interest to both LLM personalization and mechanistic interpretability researchers.

The discovery of a universal preference vector that amplifies the effect of LLM personas by controlling the LLM’s choices is interesting and has practical applications. However, the paper only shows a steering effect on Gemma3-27B, not on Qwen3.5-122B. The paper does not establish the existence of an effective steering vector for other models, reducing the significance of the results.

#### Originality

The work studies two questions about how prompt-induced personas work in LLMs. The authors draw on recent methods including linear probes and situate their work clearly in the literature. The experimental design of allowing the LLM to choose between two tasks seems to be novel. This work provides a novel analysis of LLM personas and increases understanding of how LLM preferences are induced.

### Scores

| Criterion | Score |
| --- | --- |
| Quality | 2: not good |
| Clarity | 3: good |
| Significance | 2: not good |
| Originality | 3: good |

### Questions

1. Is it possible to rule out whether the preference vector is actually a vector that increases compliance with the system prompt? For example, if the system prompt was empty and the user prompt provided the persona, would the steering effect still work? Conducting this experiment (or a similar one to investigate this alternate interpretation) with Gemma3-27B, which showed the best steering results, would improve my evaluation of the paper. However, it may be beyond the scope of the review period.
2. How accurate is the LLM labeling for harmful vs benign tasks? Identifying a small subset for human verification and reporting the results would serve to increase confidence in this labeling method. If this experiment was conducted and LLM labeling shows high accuracy, this would improve my evaluation of the paper.
3. Did the existing experiments control for positional bias in the task selection prompt? Positional bias is briefly mentioned in the caption of Figure 7 but it’s not clear whether it has been addressed (for example, by swapping tasks A and B as well as swapping whether the steering vector is applied to A or B). If not, could a small experiment measuring this bias be conducted? Addressing this potential bias would improve my evaluation of the paper.

### Assessment

**Limitations:** yes
**Rating:** 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
**Ethical concerns:** NO or VERY MINOR ethics concerns only
**Paper formatting concerns:** none
**Code of conduct acknowledgement:** Yes
**Responsible reviewing acknowledgement:** Yes
