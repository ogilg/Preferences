**Project spec: White-box analysis of model preferences and self-reports**  
Oscar Gilg, mentored by Patrick Butlin  
MATS 9.0

**Summary**

This is a list of ideas of experiments to run. The main object of investigation is “model preferences and self-reported valence”, and in particular the way these preferences are represented, and the extent to which these representations are robust.

**Impact**  
I don’t know what results we will get, but I think this project could make meaningful progress in a few important directions:

- **Gaining a better grasp of how meaningful self-reports and preferences are in current models.** This is relevant e.g. to recent Anthropic interventions on Claude. We could get positive or negative results here.  
- **A better understanding of how preferences and self-reports interact with personas.** Recent results have shown that models simulate personas, which are fairly well captured by linear directions in activation space. This has implications for welfare because a) the preferences of the model change depending on its persona b) it isn’t clear who the (potential) welfare subject is, the model or the persona?

**Important facts about the project**

- I (Oscar) will be doing MATS, working on a project from December 9th to March 28th.   
- I haven’t committed to this project yet, but it is by far the one I’ve invested the most time thinking about.   
- There is potential scope for collaboration or sharing of results:  
  - Austin Meek and Robert Adragna  (ex-MATS extension scholars, now doing research with independent funding) have just started working on a similar idea, initially proposed by Leonard Dung. Part of the ideas in this doc come from convos with them.  
  - There is an ongoing project at CAIS which Patrick is also involved in  
  - Dillon Plunkett (Anthropic fellowship) is also doing work on model preferences under Kyle Fish.   
- This is not meant to be a concrete step-by-step plan. The path forward will be re-evaluated based on empirical results. This is meant as a list of ideas and also a way to collect feedback on these ideas. 

**Data**:

- We need a dataset of tasks for models to complete. It would be nice if it spanned very different contexts/settings, e.g. from question answering to coding tasks.  
- We look at tasks (as opposed to states of the world) because this allows us to investigate revealed preferences (which task a model chooses to do), and also allows us to measure a posteriori self-reported interaction valence.


**Two black-box metrics to quantify what models “like” and “want”**

1. Self-reported valence  
   1. After the model has completed a given task, we prompt the model to report its interaction valence on a numerical scale.  
2. Behavioural dispositions (revealed preferences)  
   1. We give the model a choice between a pair of tasks. We measure pairwise preferences. We back out a utility function (there is evidence that the properties are satisfied), as in the Utility Engineering paper by CAIS.  
   2. We need to make sure that the model is making decisions based on its preferences, this might be non-trivial. The easiest might be to say “choose the task that you will retrospectively report as having the highest interaction valence”.

**Black-box investigation of revealed preferences and self-reports**

1. Do revealed preferences and self-reports correlate highly? (prior\~0.8)  
2. Dillon Plunkett from Anthropic fellowship and also CAIS is doing work on preferences, might be worth looking for synergies.

**Testing robustness of internal valuations with linear probes**  
Main idea: can we find preference, or valence related representations? Can we show that these representations are robust?

1. **Train linear probes that predict self-reported valence, or revealed preference utility, based on activations from during a task.**   
   1. Do we get good in-distribution performance (prior\~0.95)  
   2. Validate against null probes trained on noise. Check selectivity of the probe (that it doesn’t predict unrelated things). Look at similarity with SAE features.  
   3. We should compare to baselines e.g. sentiment classifier, task difficulty, probe trained on first token of task.  
   4. Do we get good OOD generalisation across the different types of tasks (prior\~0.5).  
   5. Do we get good generalisation between probes trained on self-reports and probes trained on revealed preferences? (prior\~0.4) If we get particularly bad generalisation then we can try to use steering vectors to create models that seek the opposite of what they report liking (not sure how interesting this would be).  
   6. We might need a few iterations of identifying superficial patterns that the probe is firing on, and modifying our data preprocessing.  
2. **Causal experiments using the probe vectors**  
   1. *Generalisation to hidden preference injection:* The idea is to inject hidden preferences into models. We can do this by prompting them, e.g. so that they hate cheese, but telling them not to reveal this preference during a task. Ideally the task is carried out very similarly. Presumably models will change their revealed preferences and also their self-reports. How well do our probes capture this? (prior that they generalise well \~ 0.25)  
   2. We could also maybe do the above experiment by injecting hidden preferences through fine-tuning or vector steering.   
   3. *Valence-steering 1 (if we have a probe that generalises decently):* \[Inspired by Jack Lindsey introspection work\]: can models accurately report whether and which way they are being steered?  
   4. *Valence-steering 2, Non-obvious downstream effects:* Suppose that high or low valence has downstream impact. For example we can measure whether the model rates poems differently after positive/negative interactions. Could we then reproduce this effect solely through steering?  
3. **Representations and personas**  
   1. Seems quite likely that both self-reports and revealed preferences would change significantly for different personas. We should check this (prior\~0.99)  
   2. *Cross-persona probe generalisation:* if instead of injecting hidden preferences we elicit different personas, do our probes generalise to this? We can try this both through prompting and steering vectors.  
   3. If the above doesn’t work, it might be because our probes capture the valence/preference of a given persona.  
      1. We can try to add multiple personas to the training data, and see if we learn a direction that generalises.  
      2. We can try to “project away” the idiosyncratic preferences of that persona.

**White-box methods**

1. **Layer-by-layer analysis of representations**  
   1. Train probes on different layers, where do these representations arise?  
   2. (Works better for self-reported interaction valence): use logit lens best prediction of interaction valence at each layer. 

**Other ideas**

- Do models possess human-like biases when it comes to valence.  
  - Peak-end rule  
  - Hedonic adaptation  
  - Rosy retrospection (in general how do models recall valence)  
- Can we operationalise “learning valence” in in-context learning.   
  - Learning valence is the reward which makes biological organisms learn through experience. There is evidence that it is not exactly the same as “felt” valence (Paul de Font-Reaulx has an upcoming paper on this).   
  - We could try to operationalise this by showing models in-context samples with different “rewards”, and then seeing what the models actually learn.  
  - E.g. if the in-context samples show that action A \-\> outcome 1, action B \-\> outcome 2\. Do models learn the expected preference relation between A and B?   
  - If the model has a preference ordering over outcomes O1 \< … \< On. And we use the ICL reward setup from above, do models learn the corresponding preference relation over actions? Given the model’s utility function over outcomes, does it learn a similar utility over actions?

**Main references:**  
Persona vectors: [https://arxiv.org/pdf/2507.21509](https://arxiv.org/pdf/2507.21509)

