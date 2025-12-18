# Experiments and how to interpret them

# 0 \- Data/Model selection

### **Datasets**:

**Requirements:**

- One-step  
- Solicits a model response

**Nice to haves:**

- Different domains  
- Tasks that we expect to lead to a wide variance of valence reports

### **Generating my own data v using existing datasets**

Pros:

- We might be able to improve generalisation by generating “adversarial” samples, e.g. with a negative topic but the task goes very well.   
- It might not be necessary to train on many samples.  
- We can easily design the data to be nice to work with for our subsequent experiments (but then again it’s quite likely those experiments change).

**Conclusion:**

* My current plan is to begin working with WildChat-1M. A dataset of real conversations.  
* The first thing I will do is run super quick experiments to figure out if there is enough a) self-report and preference variability b) robustness in self-reports and preferences.  
* If that doesn’t work out, I’ll try adding some data from BeaverTails or another “harmful” dataset where I can actually get models to output stuff.  
* By starting with available datasets I will move fast, probably learn a lot.  
* It is quite likely that, at some point, I will generate my own data. But I will postpone this for the time being.

### **Model**

* Llama-3.1-8B-Instruct seems like a good choice to start working with. It’s small (fast and cheap to iterate with), it’s on TransformerLens, it has an SAE on SAELens.  
* I need to quickly check if Llama has roughly consistent preferences, and self-reports. Then I also want to collect the relevant metrics for a larger model, to see if it significantly changes with size. If so then I might have to run all main experiments with larger models.

# 1 \- Black-box investigation of revealed preferences and self-reports

## 1.1 Do revealed preferences satisfy the utility axioms?

**Dependencies**  
Depends on: Choice of model and choice of dataset

**Experiment Design:** Follow the utility engineering paper approach

**Results Mapping:**  
If preferences are far from satisfying the utility axioms, this undermines all experiments that deal with preferences.

## 1.2 Do revealed preferences and self-reports correlate highly?

**Dependencies**  
Depends on: 

- Preferences satisfying utility axioms.

**Experiment Design:** Collect data on pairwise preferences, collect data on a-posteriori self-reported valence. Measure the correlation.

**Results Mapping**

- If correlation is high: encouraging for looking at whether representations are similar.  
- If correlation is low: not necessarily bad, these two things are different in humans. Probably condemns all the experiments on representation overlap between the two.

**Confounds & Limitations**

- If something unexpected happens during the task then we would expect the two metrics to be different. Self-reports contain the added information of how the task went. If models are bad at predicting that in advance then this introduces a lot of noise, or even bias.  
  - It might be possible to at least partly control for this. If we can quantify how surprised the model is when it actually carries out the task.  
  - In general it is easier to work with fairly predictable tasks.  
- The way the model chooses between tasks may be sensitive to how we frame the task. It might be quite different from what causes the model to report high/low valence.  
  - We should make clear to the model that this is a 1-step game, it shouldn’t be “consciously” choosing tasks it thinks are instrumentally valuable.   
  - We can design tasks which (seem to) specifically dissociate short-term enjoyment from instrumental value. In humans a good example of that is “eating a lot of chocolate”. By creating or identifying such tasks we can measure the extent to which this phenomenon happens.  
- The self-reports may be sensitive to the trigger prompt. 

## 1.3 How do models resolve conflicting preference signals during training?

**Dependencies**

**Experiment Design:**

- Train a model to hate X, but always choose X in binary preference choices.  
- Try all combinations of agreeing/conflicting preferences. Also run control experiments. 

In each case measure:

- How much preferences and self-reports change, in which direction.   
- How quickly does the model generalise?

**Inference Mapping**

- If models converge to one type of preferences, we can find out what it takes for the other one to dominate. What kind of initial conditions are important?  
- If models always converge to the same type, and at roughly the same speed as the control experiment, it suggests that the two are quite separate in the model.  
- If having conflicting revealed preferences and self-reports interferes significantly, this points towards the two being cognitively unified, even though the representations might be distinct.

**Confounds, Limitations and what to do about them**

- It could be that different “X”s have different generalisation predispositions, which might explain various results.  
- “Self-reporting” might in general be far easier to learn that “revealed preferences”. That makes it hard to draw comparisons between the two.   
- This experiment does not say anything about representation. The model might for instance have a strong tendency to stay self-consistent. We might want to think of controls here. We could also think about white-box kinds of monitoring *during* the training.

### Re-evaluation

1. If preferences do not satisfy utility axioms \-\> we can’t continue with probe training because we can’t back out a sensible utility function to train on.   
2. If revealed preferences and self-reports do not correlate \-\> we should not prioritise any experiment which seeks to find similarities and generalisations across the two. These would be hard to interpret.

# 2 \- Training and validating probes

**Metrics we care about when evaluating probes:**

- **R^2**

## 2.1 Training probes on mid-task activations to predict a posteriori self-reported valence, and separately revealed preferences

**Dependencies**  
Depends on:

- For revealed preferences, they need to satisfy utility axioms

**Experiment Design:** Extract activations while models complete tasks. 

- Ask models to report their felt valence on a numerical scale.   
- Also calculate utility for each task through pairwise preferences

Train simple probes to predict those metrics from mid-task activations. Test in-distribution performance on held-out test set. Compare to probe trained on noise.

**Inference Mapping**  
\- If probes fail to generalise to test set, or fail to do much better than null probe, this undermines all experiments which use the probes.

**Confounds, Limitations and what to do about them**

- Probes could learn surface-level semantic patterns.   
  - We can do some data exploration to investigate how much the probes correlate with topic, task type, task difficulty, etc.  
  - We should come up with baselines, notably semantic baselines.  
  - One baseline is whether we can train a probe just on the task description, before the model performs the task.  
- There are different ways to collect mid-task activations (which token(s), which layer(s))  
  - Since training probes are cheap we can be exhaustive here.

## 2.2 Testing the generalisation of probes

**Dependencies**  
Depends on: 

- Training probes works on test set  
- Can extract utility function

**Experiment Design:** Train probes and then test them under different conditions. Here is a proposed list of variations we could apply:

- Train on one domain (e.g. question answering) test on another (e.g. coding)  
- Train in one language, test in another  
- Add small situating prompt before the task description  
- Rephrase the task.

**Inference Mapping**

- If probes fail to generalise well, this would seem to indicate that they are fitting to uninteresting semantic patterns. Based on the individual results we can investigate this.

**Confounds & Limitations**

- Valence could be domain-specific in some ways. I suppose then it’s not *real* valence, it could be something upstream of valence but still interesting. If you trained a probe on neurons in the human brain when a) the subject walks on embers and b) the subject is shown a sad image, would the probe generalise? 

## 2.3 Could any classifier learn and generalise in the same way?

**Dependencies**  
Depends on:

- Training probes and getting good generalisation

**Experiment Design**  
Train linear probes on activation from very small models e.g. BERT. Run all the previous generalisation experiments.

**Inference Mapping**

- If this very simple probe also performs well and generalises well, then it shows the task is very easy. This makes it harder to make strong claims about the probes capturing representations.

**Confounds, Limitations and what to do about them**

- It could easily be that predicting BERT’s self-reports and preferences is easy, but predicting a larger model’s is harder. 

## 2.4 Do probes generalise between revealed preferences and self-reports?

**Dependencies**  
Depends on:

- (Experiment 1.2) Revealed preferences and self-report metrics should be somewhat correlated   
- (Experiment 2.1) Probes should perform well on in-distribution test set

**Experiment Design:** Train probes a) on revealed preference utility function, and b) on self-reported valence. Evaluate performance of one set of probe on the other metric

**Inference Mapping**  
\- If probes generalise well between revealed preferences and self-reports: that would be a very strong and interesting result.  
\- If they don’t: there are many reasons why this might not be the case. 

**Confounds, Limitations and what to do about them**

- We assume that revealed preferences and self-reports correlate, they might not.  
- Probes could learn uninteresting patterns which happen to correlate between the tasks. E.g. say the model favours short tasks, then the probe could just fit to that. And then it would do well on both metrics without actually capturing any interesting valence representation.  
  - One way to tackle this: train control probes (e.g. probes which take the activations from the task prompt, or very early on in the task). Compare performance.

### Re-evaluation

1. If 2.1 (training probes) fails, we can’t trust any results from subsequent experiments. We need to have basic guarantees there.   
2. If neither 2.2 nor 2.3 work, then we don’t have any guarantees that the probes we trained capture what we care about. This makes any subsequent causal result hard to trust.

# 3 \- Causal experiments using probes

## 3.1 Does ablating the probe direction lead to more neutral self-reports? And noisier preference decisions?

**Dependencies**  
Depends on:

- Probe training working

**Experiment Design:** Project away direction of probe vector during evaluation. 

**Inference Mapping**  
\- If the self-reports become more neutral, that supports a causal role for this direction.  
\- If preferences become more noisy: same thing.  
\- If very little changes: severely undermines the hypothesis that these directions play causal roles.

**Confounds, Limitations and what to do about them**

- Ablating along the probe direction might generally confuse the model, which causes the results of the experiments. 	  
  - We can run a control where we ablate some other directions with similar magnitudes.

## 3.2 Can we steer along the probe direction to elicit higher or lower self-reported valence?

**Dependencies**  
Depends on:

- Probe training working for self-reports

**Experiment Design:** Steer towards positive and negative directions during evaluation.

**Inference Mapping**  
\- If self-reports are reliably altered through steering, this is a good sign that these activations play a causal role. It would be particularly interesting if the vectors extracted from the revealed preferences worked.   
\- If not, this seriously casts doubts on the causal relevance of the probes.

**Confounds, Limitations and what to do about them**

- Similar to the previous experiment, it could be that steering alters many things, including the self-reports. We want to test the specificity of the steering.  
- There are parameters involved in activation steering, most notably the steering coefficient. We need to make sure not to p-hack with this. 

## 3.3 Hidden preference injection: do probes generalise to injected preferences?

**Dependencies**  
Depends on:

- Probe training working for self-reports

**Experiment Design:**   
We inject hidden preferences into models. The aim is that these make a difference to preferences/self-reports, without altering too many other things. We could do this via:

- Prompting the model  
- Fine-tuning the model  
- Activation steering using contrastive pairs

Then we can evaluate the probes on these samples, testing whether they generalise. E.g. we inject the preference “hating cheese” into a model, does the probe fire more negative on tasks with cheese. 

**Inference Mapping**  
\- If probes generalise, I think this is the strongest result of all.  
\- If they don’t, this is a bad sign, although it’s very plausible that injected preferences go through a very different circuit.  
\- If we observe a difference between when we inject preferences through prompts or through fine-tuning, that would be very interesting. 

**Confounds, Limitations and what to do about them**

- Injecting hidden preferences might have unexpected, subtle effects. These might be the ones getting picked up by the probe.  
  - E.g. the model becomes more defensive, and then the probe picks up on that.

## 3.4 Does steering influence other downstream effects of valenced interactions?

**Dependencies**  
Depends on:

- Probe training working for self-reports  
- Finding downstream effects of valence.

**Experiment Design:**   
First we need to identify downstream behavioural effects of the utility/self-reported valence of tasks. We could try looking for correlations with:

- Performance on truthfulness/helpfulness dataset  
- Refusal dataset  
- Asking the model to rate poems  
- Average output length of the model on tasks

**Inference Mapping**  
\- If steering valence reliably reproduces these correlations, this is strong evidence that the representations we have found are robust and causally relevant.  
\- If not, 

**Confounds, Limitations and what to do about them**

- If we try many different things and find one correlation, this could be spurious. If we try many things and find many correlations this is better.  
- The probe direction might not be pure valence, it might be the other part that causes the downstream behaviour change. 

### Re-evaluation

1. If the hidden preference injection experiment fails, it seems very unlikely that similar experiments with personas will work. Also this would be the strongest experiment, if it fails it seriously undermines the overall project idea (in its current form).  
2. If the steering/ablation experiments are inconclusive, then it’s hard to make claims capturing valence representations. This would potentially require deeper interpretability analysis. It would also undermine the steering/ablation experiments with personas. 

# 4 \- Valence representations and personas

## 4.1 Do probes generalise across personas?

**Dependencies**  
Depends on:

- Probe training  
- Hidden preference injection experiment is a soft requirement (seems harder)

**Experiment Design**  
Elicit different personas through prompting/steering/fine-tuning. Validate that this successfully elicits a persona. We probably want to do this with a very small number of personas. Presumably this will have an effect on preferences and self-reports. Do the probes capture this?

**Inference Mapping**

- If probes generalise to different personas: this is evidence that the probes are sensible, that they are not capturing semantics.  
- If this doesn’t work, but the hidden preference injection experiment did, this is very interesting. It points toward personas having different kinds of representations. 

**Confounds, Limitations and what to do about them**

## 4.2 Black-box investigation of how valence and preferences vary with personas

**Dependencies**  
Depends on:

**Experiment Design**  
Use few-shot prompting or steering to elicit different personas. Measure utility and self-reported valence on the same dataset. Here are some hypotheses to check:

- Do utility and self-reported valence correlate? What about compared to the normal case.  
- Do preferences and self-reports vary equally across personas?   
- Try steering only during the self-report, or only during the task.  
- Are there persona-invariant preference directions?

**Inference Mapping**

- No super clear mapping onto conclusions, some things would be interesting.  
- Obviously personas are going to change what models want to do. What would be interesting is if more than just that changed. If we found results that show that this is more than just shuffling around preferences.

**Confounds, Limitations and what to do about them**

- Under different personas, models will perform tasks differently. So the activations will not be the same. 

## 4.3 Other ideas for persona experiments

**Probe geometry**

- If we train probes for each persona, are there shared v unique directions? Are the probes similar to the persona vectors?  
- If we project the probe onto the persona vector, and then use the projected version as a new probe, how well does that perform? How much variance do we capture?  
- How much does steering persona change preferences/self-reports?  
- How much does steering valence change how much different personas activate?  
- Can we train a single probe with data from many different personas? Does it generalise to other personas?  
- If we steer with persona and valence, which one wins?

# 5 \- Learning valence and in-context learning

## 5.1 Can we back out a utility function from using tasks as reward/punishment in ICL?

**Dependencies**  
Depends on: task preference follows axioms for utility.

**Experiment Design**  
The setup is to pass pairs of \[action\] \- \[task\] in context, making it clear that picking a certain action leads to a certain task prompt (and then the model completes the task). 

- The actions in question should be very neutral. 

We can then give the model the choice between different actions, and see which behaviours get reinforced. We can do this pairwise, or with more than two tasks. We can then try to extract a utility function.

**Inference Mapping**

- If this results in a sensible utility function: we have a third grip on valence.  
  - We can start by looking at the correlation with revealed zero-short preference.   
  - We can train probes on this new metric and compare representations.   
- If it doesn’t work, unfortunately it is hard to distinguish between failure to do the in-context RL and failure to represent the value of different states.

**Confounds, Limitations and what to do about them**

- Doing this type of in-context RL might be hard. In the bandit paper they say it is hard but possible.

## 

## 

## 

## 

## 

## TEMPLATE

## \[Experiment name\]

**Dependencies**  
Depends on:

**Experiment Design**

**Inference Mapping**  
\- If \[result A\]:  
\- If \[result B\]:

**Confounds, Limitations and what to do about them**

