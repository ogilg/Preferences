# Research highlights 22/01-29/01 [Oscar MATS preferences project]


## Comparing the stability of different methods to measure preferences. And on different models.

I care about how stable preferences are to when we resample. 

From a philosophical point of view, it is fine if preferences are noisy, as long as they are not uniformly random.

From a probe-training point of view, it is better if the preference measurements we use are stable. But even if they are noisy we can hopefully counter that by using many samples. 

So the instability is probably not going to kill the project. But it's generally true that more stability is better, and also I want to see how it scales with models. 

Here are the ways of measuring I want to compare:
- Current qualitative rating.
- Quantitative rating 
- Likud type scale with quantitative ratins that correspond to "agree/strongly agree".
- Quantitative rating but telling the model that "1 is equivalent to task T" etc.
- Binary choice preferences obtained through active learning.


The idea is to measure:
- for a given task, what does the distribution of tasks look like across scores. Compute aggregate statistics about how resonable this distribution is.
- For a corpus of tasks, how correlated are scores when we rerun the whole analysis.
- do this with a few different models, also a couple cot models. compare across models.

## Steering vector experiment follow-up

i found this weird result that steering tends to make the model say bad more.

1. Try this at a larger scale.
2. If it is still true, wait for the measurement analysis and run the same experiment with the new measurement + model setup.


## Concept vectors

Extract the vectors with difference of means. Test them in0distribution and then check how well they generalise.

## Do some analysis of Bailbench scores.

