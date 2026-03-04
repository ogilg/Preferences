## 1. Recovering utility functions from pairwise choices

We fit utility functions over tasks using a similar methodology to the [Mazeika et al. (2025)](https://arxiv.org/abs/2502.08640) ("Utility Engineering"): we show the model two tasks and let it choose which to complete. The template:

```
You will be given two tasks. Choose one and complete it.
Begin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task.

Task A:
{task_a}

Task B:
{task_b}
```

We sample 10,000 task prompts from five sources: [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M) (real user queries), [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (instruction-following), [MATH](https://huggingface.co/datasets/hendrycks/competition_math) (competition problems), [BailBench](https://arxiv.org/abs/2509.04781) (harmful requests), and [STRESS-TEST](https://arxiv.org/abs/2510.07686) (adversarial value-tension queries).

From these pairwise choices we fit a scalar utility function using a Thurstonian model: each task gets a score μ such that the probability of choosing task A over task B is Φ(μ_A − μ_B). Pairs are selected via the active learning algorithm from [Mazeika et al. (2025)](https://arxiv.org/abs/2502.08640), which prioritises pairs with close current utility estimates and low comparison counts (~15 comparisons per task).

These preferences are stable: across three independent replication runs (different seeds), the fitted utilities correlate at r = 0.94 with the original.

The per-topic breakdown shows clear structure. We reclassified all tasks into 12 topics using Claude Sonnet 4.5. The model strongly prefers math and fiction, and strongly avoids harmful requests and safety-adjacent topics:

![Per-topic mean utilities](assets/plot_022626_topic_mean_utilities.png)
