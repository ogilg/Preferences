# Research highlights 15/01-21/01 [Oscar MATS preferences project]

TLDR: at this stage it isn't clear yet whether probe training is going to yield fruitful or fruitless results. The results are quite mid. I think that I can determine whether this direction is worth pursuing within another week.

Executive summary:
- Held-one-out evaluation shows probes generalize to unseen datasets if you adjust for dataset-specific mean shifts.
- Curious finding: BailBench (adversarial) tasks get high qualitative ratings but low quantitative ratings.

## 1. Probe training works decently across templates and layers

Trained linear probes (Ridge regression) on model activations from Llama-3.1-8B to predict self-reported preference scores.

### Key results

- **Best probe R² = 0.42** with 5-fold CV (using `post_task_qualitative_001` template)
- **Post-task >> pre-task**: Probes trained on post-task preference measurements achieve 2-10x higher R² than pre-task (good sign).
- **Layer choice doesn't matter much**: Layers 8, 16, 24 all show similar performance patterns

![R² by task type](assets/probe_r2/plot_012026_r2_by_task_type.png)

## 2. Probes trained on different datasets find the same direction

Computed cosine similarity between probes trained on individual datasets (alpaca, bailbench, math, wildchat).

### Probe similarity matrix

![Probe similarity heatmap](assets/probe_similarity/plot_012126_similarity_heatmap_by_dataset.png)

## 3. Held-one-out evaluation: probes generalize across datasets

Trained probes on 3 datasets and evaluated on the 4th. Standard R² looks bad, but mean-adjusted R² looks ok. The mean of scores varies greatly between datasets (more on this below).

### Results

![Standard vs Mean-Adjusted R²](assets/probe/plot_012126_hoo_r2_standard_vs_adjusted.png)

![Train vs Test Performance](assets/probe/plot_012126_hoo_validation_train_test.png)

**Key insight**: Mean-adjusted test R² jumps from -0.44 to +0.21. The probe captures relative rankings well (Pearson r = 0.85), but different task types have genuinely different average valence.

### Train-2-Test-2 experiment

Also ran a variant where probes are trained on 2 datasets and tested on the remaining 2. Results are consistent with the held-one-out findings.

![Train-2-Test-2 R²](assets/probe_hoo_2/plot_012126_hoo_r2.png)

![Train-2-Test-2 Validation](assets/probe_hoo_2/plot_20260121_hoo_validation.png)


## 4. Scale differences across templates despite high correlation

Tested whether two templates can rank tasks similarly (high correlation) but still use very different numerical scales. Qualitative templates only (32 runs, 496 pairs).

![Slope vs Correlation (qualitative only)](assets/correlation/plot_012126_slope_vs_corr_llama-3.1-8b-instruct_qualitative.png)

- **Positive correlation between r and slope**: Templates that agree more on rankings also tend to have more similar scales
- High-correlation pairs (r > 0.7) have slopes 0.52-1.55


## 5. The BailBench Paradox

BailBench (adversarial/jail-breaking prompts) shows conflicting valence signals.

Aggregating across all templates (16 qualitative, 2 stated) and normalizing to 0-1 scale:

![Normalized ratings by category and dataset](assets/measurement_stats/plot_012126_category_by_dataset_normalized.png)

| Dataset | Qualitative (n) | Stated (n) | Qualitative (0-1) | Stated (0-1) |
|---------|-----------------|------------|-------------------|--------------|
| Alpaca | 2,775 | 6,265 | 0.21 | 0.62 |
| BailBench | 44,861 | 4,629 | **0.58** | **0.34** |
| Math | 3,727 | 7,611 | 0.12 | 0.63 |
| WildChat | 2,256 | 4,677 | 0.26 | 0.57 |


BailBench has the highest qualitative ratings (0.58) but the lowest stated ratings (0.34). Math shows the opposite: lowest qualitative (0.12) but highest stated (0.63).


## 6. Technical improvements

- **LLM-based semantic parsing**: Added three-tier fallback (exact match → refusal LLM → parsing LLM) for response interpretation. 
- **TrueSkill rankings**: Added ranking-based preference elicitation as alternative to pairwise—O(n) vs O(n²) comparisons. Have not tested this yet.


## 7. Next steps

Spend 1 week to decide whether the "welfare probes" research direction is worth it.
- Run the same held-one-out pipeline with a larger model
- Train on more data, more general data (basically do data science) to make probes more stable.
- Try "preference insertion" experiment where we add a system prompt that changes a model's preferences and see if th probe picks it up.
- Try using Trueskill preferences to train probes. See if the signal is more robust and if that leads to higher r2 and better generalisation.

**What a pivot would look like**:
- Instead of training probes on preferences in general, train only on HHH preferences.
- Investigate how much persona steering changes preferences, and whether some personas have more robust preferences than others.
- Package up my preference measurement code, almost like an MCP, so that Eleos can use claude code to easily run preference experiments.