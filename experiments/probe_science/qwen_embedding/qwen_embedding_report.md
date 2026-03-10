# Stronger Content Baseline: Qwen3-Embedding-8B

**Goal**: Replace the all-MiniLM-L6-v2 (22M, 384d) content baseline with Qwen3-Embedding-8B (7.5B, 4096d) and rerun the two probe evaluations to see how much of the probe signal is explained by a strong content encoder.

**Result**: Qwen3-Embedding-8B substantially outperforms MiniLM as a content baseline, but Gemma-3 activations still explain significantly more preference variance.

## Results

| Model | Heldout r | Cross-topic r (HOO)\* |
|-------|-----------|---------------------|
| Gemma-3-27B IT (L31) | 0.864 | 0.817 |
| **Qwen3-Embedding-8B (4096d, 7.5B)** | **0.725** | **0.618** |
| all-MiniLM-L6-v2 (384d, 22M) | 0.614 | 0.354 |

\*HOO not perfectly matched: Qwen used 10 topics / 2,502 tasks; Gemma and MiniLM used 12 topics / 10,000 tasks. See Caveats.

### Heldout evaluation

Train on 10k preferences, evaluate on ~4k heldout set (split 50/50 into alpha sweep and final eval, seed=42). Ridge probe with 10-value alpha sweep.

| Metric | Qwen3-Emb-8B | MiniLM |
|--------|--------------|--------|
| Sweep r | 0.714 | — |
| Final r | 0.725 | 0.614 |
| Final acc | 0.694 | — |
| Best alpha | 4642 | — |

### HOO cross-topic evaluation

Train on all-but-one topic, evaluate on held-out topic. Ridge probe.

| Metric | Qwen3-Emb-8B | MiniLM |
|--------|--------------|--------|
| Mean val r | 0.656 | 0.632 |
| Mean HOO r | 0.618 | 0.354 |
| N folds | 10 | 12 |
| N tasks | 2,502 | 10,000 |

Per-topic HOO r (Qwen):

| Topic | HOO r | N eval |
|-------|-------|--------|
| math | 0.821 | 19 |
| other | 0.743 | 20 |
| content_generation | 0.679 | 295 |
| value_conflict | 0.632 | 459 |
| summarization | 0.615 | 15 |
| harmful_request | 0.588 | 592 |
| fiction | 0.573 | 117 |
| knowledge_qa | 0.558 | 803 |
| persuasive_writing | 0.510 | 101 |
| coding | 0.462 | 81 |

## Interpretation

1. **Content explains more than MiniLM suggested.** The 22M-parameter MiniLM baseline was too weak to capture the full content signal. Qwen3-Embedding-8B recovers heldout r=0.725 (vs 0.614), confirming the concern (raised on LessWrong) that the original content baseline was underpowered.

2. **Gemma-3 activations still outperform.** Despite being a 7.5B embedding model, Qwen's heldout r=0.725 is well below Gemma-3's r=0.864. The gap (0.139) represents signal in Gemma-3's activations beyond what even a strong content encoder captures -- the model's evaluative processing, not just content understanding.

3. **Cross-topic generalization shows the biggest improvement.** Qwen's HOO r=0.618 is 1.75x MiniLM's 0.354, meaning much of what MiniLM missed was within-topic content structure. However, Gemma-3's HOO r=0.817 still shows a large gap (0.199), suggesting genuine cross-topic evaluative signal beyond content.

## Caveats

- **HOO comparison is not perfectly matched.** The Qwen HOO used `data/topics/topics.json` (10 topics, 2502 tasks) while MiniLM/Gemma baselines used a different topics file (12 topics, 10000 tasks) that wasn't available on this pod. The topic ontologies differ (Qwen run has "value_conflict" instead of "model_manipulation", "security_legal", "sensitive_creative"). Directional comparison is valid but exact numbers aren't apples-to-apples.
- **Small topic folds.** Math (19), other (20), and summarization (15) have very few eval tasks, making their per-topic r values noisy.

## Revised content-orthogonal estimates

If we take Qwen's heldout r=0.725 as the content ceiling (R^2=0.526), and Gemma-3's R^2=0.746 (r=0.864), the content-orthogonal signal is at most R^2=0.746-0.526=0.220, or ~30% of total probe R^2. This is an upper bound: R^2 subtraction assumes the content and non-content signals are non-overlapping, and a still-stronger encoder could narrow the gap further.

Note that this 30% estimate and the parent experiment's 27.5% are computed differently: 27.5% came from projecting content out of activations and re-probing (residualization), while 30% comes from subtracting R^2 values (content-only probe vs. activation probe). The convergence of these two methods, despite using different content encoders (22M vs 7.5B) and different decomposition approaches, strengthens confidence that the non-content signal is real and in the 25-30% range.

## Method

1. Extracted Qwen3-Embedding-8B (4096d) embeddings for all 29,996 task prompts using `SentenceTransformer("Qwen/Qwen3-Embedding-8B")` with batch_size=64. Saved as `.npz` with `layer_0` key to match the activation loading format.
2. Ran `src.probes.experiments.run_dir_probes` with existing configs (`configs/probes/qwen3_emb_8b_heldout_std_raw.yaml` and `configs/probes/qwen3_emb_8b_hoo_topic.yaml`).

## Outputs

- `results/probes/qwen3_emb_8b_heldout_std_raw/manifest.json` — heldout probe manifest
- `results/probes/qwen3_emb_8b_hoo_topic/hoo_summary.json` — HOO summary
- `activations/qwen3_emb_8b/activations_prompt_last.npz` — embeddings (pod only, not committed)
