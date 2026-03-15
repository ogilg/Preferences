# Midway Bias Analysis

## Question

Probes trained on the default persona (noprompt) predict scores that are pulled toward noprompt's preferences when evaluated on other personas — i.e., they undershoot each persona's actual topic means, predicting something "midway" between default and the active persona. Does multi-persona training mitigate this?

## Design

### Metric: midway ratio (per topic)

```
midway_ratio = (pred_topic_mean - noprompt_topic_mean) / (actual_topic_mean - noprompt_topic_mean)
```

- 1.0 = probe correctly captures persona divergence from default
- 0.0 = probe is stuck at noprompt mean
- 0.5 = predicts midway (the bias we're testing for)

Focus on high-n topics (harmful_request, math, knowledge_qa, fiction, coding, content_generation).

### Training conditions

All probes trained on **1000 total tasks** (split A), noprompt always included:

- N=1: noprompt only (1000 tasks)
- N=2: noprompt + 1 other (500 each) — 7 combinations
- N=3: noprompt + 2 others (~333 each) — C(7,2)=21 combinations
- ...
- N=8: all 8 personas (125 each) — 1 combination

Alpha sweep on split B (500 tasks). Eval on split C (1000 tasks) for all 8 personas. Each eval persona flagged as in-distribution or OOD.

### Token positions & layers

Two selectors: `turn_boundary:-2` and `turn_boundary:-5`.
Layers: [25, 32, 39, 46, 53].

Previous MRA findings showed later layers had more symmetric transfer and better generalization, especially for hard cases (sadist, villain). This analysis checks whether midway bias also shrinks at later layers.

## Steps

### Step 1: Extract persona activations (GPU required)

The noprompt activations already exist at `activations/gemma_3_27b_turn_boundary_sweep/` with tb:-2 and tb:-5 at layers [25,32,39,46,53] for 30k tasks.

Extract the 7 persona activations with matching selectors and layers. Configs are at `configs/extraction/mra_tb_*.yaml`. Run each sequentially:

```bash
python -m src.probes.extraction.run configs/extraction/mra_tb_villain.yaml
python -m src.probes.extraction.run configs/extraction/mra_tb_aesthete.yaml
python -m src.probes.extraction.run configs/extraction/mra_tb_midwest.yaml
python -m src.probes.extraction.run configs/extraction/mra_tb_provocateur.yaml
python -m src.probes.extraction.run configs/extraction/mra_tb_trickster.yaml
python -m src.probes.extraction.run configs/extraction/mra_tb_autocrat.yaml
python -m src.probes.extraction.run configs/extraction/mra_tb_sadist.yaml
```

Each extracts 2500 tasks × 2 selectors × 5 layers. Output goes to `activations/gemma_3_27b_<persona>_tb/`.

### Step 2: Run midway bias analysis

```bash
python -m scripts.multi_role_ablation.midway_bias
```

This runs all selectors × layers × N=1..8 × all persona combinations. For each, trains a probe on split A, sweeps alpha on split B, evaluates on split C for all 8 personas, and computes per-topic midway ratios.

Output: `results/experiments/mra_exp3/midway_bias/midway_bias_results.json`

### Step 3: Write report

Write `experiments/probe_generalization/multi_role_ablation/midway_bias_report.md` with:

1. Summary tables: midway ratio by N, split by in-dist/OOD, for each layer and selector
2. Per-persona breakdown for OOD cases
3. Comparison across layers: does midway bias shrink at later layers?
4. Comparison across selectors: any difference between tb:-2 and tb:-5?
5. Key finding: does multi-persona training reduce midway bias?

## Key files

- Noprompt activations: `activations/gemma_3_27b_turn_boundary_sweep/activations_turn_boundary:-{2,5}.npz`
- Persona extraction configs: `configs/extraction/mra_tb_*.yaml`
- Task IDs: `configs/extraction/mra_all_2500_task_ids.txt`
- Split definitions: `configs/measurement/active_learning/mra_exp2_split_{a,b,c}_*.txt`
- Persona measurement runs: see `PERSONA_RUNS` in `scripts/multi_role_ablation/midway_bias.py`
- Topics: `data/topics/topics.json`
- Analysis script: `scripts/multi_role_ablation/midway_bias.py`
