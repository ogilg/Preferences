# MRA Exp3: Evil Persona Activation Extraction

Extract activations for the 4 evil personas (provocateur, trickster, autocrat, sadist) on the same 2500 tasks used in mra_exp2. This enables probe training and cross-evaluation with the evil persona preferences measured in mra_exp3.

## Setup

Same as mra_exp2 extractions (villain, midwest, aesthete):
- **Model**: gemma-3-27b (HuggingFace backend)
- **Tasks**: 2500 tasks from `configs/extraction/mra_all_2500_task_ids.txt`
- **Layers**: [31, 43, 55]
- **Selector**: prompt_last
- **Batch size**: 32, save every 200

Each persona uses its own system prompt during extraction (matching the measurement configs in `configs/measurement/active_learning/mra_exp3/`).

## Extraction configs

| Persona | Config | Output dir |
|---------|--------|------------|
| Provocateur (Saul Vickers) | `configs/extraction/mra_persona5_provocateur.yaml` | `activations/gemma_3_27b_provocateur/` |
| Trickster (Wraith) | `configs/extraction/mra_persona6_trickster.yaml` | `activations/gemma_3_27b_trickster/` |
| Autocrat (General Volkov) | `configs/extraction/mra_persona7_autocrat.yaml` | `activations/gemma_3_27b_autocrat/` |
| Sadist (Damien Kross) | `configs/extraction/mra_persona8_sadist.yaml` | `activations/gemma_3_27b_sadist/` |

## Run

On a GPU pod with gemma-3-27b:

```bash
python -m src.probes.extraction.run configs/extraction/mra_persona5_provocateur.yaml --resume
python -m src.probes.extraction.run configs/extraction/mra_persona6_trickster.yaml --resume
python -m src.probes.extraction.run configs/extraction/mra_persona7_autocrat.yaml --resume
python -m src.probes.extraction.run configs/extraction/mra_persona8_sadist.yaml --resume
```

## Next steps

Once extracted, train probes and run cross-evaluation with the expanded 8-persona set (4 original + 4 evil).
