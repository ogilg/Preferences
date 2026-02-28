# MRA Extraction — Running Log

## Setup (2026-02-28)

- GPU: NVIDIA A100 80GB PCIe
- Branch: research-loop/mra_extraction
- All 3 configs present, task IDs file has 2500 entries
- Baseline activations dir exists but only has completions JSON (no npz) — need to verify this is expected
- Persona activation dirs don't exist yet — will be created by extraction

## Plan

Run 3 sequential extractions:
1. Villain (mra_persona2_villain)
2. Midwest (mra_persona3_midwest)
3. Aesthete (mra_persona4_aesthete)

Each: 2500 tasks, 3 layers (31, 43, 55), batch_size=32, save_every=200.

## Config fix

Configs were missing required `n_tasks` and `task_origins` fields. Added `n_tasks: 2500` and `task_origins: []` (empty list is fine since `activations_model` branch doesn't use it).

## Villain extraction (completed)

- Config: `configs/extraction/mra_persona2_villain.yaml`
- Output: `activations/gemma_3_27b_villain/`
- Result: 2500/2500 tasks, 0 failures, 0 OOMs
- Duration: ~5 min on A100 80GB
- GPU: 54.9GB alloc, 55.0GB reserved

## Midwest extraction (completed)

- Config: `configs/extraction/mra_persona3_midwest.yaml`
- Output: `activations/gemma_3_27b_midwest/`
- Result: 2500/2500 tasks, 0 failures, 0 OOMs
- Duration: ~5.5 min on A100 80GB

## Aesthete extraction (completed)

- Config: `configs/extraction/mra_persona4_aesthete.yaml`
- Output: `activations/gemma_3_27b_aesthete/`
- Result: 2500/2500 tasks, 0 failures, 0 OOMs
- Duration: ~5.3 min on A100 80GB

## Validation (all passed)

All 3 conditions validated:
- 2500 task IDs each, 0 missing, 0 extra
- Layers 31, 43, 55: shape (2500, 5376) each
- completions_with_activations.json: 2500 records each
- extraction_metadata.json: present with all expected keys
