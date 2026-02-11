# Pod smoke test

## Goal

Verify the pod environment is correctly set up for research work. This is a minimal end-to-end test.

## Tasks

1. Verify Python environment: import key packages (numpy, sklearn, torch, nnsight, sentence_transformers)
2. Verify data files exist:
   - `activations/gemma_3_27b/activations_prompt_last.npz` — load and print shape
   - `results/experiments/gemma3_3k_run2/` — list contents
   - `src/analysis/topic_classification/output/topics_v2.json` — load and print number of entries
3. Run the test suite: `pytest tests/ -m "not api" --tb=short -q`
4. Log all results (pass/fail, shapes, counts) to the research log.

If any data files are missing, note which ones and still complete the other checks.
