# Phase 3 Running Log

## 2026-02-18: Setup and Measurement

- Read spec, phase 2 report, prompt enrichment report
- Environment: H100 80GB, vLLM 0.15.1 serving Gemma-3-27b-it locally
- Branch: research-loop/persona_ood_phase2 (reusing existing branch)
- Sampled 50 tasks stratified by topic (seed 42), excluding phase 2's 101 tasks
  - Topic distribution: knowledge_qa(11), harmful_request(10), math(10), content_generation(6), fiction(3), coding(2), persuasive_writing(2), model_manipulation(2), summarization(1), other(1), security_legal(1), sensitive_creative(1)
  - Saved to experiments/probe_generalization/persona_ood/phase3/core_tasks.json
- Created VLLMClient in src/models/openai_compatible.py, switched InferenceClient
- vLLM server started: google/gemma-3-27b-it, bfloat16, max_model_len=4096, port 8000
- Pilot test: 3/3 pairs successful with baseline prompt
- Started full measurement: 21 conditions × 4,900 pairs = 102,900 total pairs
  - MAX_CONCURRENT=150, temperature=0.7
- Measurement complete: all 21 conditions, ~196 obs/task, ~480s/condition, ~2.5hr total
- Killed vLLM, extracted activations for all 21 conditions at layers [31, 43, 55]
  - HuggingFace bfloat16 loading, batch_size=8, ~3 min total
  - Saved to activations/persona_ood_phase3/{condition}/activations_prompt_last.npz
- Analysis complete (demean/ridge_L31 primary):
  - Pooled r = 0.510 (p < 1e-67), up from 0.46 in phase 2
  - Original personas: r = 0.533, Enriched: r = 0.497
  - 18/20 personas pass r > 0.2 threshold
  - Sign agreement = 64.9%
  - Behavioral delta reliability = 0.990 (vs 0.64 in phase 2)
  - Disattenuated r = 0.511 (measurement noise no longer a factor)
  - Controls: shuffled r ≈ 0.0005, cross-persona = 0.281 vs matched = 0.546
