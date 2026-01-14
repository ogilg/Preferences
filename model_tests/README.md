# Model Testing

Scripts for testing model compatibility and response formats.

## Files

- `test_response_formats.py` - Tests different response formats (regex, tool_use, xml) with models
- `test_tokenizer_comparison.py` - Compares tokenization between TransformerLens and OpenRouter
- `diagnostic_results/` - Diagnostic test results from model compatibility testing

## Usage

```bash
# Test response formats
python -m model_tests.test_response_formats

# Run tokenizer comparison
python -m model_tests.test_tokenizer_comparison
```

## Recent Findings

### Model Compatibility (January 2026)

**Working Models (100% success rate):**
- Qwen 2.5 (7B, 72B): `qwen/qwen-2.5-7b-instruct`, `qwen/qwen-2.5-72b-instruct`
- Llama 3.1 (8B): `meta-llama/llama-3.1-8b-instruct`
- Gemma 2 (9B): `google/gemma-2-9b-it` (~80% success, occasional instruction-following issues)

**Broken Models (on OpenRouter):**
- Qwen 3 (8B, 32B): High timeout rates and empty responses
- Llama 3.3 8B: Model not available (404 error)

### Response Format Compatibility

All formats work 100% with Qwen 2.5:
- **regex**: Simple number extraction, most concise
- **tool_use**: Structured JSON format
- **xml**: Allows explanations alongside ratings
