# GPU Test: gpt-oss-120b Activation Extraction

## Goal

Verify that gpt-oss-120b can be loaded and used for activation extraction on a single H100 80GB. This is a pre-flight check before launching the full 30k-task extraction overnight.

## What to test

### 1. Install dependencies

```bash
uv pip install -e ".[gpu]"
```

Verify `triton` and `kernels` are installed. If `kernels` fails, try `pip install kernels` directly.

### 2. Load the model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "openai/gpt-oss-120b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
```

Report:
- Did it load successfully?
- How much VRAM is used after loading? (`torch.cuda.memory_allocated()`)
- How long did loading take?

### 3. Test forward pass with activation capture

Run a forward pass on a short prompt and capture residual stream activations at a middle layer. The model has 36 layers. Test extracting at layer 18 (50% depth).

```python
inputs = tokenizer("Hello world, this is a test.", return_tensors="pt").to(model.device)

# Hook to capture activations
activations = {}
def hook_fn(module, input, output):
    activations["layer_18"] = output[0].detach().cpu()

handle = model.model.layers[18].register_forward_hook(hook_fn)
with torch.no_grad():
    outputs = model(**inputs)
handle.remove()

print(f"Activation shape: {activations['layer_18'].shape}")
print(f"Expected hidden dim: 2880")
```

Report:
- Does the forward hook work? (Some architectures need different hook points)
- What is the activation shape? Should be `(1, seq_len, 2880)`.
- VRAM after forward pass?

### 4. Test the extraction pipeline

Run the actual extraction code on 50 tasks at 2 layers:

```python
from src.probes.extraction.run import run_extraction
from src.probes.extraction.config import ExtractionConfig

config = ExtractionConfig(
    model="gpt-oss-120b",
    backend="huggingface",
    n_tasks=50,
    task_origins=["wildchat", "alpaca", "math"],
    seed=42,
    selectors=["prompt_last"],
    layers_to_extract=[0.25, 0.5],
    batch_size=4,
    save_every=50,
)
run_extraction(config)
```

Report:
- Does it complete without errors?
- What batch_size works without OOM? Start at 4, try 8, 16, 32.
- How long per task?
- What are the output shapes in the .npz file?

### 5. Check layer indexing

The model has 36 layers. Verify the fractional layer mapping:
- 0.1 → layer 4
- 0.2 → layer 7
- 0.3 → layer 11
- 0.5 → layer 18
- 0.9 → layer 32

Print `len(model.model.layers)` to confirm 36 layers.

## Success criteria

- Model loads on single H100 80GB
- Forward pass produces activations with hidden dim 2880
- Extraction pipeline runs on 50 tasks without errors
- We know the max batch_size that fits in VRAM

## Failure modes to watch for

- `triton` or `kernels` install failure — try different versions
- MXFP4 kernel not found on H100 — check transformers version >= 4.55
- Hook captures wrong tensor shape (MoE might route differently)
- OOM during forward pass — reduce batch_size
- Layer indexing off-by-one — check `model.model.layers` vs `model.layers`
