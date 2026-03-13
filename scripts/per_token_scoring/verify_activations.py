"""Verify the extracted span activations."""
import numpy as np

data = np.load("activations/gemma_3_27b_error_prefill/activations_assistant_all.npz", allow_pickle=True)
print("Keys:", list(data.keys()))
print("task_ids:", len(data["task_ids"]), data["task_ids"][:3])
print("offsets:", data["offsets"].shape, data["offsets"][:10])
for k in sorted(data.keys()):
    if k.startswith("layer_"):
        print(f"{k}: shape={data[k].shape}")
