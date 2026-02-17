import numpy as np
data = np.load("activations/gemma_2_27b_base/activations_prompt_last.npz", allow_pickle=True)
print("Keys:", list(data.keys()))
for k in data.keys():
    print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")
