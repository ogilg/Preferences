import numpy as np
for persona in ["gemma_3_27b", "gemma_3_27b_villain", "gemma_3_27b_aesthete", "gemma_3_27b_midwest"]:
    path = f"activations/{persona}/activations_prompt_last.npz"
    d = np.load(path)
    layers = sorted([k for k in d.keys() if k.startswith("layer_")], key=lambda x: int(x.split("_")[1]))
    print(f"{persona}: {layers}")
