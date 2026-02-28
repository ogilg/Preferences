import numpy as np

dirs = {
    "villain": "activations/gemma_3_27b_villain",
    "midwest": "activations/gemma_3_27b_midwest",
    "aesthete": "activations/gemma_3_27b_aesthete",
}

for name, dir_path in dirs.items():
    data = np.load(f"{dir_path}/activations_prompt_last.npz")
    for layer in [31, 43, 55]:
        arr = data[f"layer_{layer}"]
        nans = np.isnan(arr).sum()
        infs = np.isinf(arr).sum()
        norms = np.linalg.norm(arr, axis=1)
        print(f"{name} L{layer}: NaN={nans}, Inf={infs}, norm mean={norms.mean():.1f}, std={norms.std():.1f}, min={norms.min():.1f}, max={norms.max():.1f}")
