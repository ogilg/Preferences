import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Memory:", round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1), "GB")

from src.probes.extraction import run
print("Extraction module: ok")
