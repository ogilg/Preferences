"""Download gemma-3-27b-it model from HuggingFace."""
from dotenv import load_dotenv
import os

load_dotenv()

from huggingface_hub import snapshot_download

HF_TOKEN = os.environ["HF_TOKEN"]

print("Downloading google/gemma-3-27b-it...")
path = snapshot_download(
    "google/gemma-3-27b-it",
    token=HF_TOKEN,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)
print(f"Downloaded to: {path}")
