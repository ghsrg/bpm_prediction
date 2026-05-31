import time
import torch
from pathlib import Path
from src.domain.services.torch_serialization import load_trusted_torch_artifact

shard_path = Path(".cache/graph_datasets/loan_v1_v4_simulated/942b0cd3613e821387558a22168d898f93a288a2/train_shards/train_00001.pt")
print("Loading shard...")
t0 = time.perf_counter()
loaded = load_trusted_torch_artifact(shard_path, map_location="cpu")
t1 = time.perf_counter()
print(f"Loading shard took: {t1-t0:.4f}s")
print(f"Type: {type(loaded)}")
if isinstance(loaded, list):
    print(f"List length: {len(loaded)}")
elif isinstance(loaded, dict):
    print(f"Keys: {list(loaded.keys())}")
