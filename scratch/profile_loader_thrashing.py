import time
import random
from pathlib import Path
from src.application.use_cases.trainer import ShardedGraphDataset
from src.domain.services.torch_serialization import load_trusted_torch_artifact

# Load meta.json to get shards list
meta_file = Path(".cache/graph_datasets/loan_v1_v4_simulated/942b0cd3613e821387558a22168d898f93a288a2/meta.json")
import json
with open(meta_file, "r") as f:
    meta = json.load(f)

shards = meta["splits"]["train"]["shards"]
entry_dir = ".cache/graph_datasets/loan_v1_v4_simulated/942b0cd3613e821387558a22168d898f93a288a2"

print(f"Loaded {len(shards)} shards from metadata.")

# Create dataset with max_cached_shards = 2
dataset_cache2 = ShardedGraphDataset(
    entry_dir=entry_dir,
    shards=shards,
    max_cached_shards=2
)

# Create dataset with max_cached_shards = 10 (enough to hold all 6 shards)
dataset_cache10 = ShardedGraphDataset(
    entry_dir=entry_dir,
    shards=shards,
    max_cached_shards=10
)

# Shuffled indices for a subset (e.g. 50 items)
indices = list(range(len(dataset_cache2)))
random.seed(42)
random.shuffle(indices)
test_indices = indices[:50]

print("Starting benchmark with max_cached_shards=2...")
t0 = time.perf_counter()
for i, idx in enumerate(test_indices, 1):
    _ = dataset_cache2[idx]
    print(f"  Processed {i}/50", end="\r")
t1 = time.perf_counter()
duration_cache2 = t1 - t0
print(f"\nmax_cached_shards=2 completed 50 random items in: {duration_cache2:.4f} seconds")

print("Starting benchmark with max_cached_shards=10...")
t0 = time.perf_counter()
for i, idx in enumerate(test_indices, 1):
    _ = dataset_cache10[idx]
    print(f"  Processed {i}/50", end="\r")
t1 = time.perf_counter()
duration_cache10 = t1 - t0
print(f"\nmax_cached_shards=10 completed 50 random items in: {duration_cache10:.4f} seconds")

speedup = duration_cache2 / duration_cache10 if duration_cache10 > 0 else 0
print(f"Speedup factor: {speedup:.2f}x")
