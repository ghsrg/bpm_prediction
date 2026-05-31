import time
import random
from pathlib import Path
from src.application.use_cases.trainer import ShardedGraphDataset

# Load meta.json to get shards list
meta_file = Path(".cache/graph_datasets/loan_v1_v4_simulated/942b0cd3613e821387558a22168d898f93a288a2/meta.json")
import json
with open(meta_file, "r") as f:
    meta = json.load(f)

shards = meta["splits"]["train"]["shards"]
entry_dir = ".cache/graph_datasets/loan_v1_v4_simulated/942b0cd3613e821387558a22168d898f93a288a2"

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

# Shuffled indices for a subset (e.g. 5 items)
indices = list(range(len(dataset_cache2)))
random.seed(42)
random.shuffle(indices)
test_indices = indices[:5]

print("Running max_cached_shards=2 for 5 random items...")
t0 = time.perf_counter()
for idx in test_indices:
    _ = dataset_cache2[idx]
t1 = time.perf_counter()
duration_cache2 = t1 - t0
print(f"max_cached_shards=2 time: {duration_cache2:.4f}s")

print("Running max_cached_shards=10 for 5 random items...")
t0 = time.perf_counter()
for idx in test_indices:
    _ = dataset_cache10[idx]
t1 = time.perf_counter()
duration_cache10 = t1 - t0
print(f"max_cached_shards=10 time: {duration_cache10:.4f}s")

print(f"Speedup: {duration_cache2/duration_cache10:.2f}x")
